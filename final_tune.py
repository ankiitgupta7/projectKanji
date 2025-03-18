import os
import json
import torch
import argparse
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler
from peft import get_peft_model, LoraConfig
import torch.nn.functional as F
from tqdm import tqdm  # For progress bar

# 1️⃣ Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--json_path", type=str, default="kanji_dataset.json", help="Path to the dataset JSON file")
parser.add_argument("--images_dir", type=str, default="", help="Directory containing Kanji images")
parser.add_argument("--pretrained_model", type=str, default="CompVis/stable-diffusion-v1-4", help="Base Stable Diffusion model")
parser.add_argument("--output_dir", type=str, default="lora_kanji_unet", help="Directory to save fine-tuned model")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
args = parser.parse_args()

# 2️⃣ Custom Kanji Dataset
class KanjiDataset(Dataset):
    def __init__(self, json_path, images_dir, transform=None):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.kanji_keys = list(self.data.keys())
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.kanji_keys)

    def __getitem__(self, idx):
        kanji = self.kanji_keys[idx]
        entry = self.data[kanji]
        image_path = os.path.join(self.images_dir, entry["image"])

        # Convert grayscale (1-channel) to RGB (3-channel)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Use the first meaning as prompt
        prompt = entry["meanings"][0]
        return {"image": image, "prompt": prompt}

# 3️⃣ Image Transformations
# Increasing resolution to 256x256 for more detail (adjust as needed)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # Normalize for Stable Diffusion
])

# Load dataset
dataset = KanjiDataset(args.json_path, args.images_dir, transform)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=12)

# 4️⃣ Load Base Stable Diffusion Model and Scheduler
print("Loading base model and scheduler...")
pipeline = StableDiffusionPipeline.from_pretrained(args.pretrained_model)
unet = pipeline.unet  # Extract U-Net
vae = pipeline.vae.to("cuda" if torch.cuda.is_available() else "cpu")
text_encoder = pipeline.text_encoder.to("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = pipeline.tokenizer

# Load noise scheduler (e.g., DDPMScheduler)
noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")

# 5️⃣ LoRA Configuration & Injection
# Updated target modules for Stable Diffusion U-Net cross-attention layers
lora_config = LoraConfig(
    r=8,             # LoRA rank
    lora_alpha=32,   # Scaling factor
    lora_dropout=0.1,  # Dropout for regularization
    target_modules=[
        "down_blocks.*.attentions.*.transformer_blocks.*.attn2.to_k",
        "down_blocks.*.attentions.*.transformer_blocks.*.attn2.to_v",
        "up_blocks.*.attentions.*.transformer_blocks.*.attn2.to_k",
        "up_blocks.*.attentions.*.transformer_blocks.*.attn2.to_v",
        "mid_block.attentions.*.transformer_blocks.*.attn2.to_k",
        "mid_block.attentions.*.transformer_blocks.*.attn2.to_v"
    ],
    bias="none"
)
unet = get_peft_model(unet, lora_config)
device = "cuda" if torch.cuda.is_available() else "cpu"
unet.to(device)

# 6️⃣ Setup Training with Accelerator and Optimizer
precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
accelerator = Accelerator(mixed_precision=precision)
optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr)
unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

# 7️⃣ Training Loop with Classifier-Free Guidance
num_epochs = args.epochs
print(f"Starting fine-tuning for {num_epochs} epochs...")

for epoch in range(num_epochs):
    unet.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

    for batch in progress_bar:
        images = batch["image"].to(device)
        prompts = batch["prompt"]

        # Get conditional text embeddings
        with torch.no_grad():
            # Tokenize and encode prompts
            input_ids_cond = tokenizer(prompts, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
            encoder_hidden_states_cond = text_encoder(input_ids_cond)[0]
            
            # For classifier-free guidance, also get unconditional embeddings (empty prompt)
            empty_prompts = [""] * images.shape[0]
            input_ids_uncond = tokenizer(empty_prompts, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
            encoder_hidden_states_uncond = text_encoder(input_ids_uncond)[0]
            
            # Concatenate unconditional and conditional embeddings along batch dimension
            encoder_hidden_states = torch.cat([encoder_hidden_states_uncond, encoder_hidden_states_cond], dim=0)

        # Encode images into latent space using VAE (no gradient)
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample() * 0.18215

        # Sample random timesteps for each image in the batch
        batch_size = images.shape[0]
        t = torch.randint(0, noise_scheduler.num_train_timesteps, (batch_size,), device=device).long()

        # Sample noise and add noise to latents using the scheduler's method
        noise = torch.randn_like(latents)
        noisy_latents = noise_scheduler.add_noise(latents, noise, t)

        # Duplicate noisy latents and timesteps for unconditional & conditional inputs
        noisy_latents = torch.cat([noisy_latents, noisy_latents], dim=0)
        t = torch.cat([t, t], dim=0)

        # Forward pass through the UNet with the correct timestep and encoder hidden states
        model_output = unet(noisy_latents, t, encoder_hidden_states).sample
        # Split the outputs: first half for unconditional, second half for conditional
        model_output_uncond, model_output_cond = torch.chunk(model_output, 2, dim=0)

        # Compute loss only on the conditional branch
        loss = F.mse_loss(model_output_cond, noise)
        
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"Loss": f"{loss.item():.6f}"})

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.6f}")

# 8️⃣ Save Fine-Tuned Model
print("Saving fine-tuned LoRA model...")
unet.save_pretrained(args.output_dir)
print(f"Model saved to {args.output_dir}")
