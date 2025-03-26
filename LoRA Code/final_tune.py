import os
import json
import torch
import argparse
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import get_peft_model, LoraConfig
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

"""
After you successfully train Stable Diffusion to produce Kanji images from short English descriptions,
please include many interesting (successful, failure, novel, funny, culturally meaningful) examples in your report.
In your report, document your procedure, assumptions, and decision/engineering choices made to produce your results.
This code saves checkpoints every 3 epochs, training curves, experimental configuration, and dataset details.
"""

# 1️⃣ Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--json_path", type=str, default="kanji_dataset.json", help="Path to the dataset JSON file")
parser.add_argument("--images_dir", type=str, default="", help="Directory containing Kanji images")
parser.add_argument("--pretrained_model", type=str, default="CompVis/stable-diffusion-v1-4", help="Base Stable Diffusion model")
parser.add_argument("--output_dir", type=str, default="lora_kanji_unet", help="Directory to save fine-tuned model, checkpoints, and logs")
parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")  # Default set to 30 epochs
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Initialize TensorBoard writer for logging training curves
writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "runs"))

# Save experimental configuration for reproducibility
with open(os.path.join(args.output_dir, "experiment_config.json"), "w") as f:
    json.dump(vars(args), f, indent=4)

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
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # Use the first meaning if available; otherwise, fallback to a default prompt
        prompt = entry["meanings"][0] if entry.get("meanings") and len(entry["meanings"]) > 0 else "kanji character"
        return {"image": image, "prompt": prompt}

# 3️⃣ Image Transformations (using 128x128 for memory efficiency; adjust if needed)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load dataset and create DataLoader
dataset = KanjiDataset(args.json_path, args.images_dir, transform)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

# 4️⃣ Load Base Stable Diffusion Model and Noise Scheduler
print("Loading base model and scheduler...")
pipeline = StableDiffusionPipeline.from_pretrained(args.pretrained_model)
unet = pipeline.unet  # Extract U-Net
vae = pipeline.vae.to("cuda" if torch.cuda.is_available() else "cpu")
text_encoder = pipeline.text_encoder.to("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = pipeline.tokenizer
noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")

# 5️⃣ LoRA Configuration & Injection with target modules "to_k" and "to_v"
lora_config = LoraConfig(
    r=8,             # LoRA rank
    lora_alpha=32,   # Scaling factor
    lora_dropout=0.1,  # Dropout for regularization
    target_modules=["to_k", "to_v"],
    bias="none"
)
unet = get_peft_model(unet, lora_config)
device = "cuda" if torch.cuda.is_available() else "cpu"
unet.to(device)

# 6️⃣ Setup Training with Accelerator, Optimizer, and Gradient Clipping
precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
accelerator = Accelerator(mixed_precision=precision)
optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr)
unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

# 7️⃣ Training Loop with Classifier-Free Guidance, Checkpoint Saving, and Logging
num_epochs = args.epochs
print(f"Starting fine-tuning for {num_epochs} epochs...")

global_step = 0
for epoch in range(num_epochs):
    unet.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

    for batch in progress_bar:
        images = batch["image"].to(device)
        prompts = batch["prompt"]

        # Get text embeddings for both conditional and unconditional (empty prompt) cases
        with torch.no_grad():
            input_ids_cond = tokenizer(prompts, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
            encoder_hidden_states_cond = text_encoder(input_ids_cond)[0]

            empty_prompts = [""] * images.shape[0]
            input_ids_uncond = tokenizer(empty_prompts, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
            encoder_hidden_states_uncond = text_encoder(input_ids_uncond)[0]

            encoder_hidden_states = torch.cat([encoder_hidden_states_uncond, encoder_hidden_states_cond], dim=0)

        # Encode images into latent space via VAE (scaled as in original code)
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample() * 0.18215

        # Sample a random timestep for each image
        batch_size = images.shape[0]
        t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()

        # Add noise to the latents using the noise scheduler
        noise = torch.randn_like(latents)
        noisy_latents = noise_scheduler.add_noise(latents, noise, t)

        # Duplicate for unconditional and conditional inputs
        noisy_latents = torch.cat([noisy_latents, noisy_latents], dim=0)
        t = torch.cat([t, t], dim=0)

        # Forward pass through UNet
        model_output = unet(noisy_latents, t, encoder_hidden_states).sample
        model_output_uncond, model_output_cond = torch.chunk(model_output, 2, dim=0)

        # Compute MSE loss on the conditional branch
        loss = F.mse_loss(model_output_cond, noise)
        
        optimizer.zero_grad()
        accelerator.backward(loss)
        torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        global_step += 1
        progress_bar.set_postfix({"Loss": f"{loss.item():.6f}"})

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.6f}")
    writer.add_scalar("Loss/train", avg_loss, epoch)

    # Save checkpoint every 3 epochs
    if (epoch + 1) % 3 == 0:
        unet = accelerator.unwrap_model(unet)
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        unet.save_pretrained(checkpoint_dir)
        print(f"Checkpoint saved to {checkpoint_dir}")
        # Re-wrap the model for continued training
        unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

# 8️⃣ Final Save of the Fine-Tuned Model
unet = accelerator.unwrap_model(unet)
print("Saving final fine-tuned LoRA model...")
unet.save_pretrained(args.output_dir)
print(f"Model saved to {args.output_dir}")

# Close the TensorBoard writer
writer.close()
