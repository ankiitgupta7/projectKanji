import os
import json
import torch
import argparse
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from peft import get_peft_model, LoraConfig
import torch.nn.functional as F
from tqdm import tqdm  # ✅ Import tqdm for progress bar

# 1️⃣ Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--json_path", type=str, default="kanji_dataset.json", help="Path to the dataset JSON file")
parser.add_argument("--images_dir", type=str, default="", help="Directory containing Kanji images")
parser.add_argument("--pretrained_model", type=str, default="CompVis/stable-diffusion-v1-4", help="Base Stable Diffusion model")
parser.add_argument("--output_dir", type=str, default="lora_kanji_unet", help="Directory to save fine-tuned model")
parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")  # ✅ Reduced epochs for quick test
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")  # ✅ Lower batch size for CPU efficiency
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")  # ✅ Slightly increased learning rate for faster convergence
args = parser.parse_args()

# 2️⃣ Custom Kanji Dataset (✅ Limited to 500 images for testing)
class KanjiDataset(Dataset):
    def __init__(self, json_path, images_dir, transform=None):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.kanji_keys = list(self.data.keys())[:500]  # ✅ Only use first 500 images for a quick test
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.kanji_keys)

    def __getitem__(self, idx):
        kanji = self.kanji_keys[idx]
        entry = self.data[kanji]
        image_path = os.path.join(self.images_dir, entry["image"])

        # ✅ Convert grayscale (1-channel) to RGB (3-channel)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        prompt = entry["meanings"][0]  # Use first meaning as prompt
        return {"image": image, "prompt": prompt}

# 3️⃣ Image Transformations (Ensure 3-channel RGB images)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize for Stable Diffusion
])

# Load dataset
dataset = KanjiDataset(args.json_path, args.images_dir, transform)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)  # ✅ Lower num_workers for CPU

# 4️⃣ Load Base Stable Diffusion Model
print("Loading base model...")
pipeline = StableDiffusionPipeline.from_pretrained(args.pretrained_model)
unet = pipeline.unet  # Extract U-Net
vae = pipeline.vae.to("cuda" if torch.cuda.is_available() else "cpu")  # Move VAE to GPU (if available)
text_encoder = pipeline.text_encoder.to("cuda" if torch.cuda.is_available() else "cpu")  # Move text encoder to GPU (if available)
tokenizer = pipeline.tokenizer

# 5️⃣ LoRA Configuration (Correct Targets for SD U-Net)
lora_config = LoraConfig(
    r=8,  # LoRA rank
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1,  # Dropout for regularization
    target_modules=["to_q", "to_v"],  # ✅ Correct targets for Stable Diffusion
    # target_modules=["attn2"],
    bias="none"
)

# Inject LoRA layers into U-Net
unet = get_peft_model(unet, lora_config)
device = "cuda" if torch.cuda.is_available() else "cpu"
unet.to(device)

# 6️⃣ Setup Training (Use Accelerate for CPU Efficiency)
accelerator = Accelerator(mixed_precision="bf16" if torch.cuda.is_available() else "no")  # ✅ Use bf16 if possible
optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr)
unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

# 7️⃣ Training Loop with tqdm Progress Bar
num_epochs = args.epochs
print(f"Starting quick test training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    unet.train()
    total_loss = 0

    # ✅ Add tqdm progress bar for each epoch
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

    for batch in progress_bar:
        images = batch["image"].to(device)
        prompts = batch["prompt"]

        # ✅ Convert text prompts into CLIP embeddings
        with torch.no_grad():
            text_inputs = tokenizer(prompts, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
            encoder_hidden_states = text_encoder(text_inputs)[0]  # Get text embeddings

        # ✅ Convert images to latent space using VAE
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample() * 0.18215  # Scale for SD compatibility

        # ✅ Add noise in latent space (not RGB space)
        noise = torch.randn_like(latents)  
        noisy_latents = latents + noise  

        # ✅ Forward pass with correct `encoder_hidden_states`
        pred = unet(noisy_latents, timestep=0, encoder_hidden_states=encoder_hidden_states)  
        loss = F.mse_loss(pred.sample, noise)

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"Loss": f"{loss.item():.6f}"})  # ✅ Update tqdm with loss

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.6f}")

# 8️⃣ Save Fine-Tuned Model
print("Saving fine-tuned LoRA model...")
unet.save_pretrained(args.output_dir)
print(f"Model saved to {args.output_dir}")
