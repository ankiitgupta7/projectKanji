import os, json, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from accelerate import Accelerator
from torch.optim import AdamW
from tqdm.auto import tqdm
import torch.nn.functional as F

from transformers import CLIPTextModel, CLIPTokenizer, get_scheduler
from diffusers import (
    StableDiffusionPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler  # We'll use this noise scheduler for real forward diffusion
)

# ---------- CONFIGURATION -------------
model_name = "CompVis/stable-diffusion-v1-4"
json_file = "kanji_dataset.json"
image_dir = "kanji_png_128"  # Folder containing your 128x128 PNG images
output_dir = "kanji-model-finetuned131"
num_epochs = 131
batch_size = 16
base_learning_rate = 5e-6
resolution = 128
sample_prompts = ["Elon Musk", "YouTube", "Skyscraper", "Neural Samurai"]

device = "cuda" if torch.cuda.is_available() else "cpu"
accelerator = Accelerator()

# ---------- LOAD PRETRAINED COMPONENTS -------------
tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer", use_auth_token=True)
text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder", use_auth_token=True)
vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", use_auth_token=True)
unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet", use_auth_token=True)

text_encoder = text_encoder.to(device)
vae = vae.to(device).eval()        # We'll keep it eval/frozen
unet = unet.to(device).train()

# ---------------- PARTIALLY UNFREEZE TEXT ENCODER ----------------
# Let's unfreeze final 2 blocks (resblocks.10 & .11) for better adaptation
for name, param in text_encoder.named_parameters():
    if not ("resblocks.10" in name or "resblocks.11" in name):
        param.requires_grad = False

# Freeze VAE
vae.requires_grad_(False)

# We'll train the UNet (and the final 2 blocks of text encoder)
unet.train()
text_encoder.train()

# ---------- OPTIMIZER & SCHEDULER -------------
params_to_optimize = list(unet.parameters()) + [p for p in text_encoder.parameters() if p.requires_grad]
optimizer = AdamW(params_to_optimize, lr=base_learning_rate)

# We'll do a real diffusion-based approach, so let's load the DDPMScheduler
noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler", use_auth_token=True)

# ---------- DATASET DEFINITION -------------
class KanjiDataset(Dataset):
    def __init__(self, json_path, img_root, tokenizer, transform):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.items = []
        for kanji, info in data.items():
            meanings = [m for m in info["meanings"] if isinstance(m, str)]
            if meanings:
                prompt = ", ".join(meanings)
                img_path = os.path.join(img_root, os.path.basename(info["image"]))
                if os.path.exists(img_path):
                    self.items.append({"prompt": prompt, "image": img_path})
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = Image.open(item["image"]).convert("RGB")
        pixel_values = self.transform(image)
        input_ids = self.tokenizer(
            item["prompt"], max_length=77, padding="max_length",
            truncation=True, return_tensors="pt"
        ).input_ids.squeeze(0)
        return {"pixel_values": pixel_values, "input_ids": input_ids}

transform = transforms.Compose([
    transforms.Resize((resolution, resolution)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

dataset = KanjiDataset(json_file, image_dir, tokenizer, transform)
print(f"📊 Loaded {len(dataset)} samples from {image_dir}")
if len(dataset) == 0:
    raise RuntimeError("No images found – check json paths and image_dir")

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# ---------- PREPARE MODELS WITH ACCELERATOR -------------
unet, text_encoder, optimizer, dataloader = accelerator.prepare(unet, text_encoder, optimizer, dataloader)
vae = vae.to(accelerator.device)

# ---------- LEARNING RATE SCHEDULER -------------
num_training_steps = len(dataloader) * num_epochs
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=int(0.05 * num_training_steps),  # 5% warmup
    num_training_steps=num_training_steps
)

# ---------- TRAINING LOOP -------------
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    unet.train()
    text_encoder.train()
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
    for step, batch in enumerate(progress_bar):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        
        with torch.no_grad():
            # Encode images to latents using VAE
            latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

        # Sample random timesteps
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                  (latents.size(0),), device=device).long()

        # Add noise to latents according to the noise scheduler
        noise = torch.randn_like(latents)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Forward pass
        encoder_hidden_states = text_encoder(input_ids)[0]
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        accelerator.backward(loss)

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(params_to_optimize, max_norm=1.0)

        optimizer.step()
        lr_scheduler.step()

        progress_bar.set_postfix({"loss": loss.item()})
    
    # Optional: generate sample images after each epoch
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet.eval()
        text_encoder.eval()
        
        temp_pipe = StableDiffusionPipeline(
            unet=accelerator.unwrap_model(unet),
            vae=vae,
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            scheduler=noise_scheduler,  # or load DDIM if you prefer
            safety_checker=None,
            feature_extractor=None,
        ).to(device)
        
        os.makedirs(f"samples131/epoch_{epoch+1}", exist_ok=True)
        for prompt in sample_prompts:
            image = temp_pipe(prompt, guidance_scale=7.5, num_inference_steps=50).images[0]
            # Convert to B&W threshold
            image = image.convert("L").point(lambda p: 0 if p < 128 else 255)
            image = image.resize((128, 128))
            out_path = f"samples131/epoch_{epoch+1}/{prompt.replace(' ','_')}.png"
            image.save(out_path)

    # Save checkpoint every 10 epochs
    if accelerator.is_main_process and (epoch + 1) % 10 == 0:
        print(f"Saving checkpoint at epoch {epoch+1}...")
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
        # Create pipeline for final checkpoint
        ckpt_pipe = StableDiffusionPipeline(
            unet=unwrapped_unet,
            vae=vae,
            text_encoder=unwrapped_text_encoder,
            tokenizer=tokenizer,
            scheduler=noise_scheduler,
            safety_checker=None,
            feature_extractor=None,
        )
        ckpt_dir = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}")
        ckpt_pipe.save_pretrained(ckpt_dir)

# ------------- SAVE THE FINAL PIPELINE -------------
if accelerator.is_main_process:
    print("Saving final fine-tuned pipeline...")
    unwrapped_unet = accelerator.unwrap_model(unet)
    unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
    final_pipe = StableDiffusionPipeline(
        unet=unwrapped_unet,
        vae=vae,
        text_encoder=unwrapped_text_encoder,
        tokenizer=tokenizer,
        scheduler=noise_scheduler,
        safety_checker=None,
        feature_extractor=None,
    )
    final_pipe.save_pretrained(output_dir)
    print(f"✅ Pipeline saved to {output_dir}")
