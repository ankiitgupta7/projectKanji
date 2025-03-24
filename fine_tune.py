import json, os, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from accelerate import Accelerator
from torch.optim import AdamW
from tqdm.auto import tqdm

from transformers import CLIPTextModel, CLIPTokenizer, get_scheduler
from diffusers import (
    StableDiffusionPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler
)

# ------------- CONFIGURATION -------------
model_name = "CompVis/stable-diffusion-v1-4"
json_file = "kanji_dataset.json"
image_dir = "kanji_png_128"  # Folder containing your 128x128 PNG images
output_dir = "kanji-model-finetuned"
num_epochs = 50           # Try 3-5 epochs for a test run
batch_size = 16
base_learning_rate = 5e-6  # Starting learning rate; you can adjust as needed
resolution = 128
sample_prompts = ["Elon Musk", "YouTube", "Skyscraper", "Neural Samurai"]

device = "cuda" if torch.cuda.is_available() else "cpu"
accelerator = Accelerator()

# ------------- LOAD PRETRAINED COMPONENTS -------------
tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer", use_auth_token=True)

# Load the text encoder, VAE, and UNet.
text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder", use_auth_token=True)
vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", use_auth_token=True)
unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet", use_auth_token=True)

# Move models to the device
text_encoder = text_encoder.to(device)
vae = vae.to(device)  # Ensure VAE is on GPU
unet = unet.to(device)

# (Optional) Partially unfreeze the text encoder (e.g., only the last block)
for name, param in text_encoder.named_parameters():
    if "transformer.resblocks.11" not in name:
        param.requires_grad = False

# Freeze VAE (it’s not being trained)
vae.requires_grad_(False)
unet.train()  # UNet will be trained

# ------------- OPTIMIZER & SCHEDULER -------------
# Combine parameters of UNet and the unfrozen part of the text encoder
params_to_optimize = list(unet.parameters()) + [p for p in text_encoder.parameters() if p.requires_grad]
optimizer = AdamW(params_to_optimize, lr=base_learning_rate)

# The learning rate scheduler will linearly decay from base_learning_rate to 0 over the training steps
# (We compute total training steps as number of batches * epochs)
# We assume dataloader will be created below.
 
# ------------- DATASET DEFINITION -------------
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
        return {
            "pixel_values": self.transform(image),
            "input_ids": self.tokenizer(
                item["prompt"], max_length=77, padding="max_length",
                truncation=True, return_tensors="pt"
            ).input_ids.squeeze(0),
        }

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

# ------------- PREPARE MODELS WITH ACCELERATOR -------------
# Note: We do NOT include vae because it is frozen, but ensure it's on device.
unet, text_encoder, optimizer, dataloader = accelerator.prepare(unet, text_encoder, optimizer, dataloader)

# ------------- LEARNING RATE SCHEDULER -------------
num_training_steps = len(dataloader) * num_epochs
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# ------------- TRAINING LOOP -------------
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    unet.train()
    text_encoder.train()
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
    for step, batch in enumerate(progress_bar):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        
        # Encode images to latent space using the VAE (in no_grad mode)
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

        noise = torch.randn_like(latents).to(device)
        timesteps = torch.randint(0, 1000, (latents.size(0),), device=device).long()
        noisy_latents = latents + noise
        
        encoder_hidden_states = text_encoder(input_ids)[0]
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()

        progress_bar.set_postfix({"loss": loss.item()})
    
    # Optional: Generate sample images after each epoch to track progress
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet.eval()
        text_encoder.eval()
        
        # Create a temporary pipeline for generating samples.
        # Note: We load the scheduler from the base model.
        scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler", use_auth_token=True)
        temp_pipe = StableDiffusionPipeline(
            unet=accelerator.unwrap_model(unet),
            vae=vae,
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
        ).to(device)
        
        os.makedirs(f"samples/epoch_{epoch+1}", exist_ok=True)
        for prompt in sample_prompts:
            image = temp_pipe(prompt, guidance_scale=7.5, num_inference_steps=50).images[0]
            # Post-process image: convert to grayscale and threshold to pure B&W
            image = image.convert("L").point(lambda p: 0 if p < 128 else 255)
            image = image.resize((128, 128), Image.LANCZOS)
            image.save(f"samples/epoch_{epoch+1}/{prompt.replace(' ','_')}.png")
    
# ------------- SAVE THE FINAL PIPELINE -------------
if accelerator.is_main_process:
    print("Saving final fine-tuned pipeline...")
    final_scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler", use_auth_token=True)
    final_pipe = StableDiffusionPipeline(
        unet=accelerator.unwrap_model(unet),
        vae=vae,
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        scheduler=final_scheduler,
        safety_checker=None,
        feature_extractor=None,
    )
    final_pipe.save_pretrained(output_dir)
    print(f"✅ Pipeline saved to {output_dir}")
