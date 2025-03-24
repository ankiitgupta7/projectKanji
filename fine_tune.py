import json, os, torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from torch.optim import AdamW
from tqdm.auto import tqdm

# Configurations
model_name = "CompVis/stable-diffusion-v1-4"
json_file = "kanji_dataset.json"
image_dir = "kanji_png_128"  # Folder containing your 128x128 images
output_dir = "kanji-model-finetuned"
num_epochs = 5
batch_size = 16
learning_rate = 5e-6
resolution = 128

device = "cuda" if torch.cuda.is_available() else "cpu"
accelerator = Accelerator()

# Load pretrained components from the base model
tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer", use_auth_token=True)
text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder", use_auth_token=True).to(device)
vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", use_auth_token=True).to(device)
unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet", use_auth_token=True).to(device)

# Freeze VAE and text encoder
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.train()

optimizer = AdamW(unet.parameters(), lr=learning_rate)

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((resolution, resolution)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# Define dataset
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

# Create dataset and dataloader
dataset = KanjiDataset(json_file, image_dir, tokenizer, transform)
print(f"📊 Loaded {len(dataset)} samples from {image_dir}")
if len(dataset) == 0:
    raise RuntimeError("No images found – check json paths and image_dir")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Prepare for distributed training if applicable
unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

# Training loop
for epoch in range(num_epochs):
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch in progress_bar:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)

        # Encode images into latent space using VAE
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

        progress_bar.set_postfix({"loss": loss.item()})

# Save individual components
unwrapped_unet = accelerator.unwrap_model(unet)
unwrapped_unet.save_pretrained(output_dir, safe_serialization=True)
text_encoder.save_pretrained(output_dir)
vae.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# --- Build and Save the Complete Pipeline ---
# Load a scheduler (using DDIM as an example)
scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler", use_auth_token=True)

# Construct the full pipeline with scheduler
pipe = StableDiffusionPipeline(
    unet=unwrapped_unet,
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    scheduler=scheduler,
    safety_checker=None,       # Optional: disable if not needed
    feature_extractor=None     # Optional: disable if not needed
)
pipe.save_pretrained(output_dir)
print(f"✅ Fine-tuning complete. Pipeline saved to {output_dir}.")
