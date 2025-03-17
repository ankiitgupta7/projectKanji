# import os
# import json
# import csv
# import torch
# from torch import nn
# from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
# from tqdm import tqdm
# from PIL import Image

# # ----- Configuration -----
# MODEL_ID = "runwayml/stable-diffusion-v1-5"
# DATASET_FILE = "kanji_mini_dataset.json"
# CHECKPOINT_DIR = "checkpoints"
# SAMPLES_DIR = "samples"
# TRAINING_CURVE_FILE = "training_curve.csv"
# NUM_EPOCHS = 3
# LEARNING_RATE = 5e-5
# MAX_TOKENS = 77  # CLIP token limit

# # Create directories if they don't exist
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)
# os.makedirs(SAMPLES_DIR, exist_ok=True)

# # ----- Load Mini Dataset (Ensure Only 10 Samples) -----
# with open(DATASET_FILE, "r", encoding="utf-8") as f:
#     kanji_data = json.load(f)

# # Debugging: Print dataset size before training
# print(f"🚀 Loaded {len(kanji_data)} Kanji samples for training.")

# # ----- Load Model & Scheduler -----
# print("🚀 Loading model and scheduler...")
# pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, safety_checker=None)
# pipe.to("cpu")  # Use "cuda" for GPU
# pipe.scheduler = EulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

# # ----- Set Up Optimizer -----
# optimizer = torch.optim.Adam(pipe.unet.parameters(), lr=LEARNING_RATE)

# # ----- Training Loop -----
# print("🚀 Starting mini training run...")
# training_logs = []
# global_step = 0

# try:
#     for epoch in range(1, NUM_EPOCHS + 1):
#         epoch_losses = []
        
#         for kanji, data in tqdm(kanji_data.items(), desc=f"Epoch {epoch}"):
#             # Generate prompt (truncate to avoid CLIP token limit)
#             words = data["meanings"] + ["in Kanji style"]
#             prompt = " ".join(words)[:MAX_TOKENS]

#             # Generate image from prompt
#             image = pipe(prompt).images[0]

#             # Simulate training with a dummy loss
#             loss = torch.randn(1, requires_grad=True)
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()

#             # Log loss
#             loss_value = loss.item()
#             epoch_losses.append(loss_value)
#             training_logs.append({"epoch": epoch, "step": global_step, "loss": loss_value})
#             global_step += 1

#         # Save model checkpoint after each epoch
#         checkpoint_path = os.path.join(CHECKPOINT_DIR, f"unet_epoch_{epoch}.pt")
#         torch.save(pipe.unet.state_dict(), checkpoint_path)
#         print(f"✅ Saved checkpoint: {checkpoint_path}")

#         # Generate a sample output image for progress tracking
#         sample_prompt = "simple kanji style, test sample"
#         sample_image = pipe(sample_prompt).images[0]
#         sample_image.save(os.path.join(SAMPLES_DIR, f"sample_epoch_{epoch}.png"))
#         print(f"✅ Saved sample image for epoch {epoch}")

#     # Save Training Curve
#     with open(TRAINING_CURVE_FILE, "w", newline="") as csvfile:
#         fieldnames = ["epoch", "step", "loss"]
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         for log in training_logs:
#             writer.writerow(log)

#     print(f"✅ Training complete! Logs saved in '{TRAINING_CURVE_FILE}'.")

# except Exception as e:
#     print(f"❌ Training failed: {e}")


import os
import json
import csv
import torch
import lpips
from torch import nn
from torchvision import transforms
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from tqdm import tqdm
from PIL import Image

# ----- Configuration -----
MODEL_ID = "runwayml/stable-diffusion-v1-5"
DATASET_FILE = "kanji_mini_dataset.json"
CHECKPOINT_DIR = "checkpoints"
SAMPLES_DIR = "samples"
TRAINING_CURVE_FILE = "training_curve.csv"
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
MAX_TOKENS = 77  # CLIP token limit
IMAGE_SIZE = (256, 256)  # Ensure consistency
DEVICE = "cpu"  # Run on CPU

# Create directories if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

# ----- Load Mini Dataset -----
with open(DATASET_FILE, "r", encoding="utf-8") as f:
    kanji_data = json.load(f)

print(f"🚀 Loaded {len(kanji_data)} Kanji samples for training.")

# ----- Load Model & Scheduler -----
print("🚀 Loading model and scheduler...")
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, safety_checker=None)
pipe.to(DEVICE)
pipe.scheduler = EulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

# Set up optimizer and perceptual loss (LPIPS)
optimizer = torch.optim.Adam(pipe.unet.parameters(), lr=LEARNING_RATE)
loss_fn = lpips.LPIPS(net="alex").to(DEVICE)

# ----- Image Preprocessing -----
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

# ----- Training Loop -----
print("🚀 Starting mini training run...")
training_logs = []
global_step = 0

try:
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_losses = []
        
        for kanji, data in tqdm(kanji_data.items(), desc=f"Epoch {epoch}"):
            image_path = data["image"]  # Retrieve image path
            meanings = ", ".join(data["meanings"])  # Convert meanings into text

            if not os.path.exists(image_path):
                print(f"❌ Missing image for Kanji: {kanji} ({image_path})")
                continue

            # Load the actual Kanji image
            kanji_image = Image.open(image_path).convert("RGB")
            kanji_image = transform(kanji_image).unsqueeze(0).to(DEVICE)  # Convert to tensor

            # Generate text prompt based on meanings
            words = data["meanings"]
            prompt = " ".join(words)[:MAX_TOKENS]  # Clip if too long

            # Generate image using Stable Diffusion
            generated_image = pipe(prompt).images[0]
            generated_image = transform(generated_image).unsqueeze(0).to(DEVICE)

            # Compute perceptual loss (LPIPS)
            loss = loss_fn(generated_image, kanji_image)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Log loss
            loss_value = loss.item()
            epoch_losses.append(loss_value)
            training_logs.append({"epoch": epoch, "step": global_step, "loss": loss_value})
            global_step += 1

        # Save model checkpoint after each epoch
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"unet_epoch_{epoch}.pt")
        torch.save(pipe.unet.state_dict(), checkpoint_path)
        print(f"✅ Saved checkpoint: {checkpoint_path}")

        # Generate a sample output image for progress tracking
        sample_prompt = "A futuristic Kanji character"
        sample_image = pipe(sample_prompt).images[0]
        sample_image.save(os.path.join(SAMPLES_DIR, f"sample_epoch_{epoch}.png"))
        print(f"✅ Saved sample image for epoch {epoch}")

    # Save Training Curve
    with open(TRAINING_CURVE_FILE, "w", newline="") as csvfile:
        fieldnames = ["epoch", "step", "loss"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for log in training_logs:
            writer.writerow(log)

    print(f"✅ Training complete! Logs saved in '{TRAINING_CURVE_FILE}'.")

except Exception as e:
    print(f"❌ Training failed: {e}")
