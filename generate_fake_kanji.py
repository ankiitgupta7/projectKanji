import torch
import argparse
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import os
from tqdm import tqdm  # ✅ For progress tracking
import matplotlib.pyplot as plt

# 1️⃣ Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_model", type=str, default="CompVis/stable-diffusion-v1-4", help="Base Stable Diffusion model")
parser.add_argument("--lora_path", type=str, default="lora_kanji_unet", help="Path to fine-tuned LoRA model")
parser.add_argument("--output_dir", type=str, default="generated_kanji", help="Directory to save generated Kanji images")
parser.add_argument("--num_images", type=int, default=5, help="Number of test images to generate")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
args = parser.parse_args()

# 2️⃣ Define Test Prompts (Kanji Meanings)
test_prompts = ["Singularity", "Machine Learning", "Laptop", "Serendipity", "Taj Mahal"]  # ✅ Modify this to test different Kanji

# 3️⃣ Load Base Stable Diffusion Model
print("Loading base model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = StableDiffusionPipeline.from_pretrained(args.pretrained_model).to(device)

# 4️⃣ Load Fine-Tuned LoRA Model
print("Loading fine-tuned LoRA model...")
pipeline.unet = PeftModel.from_pretrained(pipeline.unet, args.lora_path).to(device)

# 5️⃣ Create Output Directory
os.makedirs(args.output_dir, exist_ok=True)

# 6️⃣ Generate Kanji Images
print(f"Generating {args.num_images} Kanji images...")
torch.manual_seed(args.seed)  # ✅ Set seed for reproducibility

progress_bar = tqdm(range(args.num_images), desc="Generating Images", leave=True)

for i in progress_bar:
    prompt = test_prompts[i % len(test_prompts)]  # Cycle through test prompts
    image = pipeline(prompt).images[0]  # Generate image

    # ✅ Save image
    output_path = os.path.join(args.output_dir, f"kanji_{prompt}.png")
    image.save(output_path)

    # ✅ Display image
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Kanji for '{prompt}'")
    plt.show()

print(f"✅ {args.num_images} Kanji images saved to '{args.output_dir}'")
