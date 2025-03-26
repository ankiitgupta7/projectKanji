import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from PIL import Image
import os

# 1. Choose GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Load the base Stable Diffusion pipeline in half precision
pipeline = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16
)
pipeline.to(device)

# 3. Load your epoch 30 LoRA checkpoint into the UNet
lora_checkpoint_path = "lora_kanji_unet/checkpoint_epoch_30"  # Adjust path as needed
pipeline.unet = PeftModel.from_pretrained(
    pipeline.unet,
    lora_checkpoint_path
)
pipeline.unet.to(device)

# 4. Optionally disable the safety checker if you want raw outputs
# pipeline.safety_checker = None

# 5. Define a list of interesting prompts
prompts = [
  "Kanji for Elon Musk",
  "Kanji for mountain",
  "Kanji for Taj Mahal",
  "Kanji for Machine Learning",
  "Kanji for Watermelon",
  "Kanji for Kolkata",
  "A black ink Japanese kanji character for Kolkata on a white background",
  "A black ink Japanese kanji character for mountain on a white background",
  "A black ink Japanese kanji character for Watermelon on a white background",
  "A black ink Japanese kanji character for Elon Musk on a white background",
  "A black ink Japanese kanji character for Modi on a white background",
  "A black ink Japanese kanji character for Trump on a white background",
  "A black ink Japanese kanji character for Obama on a white background",
]

# 6. Set generation parameters
num_inference_steps = 50
guidance_scale = 6

# 7. (Optional) Fix a random seed for reproducibility
generator = torch.Generator(device=device).manual_seed(42)

# 8. Create an output folder for generated images
output_dir = "kanji_images_epoch30"
os.makedirs(output_dir, exist_ok=True)

# 9. Generate an image for each prompt
for i, prompt in enumerate(prompts, start=1):
    print(f"Generating image for prompt: {prompt}")
    result = pipeline(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    )
    generated_image = result.images[0]

    # Save the resulting image
    out_path = os.path.join(output_dir, f"kanji_{i}_{prompt.replace(' ', '_')}.png")
    generated_image.save(out_path)
    print(f"Saved to: {out_path}")
