import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Directory where the full fine-tuned pipeline is saved.
model_path = "./kanji-model-finetuned"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the complete pipeline.
# Make sure that your fine-tuning script saved the complete pipeline (including model_index.json) to model_path.
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe = pipe.to(device)

def generate_kanji(prompt, filename):
    # Generate an image from the given prompt.
    image = pipe(prompt, guidance_scale=7.5).images[0]
    
    # Post-process: convert to grayscale and threshold to pure black-and-white.
    # image = image.convert("L").point(lambda p: 0 if p < 128 else 255)
    # Resize the image to 128x128 (if not already)
    image = image.resize((128, 128), Image.LANCZOS)
    
    # Save the image.
    image.save(filename)
    print(f"Image saved as {filename}")

if __name__ == "__main__":
    prompts = [
        "Elon Musk",
        "YouTube",
        "Skyscraper",
        "Neural Samurai"
    ]
    
    for prompt in prompts:
        out_file = prompt.replace(" ", "_") + ".png"
        generate_kanji(prompt, out_file)
