import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Path where the full pipeline was saved
model_path = "./kanji-model-finetuned"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the complete pipeline (ensure the folder includes model_index.json, etc.)
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to(device)

def generate_kanji(prompt, filename):
    # Generate an image using the pipeline
    image = pipe(prompt, guidance_scale=7.5).images[0]
    
    # Optionally post-process the image to enforce black-and-white output
    image = image.convert("L").point(lambda p: 0 if p < 128 else 255)
    image = image.resize((128, 128), Image.LANCZOS)
    
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
