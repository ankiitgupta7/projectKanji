import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# ---- SETTINGS ----
output_dir = "kanji-model-finetuned131"  # your main output dir with checkpoints
checkpoint_prefix = "checkpoint_epoch_"   # used when naming subfolders
save_root = "generatedKanji131"                  # base folder for generated images
prompts = [
    "A Single Kanji Character in Black for Narendra Modi in white background",
    "A Single Kanji Character in Black for Taj Mahal in white background",
    "A Single Kanji Character in Black for Tea in white background",
    "A Single Kanji Character in Black for Open-endedness in white background",
    "A Single Kanji Character in Black for Mathematics in white background",
    "A Single Kanji Character in Black for Machine Learning in white background",
    "A Single Kanji Character in Black for Artificial Intelligence in white background"

]
guidance_scale = 7.5
num_inference_steps = 50

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # Find all checkpoint dirs named "checkpoint_epoch_{X}"
    checkpoint_folders = []
    for folder in os.listdir(output_dir):
        path = os.path.join(output_dir, folder)
        if os.path.isdir(path) and folder.startswith(checkpoint_prefix):
            checkpoint_folders.append(folder)

    # Sort them by epoch number
    # e.g. "checkpoint_epoch_10" -> 10
    def extract_epoch_num(folder_name):
        # folder_name: "checkpoint_epoch_10" -> 10 as int
        return int(folder_name.split("_")[-1])

    checkpoint_folders.sort(key=extract_epoch_num)

    if not checkpoint_folders:
        print(f"No folders found with prefix '{checkpoint_prefix}'. Exiting.")
        return

    print("Generating images for these checkpoints:")
    for cf in checkpoint_folders:
        print("  ", cf)

    for folder in checkpoint_folders:
        ckpt_path = os.path.join(output_dir, folder)
        epoch_num = extract_epoch_num(folder)

        # Load the pipeline from this checkpoint
        print(f"\nLoading checkpoint: {ckpt_path}")
        pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
        pipe = pipe.to(device)

        # Subdir for saving
        save_subdir = os.path.join(save_root, folder)
        os.makedirs(save_subdir, exist_ok=True)

        print(f"Generating images for epoch {epoch_num}, saving to {save_subdir}")
        for prompt in prompts:
            with torch.autocast(device) if device=="cuda" else nullcontext():
                image = pipe(prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]

            # Optional post-process to black & white threshold
            # image = image.convert("L").point(lambda p: 0 if p < 128 else 255)

            # Save
            filename = f"{prompt.replace(' ', '_')}.png"
            out_path = os.path.join(save_subdir, filename)
            image.save(out_path)
            print(f"  Saved: {out_path}")

    print("\nAll checkpoints processed. Done!")

# For the CPU autocast fallback, define a no-op context manager
import contextlib
@contextlib.contextmanager
def nullcontext(enter_result=None):
    yield enter_result

if __name__ == "__main__":
    main()
