import os
import json
import shutil

# Set paths
json_path = "kanji_dataset.json"  # Update this if needed
image_base_path = "kanji_png_128"  # Folder containing the original images
output_folder = "kanji_6413"     # Folder where we will copy the images

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load JSON data
with open(json_path, "r", encoding="utf-8") as f:
    kanji_data = json.load(f)

# Extract and copy images
for kanji, data in kanji_data.items():
    image_rel_path = data.get("image")
    if image_rel_path:
        src_path = os.path.join(image_rel_path)
        filename = os.path.basename(image_rel_path)
        dest_path = os.path.join(output_folder, filename)
        
        try:
            shutil.copy(src_path, dest_path)
            print(f"Copied: {src_path} → {dest_path}")
        except FileNotFoundError:
            print(f"Missing: {src_path}")

print("Done copying images.")
