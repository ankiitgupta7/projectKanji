import os
import json
from tqdm import tqdm

# Paths
KANJI_IMAGE_FOLDER = "kanji_png_128"  # Folder containing Kanji images
KANJI_MEANINGS_FILE = "kanji_meanings.json"
OUTPUT_JSON = "kanji_dataset.json"

# Load extracted Kanji meanings
with open(KANJI_MEANINGS_FILE, "r", encoding="utf-8") as f:
    kanji_meanings = json.load(f)

# Convert filenames from Unicode hex (e.g., 05ed0.png) to Kanji
image_filenames = {
    chr(int(filename.replace(".png", ""), 16)): filename
    for filename in os.listdir(KANJI_IMAGE_FOLDER) if filename.endswith(".png")
}

# Create dataset mapping
kanji_dataset = {}

for kanji, meanings in tqdm(kanji_meanings.items(), desc="Mapping images to meanings"):
    if kanji in image_filenames:
        kanji_dataset[kanji] = {
            "image": os.path.join(KANJI_IMAGE_FOLDER, image_filenames[kanji]),
            "meanings": meanings
        }

# Save dataset
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(kanji_dataset, f, ensure_ascii=False, indent=2)

print(f"✅ Mapping complete! Dataset saved as '{OUTPUT_JSON}' with {len(kanji_dataset)} entries.")
