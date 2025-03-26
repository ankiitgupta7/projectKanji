import json

# Paths
FULL_DATASET_FILE = "kanji_dataset.json"
MINI_DATASET_FILE = "kanji_mini_dataset.json"

# Load full dataset
with open(FULL_DATASET_FILE, "r", encoding="utf-8") as f:
    kanji_data = json.load(f)

# Select exactly 10 Kanji samples
kanji_mini = dict(list(kanji_data.items())[:20])

# Save the reduced dataset
with open(MINI_DATASET_FILE, "w", encoding="utf-8") as f:
    json.dump(kanji_mini, f, ensure_ascii=False, indent=2)

print(f"✅ Mini dataset saved: {len(kanji_mini)} Kanji samples in '{MINI_DATASET_FILE}'.")
