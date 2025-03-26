import xml.etree.ElementTree as ET
import json

# Load the XML file
KANJIDIC2_PATH = "kanjidic2.xml"

def extract_kanji_meanings(xml_path):
    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    kanji_dict = {}

    # Iterate over each <character> entry
    for char in root.findall("character"):
        literal = char.find("literal").text  # Kanji character
        meanings = [m.text for m in char.findall("reading_meaning/rmgroup/meaning")]

        # Store in dictionary
        kanji_dict[literal] = meanings

    return kanji_dict

# Extract meanings
kanji_meanings = extract_kanji_meanings(KANJIDIC2_PATH)

# Save as JSON
with open("kanji_meanings.json", "w", encoding="utf-8") as f:
    json.dump(kanji_meanings, f, ensure_ascii=False, indent=2)

print(f"✅ Extracted {len(kanji_meanings)} Kanji entries and saved to 'kanji_meanings.json'.")
