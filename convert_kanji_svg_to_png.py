import os
import cairosvg
import xml.etree.ElementTree as ET
from tqdm import tqdm

# Paths
KANJIVG_PATH = "kanjivg.xml"  # Path to the KanjiVG XML dataset
OUTPUT_FOLDER = "kanji_png_128"  # Output directory for PNG files (128x128)

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Parse XML
tree = ET.parse(KANJIVG_PATH)
root = tree.getroot()

# Extract and convert each Kanji
for kanji in tqdm(root.findall("kanji"), desc="Converting SVGs"):
    kanji_id = kanji.attrib["id"].replace("kvg:kanji_", "")  # Extract Kanji character
    svg_output = os.path.join(OUTPUT_FOLDER, f"{kanji_id}.svg")
    png_output = os.path.join(OUTPUT_FOLDER, f"{kanji_id}.png")

    try:
        # Extract SVG content (Kanji strokes only)
        svg_content = ET.tostring(kanji, encoding="utf-8").decode()

        # Wrap the original strokes inside an <svg> container
        svg_content = f'''
        <svg xmlns="http://www.w3.org/2000/svg" width="128" height="128" viewBox="0 0 109 109">
            {svg_content}  <!-- Keep strokes unchanged -->
        </svg>
        '''

        # Save as SVG (preserving original strokes)
        with open(svg_output, "w", encoding="utf-8") as f:
            f.write(svg_content)

        # Convert to PNG (extract strokes only, no fill) at 128x128 resolution
        cairosvg.svg2png(url=svg_output, write_to=png_output, output_width=128, output_height=128)

        # Remove SVG file after conversion (optional)
        os.remove(svg_output)

    except Exception as e:
        print(f"❌ Error processing {kanji_id}: {e}")

print(f"✅ Conversion complete! 128x128 PNG files saved in '{OUTPUT_FOLDER}'.")
