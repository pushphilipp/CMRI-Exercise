import json
import os
import base64

notebook = "lab10_exercise.ipynb"  # <-- change this
out_dir = "extracted_plots_solution"
os.makedirs(out_dir, exist_ok=True)

with open(notebook, "r", encoding="utf-8") as f:
    data = json.load(f)

img_count = 1

for cell in data.get("cells", []):
    if "outputs" not in cell:
        continue
    for output in cell["outputs"]:
        if "data" not in output:
            continue

        # PNG images
        if "image/png" in output["data"]:
            img_data = base64.b64decode(output["data"]["image/png"])
            filename = os.path.join(out_dir, f"plot_{img_count}.png")
            with open(filename, "wb") as img_file:
                img_file.write(img_data)
            img_count += 1

        # SVG images (optional)
        if "image/svg+xml" in output["data"]:
            svg_data = "".join(output["data"]["image/svg+xml"])
            filename = os.path.join(out_dir, f"plot_{img_count}.svg")
            with open(filename, "w", encoding="utf-8") as svg_file:
                svg_file.write(svg_data)
            img_count += 1

print(f"Extracted {img_count - 1} images into '{out_dir}'")

