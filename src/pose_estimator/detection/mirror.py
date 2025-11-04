import os
from PIL import Image

# Input and output folders
image_dir = "/home/skoumal/dev/ObjectDetection/data/seibersdorf/yolo_format/rgb"
label_dir = "/home/skoumal/dev/ObjectDetection/data/seibersdorf/yolo_format/labels"
out_image_dir = "/home/skoumal/dev/ObjectDetection/data/seibersdorf/yolo_format/rgb_m"
out_label_dir = "/home/skoumal/dev/ObjectDetection/data/seibersdorf/yolo_format/labels_m"

os.makedirs(out_image_dir, exist_ok=True)
os.makedirs(out_label_dir, exist_ok=True)

def flip_coords(coords, flip_type):
    flipped = []
    for i, val in enumerate(coords):
        if i % 2 == 0:  # x-coordinate
            if flip_type in ["h", "hv"]:
                flipped.append(1 - val)
            else:
                flipped.append(val)
        else:  # y-coordinate
            if flip_type in ["v", "hv"]:
                flipped.append(1 - val)
            else:
                flipped.append(val)
    return flipped

# Process each image
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg"):
        # Paths
        img_path = os.path.join(image_dir, filename)
        label_name = filename.replace(".jpg", ".txt")
        label_path = os.path.join(label_dir, label_name)

        # Load image
        img = Image.open(img_path)

        # Load label
        if not os.path.exists(label_path):
            print(f"Warning: No label for {filename}")
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        # Parse labels
        objects = []
        for line in lines:
            parts = line.strip().split()
            cls = parts[0]
            coords = list(map(float, parts[1:]))
            objects.append((cls, coords))

        # Flip modes
        flips = {
            # "h": img.transpose(Image.FLIP_LEFT_RIGHT),
            # "v": img.transpose(Image.FLIP_TOP_BOTTOM),
            "hv": img.transpose(Image.ROTATE_180)
        }

        for key, flipped_img in flips.items():
            # Save flipped image
            out_img_name = f"{filename[:-4]}.jpg"
            flipped_img.save(os.path.join(out_image_dir, out_img_name))

            # Flip labels
            flipped_lines = []
            for cls, coords in objects:
                flipped_coords = flip_coords(coords, key)
                flipped_line = cls + " " + " ".join(f"{c:.6f}" for c in flipped_coords)
                flipped_lines.append(flipped_line)

            # Save flipped label
            out_label_name = f"{label_name[:-4]}.txt"
            with open(os.path.join(out_label_dir, out_label_name), "w") as f:
                f.write("\n".join(flipped_lines))