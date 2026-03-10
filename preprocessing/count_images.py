import os

# image extensions
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

# root dataset folder
root = "datasets/raw"

total = 0

for dataset in os.listdir(root):
    dataset_path = os.path.join(root, dataset)

    if os.path.isdir(dataset_path):
        count = 0
        for r, d, f in os.walk(dataset_path):
            for file in f:
                if file.lower().endswith(IMG_EXT):
                    count += 1

        print(f"{dataset} → {count} images")
        total += count

print("\nTOTAL images:", total)