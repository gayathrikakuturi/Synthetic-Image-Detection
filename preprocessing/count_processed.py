import os

IMG_EXT = (".jpg",".jpeg",".png",".bmp",".webp")

root = "datasets/processed"

total = 0

for split in os.listdir(root):
    split_path = os.path.join(root, split)

    if os.path.isdir(split_path):
        print(f"\n{split.upper()}")

        for label in os.listdir(split_path):
            label_path = os.path.join(split_path, label)

            count = 0
            for r,d,f in os.walk(label_path):
                for file in f:
                    if file.lower().endswith(IMG_EXT):
                        count += 1

            print(f"{label} → {count}")
            total += count

print("\nTOTAL images:", total)