import os
import random

IMG_EXT = (".jpg",".jpeg",".png",".bmp",".webp")

targets = {
    "train": 30000,
    "val": 5000,
    "test": 5000
}

root = "datasets/processed"

for split in targets:
    for label in ["real","fake"]:
        path = os.path.join(root, split, label)

        imgs = [os.path.join(path,f) for f in os.listdir(path) if f.lower().endswith(IMG_EXT)]
        random.shuffle(imgs)

        remove = imgs[targets[split]:]

        for img in remove:
            os.remove(img)

        print(f"{split} {label} → {targets[split]} kept")

print("Dataset reduction complete")