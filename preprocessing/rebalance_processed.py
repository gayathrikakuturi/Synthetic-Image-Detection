import os
import random

IMG_EXT = (".jpg",".jpeg",".png",".bmp",".webp")

root = "datasets/processed"

for split in os.listdir(root):
    split_path = os.path.join(root, split)

    real_path = os.path.join(split_path,"real")
    fake_path = os.path.join(split_path,"fake")

    real_imgs = [os.path.join(real_path,f) for f in os.listdir(real_path) if f.lower().endswith(IMG_EXT)]
    fake_imgs = [os.path.join(fake_path,f) for f in os.listdir(fake_path) if f.lower().endswith(IMG_EXT)]

    # target = min(real,fake)
    target = min(len(real_imgs),len(fake_imgs))

    # remove extra fake
    random.shuffle(fake_imgs)
    remove = fake_imgs[target:]

    for img in remove:
        os.remove(img)

    print(f"{split} balanced → {target} each")