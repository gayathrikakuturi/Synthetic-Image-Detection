import os
import shutil
import random

IMG_EXT = (".jpg",".jpeg",".png",".bmp",".webp")

REAL = [
    "datasets/raw/cifar10",
    "datasets/raw/stylegan_real",
    "datasets/raw/SuSy/train/coco",
    "datasets/raw/SuSy/val/coco",
    "datasets/raw/SuSy/test/coco"
]

FAKE = [
    "datasets/raw/sfhq",
    "datasets/raw/synthetic_objects",
    "datasets/raw/SuSy/train/dalle-3-images",
    "datasets/raw/SuSy/train/diffusiondb",
    "datasets/raw/SuSy/train/midjourney_tti",
    "datasets/raw/SuSy/train/midjourney-images",
    "datasets/raw/SuSy/train/realisticSDXL"
]

def collect(paths):
    imgs=[]
    for p in paths:
        for r,d,f in os.walk(p):
            for file in f:
                if file.lower().endswith(IMG_EXT):
                    imgs.append(os.path.join(r,file))
    return imgs

real_imgs = collect(REAL)
fake_imgs = collect(FAKE)

# balance
random.shuffle(real_imgs)
real_imgs = real_imgs[:len(fake_imgs)]

# merge balanced
all_data = {
    "real": real_imgs,
    "fake": fake_imgs
}

train_ratio=0.7
val_ratio=0.15

for label, imgs in all_data.items():
    random.shuffle(imgs)

    train_end=int(len(imgs)*train_ratio)
    val_end=int(len(imgs)*(train_ratio+val_ratio))

    splits={
        "train":imgs[:train_end],
        "val":imgs[train_end:val_end],
        "test":imgs[val_end:]
    }

    for split in splits:
        dest=f"datasets/processed/{split}/{label}"
        os.makedirs(dest,exist_ok=True)

        for img in splits[split]:
            shutil.copy(img,dest)

print("Balanced split complete")