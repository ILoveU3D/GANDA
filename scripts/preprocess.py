import os
import shutil

path = r"F:\Download\InstaGAN\czech\czech旧"
target = r"F:\Download\InstaGAN\czech\front"

for img in os.listdir(path):
    if "front" in img:
        shutil.copy(os.path.join(path,img), os.path.join(target, os.path.splitext(img)[0]))
        print(os.path.splitext(img)[0])
