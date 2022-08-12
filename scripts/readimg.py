import os
from PIL import Image
from options import root

for img in os.listdir(root):
    print(Image.open(os.path.join(root, img)).size)