import os
import torch
from PIL import Image
from torchvision.transforms import transforms
from options import checkpointPath, model, outputPath

net = torch.load(os.path.join(checkpointPath, model))
net.eval()
input = torch.randn([23*34*32*8]).cuda()
output = net.generate(input)
output = torch.squeeze(output, 0)
img = transforms.ToPILImage()(output)
path = os.path.join(outputPath, "test.jpg")
img.save(path)
os.rename(path, os.path.join(os.path.split(path)[0], os.path.splitext(os.path.basename(path))[0]+".svg"))