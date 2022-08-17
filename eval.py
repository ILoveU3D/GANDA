import os
import torch
from PIL import Image
from torchvision.transforms import transforms
from options import checkpointPath, model, outputPath

net = torch.load(os.path.join(checkpointPath, model))
net.eval()
input = Image.open(os.path.join(outputPath, "100_10_front"))
input = transforms.ToTensor()(input.resize((350,525))).cuda()
input = torch.unsqueeze(input, 0)
output, _, _ = net(input)
output = torch.squeeze(output, 0)
img = transforms.ToPILImage()(output)
path = os.path.join(outputPath, "test.jpg")
img.save(path)
os.rename(path, os.path.join(os.path.split(path)[0], os.path.splitext(os.path.basename(path))[0]+".svg"))