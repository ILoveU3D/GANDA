import os
import torch
import numpy as np
import warnings
from torchvision.transforms import transforms
from PIL import Image
from collections import OrderedDict
from model.U2Net import U2NET
from options import outputPath, parseModelPath

warnings.filterwarnings("ignore", category=UserWarning)
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3),
])
net = U2NET(3,4).cuda()
stateDict = torch.load(parseModelPath)
stateDictOrdered = OrderedDict()
for k,v in stateDict.items():
    name = k[7:]
    stateDictOrdered[name] = v
net.load_state_dict(stateDictOrdered)
print("mission load complete")
net.eval()
img = Image.open(os.path.join(outputPath, "02")).convert("RGB")
# img = img.resize((int(img.size[0]/4),int(img.size[1]/4)), Image.ANTIALIAS)
# img.save(os.path.join(outputPath, "lx.png"))
img = trans(img).unsqueeze(0)
output = net(img.cuda())
output = torch.nn.functional.log_softmax(output[0], dim=1)
output = torch.max(output, dim=1, keepdim=True)[1]
output = output.squeeze(0).squeeze(0).cpu().numpy()
output = output * 10e4
outimg = Image.fromarray(output.astype("uint8"),mode="L")
outimg.save(os.path.join(outputPath, "out.jpg"))