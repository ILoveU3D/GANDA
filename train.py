import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchsummary import summary
from visdom import Visdom
from dataloader.loader import CrezhFront
from model.VAE import VAE
from loss import KLDivLoss
from options import checkpointPath

trainSet = CrezhFront()
trainLoader = DataLoader(trainSet, batch_size=16, shuffle=True)
net = VAE(128).cuda()
net.initiate()
optimizer = torch.optim.Adam(net.parameters(), lr=10e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
viz = Visdom()
viz.line([[0,0]],[0],win="train",opts={"title":"Train Loss","legend":["trainLoss","validLoss"]})

epoch = 500

for i in range(epoch):
    net.train()
    trainLoss = []
    mseLoss = []
    klLoss = []
    with tqdm(trainLoader) as iterator:
        iterator.set_description("Epoch {}".format(i))
        for idx,data in enumerate(iterator):
            data = data.cuda()
            output, mean, var = net(data)
            optimizer.zero_grad()
            loss, mse, kld = KLDivLoss(output, data, mean, var)
            loss.backward()
            optimizer.step()
            trainLoss.append(loss.item())
            mseLoss.append(mse)
            klLoss.append(kld)
            iterator.set_postfix_str("loss:{:.2f}({:.2f},{:.2f})".format(np.mean(np.array(trainLoss)),np.mean(np.array(mseLoss)),np.mean(np.array(klLoss))))
    viz.line([[np.mean(np.array(trainLoss)),0]], [i], win="train", update="append")
    scheduler.step()
    if i % 10 == 0:
        torch.save(net, "{}/vae_{:.10f}.pth".format(checkpointPath, np.mean(np.array(trainLoss))))