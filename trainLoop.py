# +
from torch.utils import data
import torch.nn as nn
import torch.optim as optim
import torch

from tqdm import tqdm
# -

from utils import createIfNotExist, cityscapeColors, preview, saveModel

def train(model, dataset, device, modelName, epochs, batch_size=1, saveEachEpoch = False, previewShow = False):
    trainloader = data.DataLoader(
        dataset, 
        batch_size = batch_size, 
        num_workers = 10)

    colors = cityscapeColors()

    # loss function
    criterion = nn.MSELoss()
    
    # optimizer variable
    opt = optim.Adam(model.parameters())

    try:
        for epoch in tqdm(range(epochs)):
            if saveEachEpoch:
                torch.save(model, modelName)
            
            for i, (inputs, targets) in enumerate(trainloader):
                inputs = inputs.to(device)
                targets = targets.to(device)

                opt.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                opt.step()  

                if previewShow:
                    if i % 100 == 0: #each 100 iterations show current output
                        preview(inputs, outputs, targets, epoch)

    except KeyboardInterrupt:
        pass

    torch.save(model, modelName)
