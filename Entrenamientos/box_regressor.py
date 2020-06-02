import os
import torch
from torch.nn import functional as F
import numpy as np
import TwoDtoThreeDDataset
from torch.utils.data import Dataset, DataLoader, Subset
from torch.autograd import Variable
import time

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 14, 128, 10
C = 10
EPOCHS = range(40)
gpuID = 2

box3d_regressor_dataset = TwoDtoThreeDDataset.TwoDtoThreeDDataset(json_file='data/dataset2Dto3D.json')

trainset = Subset(box3d_regressor_dataset,range(0,659106))
valset = Subset(box3d_regressor_dataset,range(659106,len(box3d_regressor_dataset)))

trainloader = DataLoader(trainset, batch_size=N,
                        shuffle=False, num_workers=1)
valloader = DataLoader(valset, batch_size=N,
                        shuffle=False, num_workers=1)



class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, C, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.C = C
        self.linear1 = torch.nn.Linear(D_in, H)
        self.hiddenlinears = torch.nn.Linear(H, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = F.relu(self.linear1(x))
        for _ in  range(self.C):
            print('holi')
            x = F.relu(self.hiddenlinears(x))
        y_pred = self.linear2(x)
        return y_pred


# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, C ,D_out)
print(model)
model.cuda(gpuID)
# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

# Create random Tensors to hold inputs and outputs
for epoch in EPOCHS:
    actual_loss=0.0
    for batch_idx, data in enumerate(trainloader, 0):
        x, y = data
        x, y = x.cuda(gpuID), y.cuda(gpuID)
        x, y = Variable(x), Variable(y)

        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        actual_loss += loss.item()

        if batch_idx % 1000 == 999:
            timestr = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            print("%s epoch: %d iter:%d loss:%.6f"%(
                        timestr, epoch+1, batch_idx+1, actual_loss/1000))
            actual_loss=0.0

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "./output/box_regressor_model.pth")