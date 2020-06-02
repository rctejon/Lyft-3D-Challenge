import torch
from torch.nn import functional as F
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import TwoDtoThreeDDataset 
from scipy.spatial import ConvexHull
from tqdm import tqdm
from torch.autograd import Variable
from typing import List, Tuple


filejson = os.path.join("/media","user_home2","vision2020_01","Data","LYFT","data","dataset2Dto3D.json")
box3d_regressor_dataset = TwoDtoThreeDDataset.TwoDtoThreeDDataset(json_file=filejson)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

valset = Subset(box3d_regressor_dataset,range(659106,len(box3d_regressor_dataset)))

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
            x = F.relu(self.hiddenlinears(x))
        y_pred = self.linear2(x)
        return y_pred

N, D_in, H, D_out = 64, 14, 128, 10
C = 10
model = TwoLayerNet(D_in, H, C ,D_out)
model.cpu()
W = torch.load(os.path.join("/media","user_home2","vision2020_01","Data","LYFT","modelos","box_regressor_model.pth"),map_location=torch.device('cpu') )
model.load_state_dict(W)
model.eval()

def predecir(elem):
    
    # import pdb; pdb.set_trace()
    ypred = model(torch.from_numpy(elem).float())
    ypred = ypred.detach().numpy()

    return ypred

        