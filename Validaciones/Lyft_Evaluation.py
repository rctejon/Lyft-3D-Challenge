import torch
from torch.nn import functional as F
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import TwoDtoThreeDDataset 
from scipy.spatial import ConvexHull
from tqdm import tqdm
from torch.autograd import Variable


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

def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0  
def polygon_clip(subjectPolygon, clipPolygon):
   """ Clip a polygon with another polygon.
   Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python
   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**
   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)

def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def get_3d_box(box_size, heading_angle, center):
    ''' Calculate 3D bounding box corners from its parameterization.
    Input:
        box_size: tuple of (length,wide,height)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''
    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])

    R = roty(heading_angle)
    l,w,h = box_size
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + center[0]
    corners_3d[1,:] = corners_3d[1,:] + center[1]
    corners_3d[2,:] = corners_3d[2,:] + center[2]
    corners_3d = np.transpose(corners_3d)
    return corners_3d

def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.
    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU
    todo (kent): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
    
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
   
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])

    inter_vol = inter_area * max(0.0, ymax-ymin)
    
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d


def IoU(det, gt):
    ''' Calculate 3D bounding box corners from its parameterization.
    Input:
        box_size: tuple of (length,wide,height)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''
    corners_det = get_3d_box((det[4],det[3],det[5]),0,(det[0],det[1],det[2]))
    corners_gt = get_3d_box((gt[4],gt[3],gt[5]),0,(gt[0],gt[1],gt[2]))
    return box3d_iou(corners_det,corners_gt)[0]


def validar(dets, gts):
    categories = ['car', 'pedestrian', 'truck', 'bicycle', 'bus', 'other_vehicle', 'motorcycle', 'emergency_vehicle', 'animal']
    

    thresholds =  [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    threshsum = 0
    APs =[]
    APcats = [0]*len(categories)
    
    
    for thresh in thresholds:
        detections = np.ones((len(dets)))
        grounds= np.ones((len(gts)))
        TP = 0
        FP = 0
        FN = 0
        TPcats = [0]*len(categories)
        FPcats = [np.ones((len(dets)))]*len(categories)
        FNcats = [np.ones((len(gts)))]*len(categories)
        for i, elem in enumerate(dets):
            # import pdb; pdb.set_trace()
            ypred = model(torch.from_numpy(elem).float())
            ypred = ypred.detach().numpy()
            cat = categories[int(elem[-1])]
            for j, g in enumerate(gts):
                if g["cat"] == cat:
                    iou = IoU(ypred,g["value"])
                    if iou >= thresh:
                        TP += 1
                        TPcats[int(elem[-1])]+=1
                        FPcats[int(elem[-1])][i]=0
                        FNcats[int(elem[-1])][j]=0
                        detections[i]=0
                        grounds[j]=0
                else:
                    FPcats[categories.index(g["cat"])][i]=0
                    FNcats[int(elem[-1])][j]=0
        FPcats = list(map(lambda x: np.sum(x),FPcats))
        FNcats = list(map(lambda x: np.sum(x),FNcats))
        APcatss = list(map(lambda x:x[0]/(x[0]+x[1]+x[2]) if (x[0]+x[1]+x[2]) >0 else -1,list(zip(TPcats,FPcats,FNcats))))
        FP += np.sum(detections)
        FN += np.sum(grounds)
        APs.append(TP/(TP + FP + FN))
        for aa, APcat in enumerate(APcatss):
            if APcat>0:
                APcats[aa]+=APcat/len(thresholds)
        
        threshsum += (APs[-1])/len(thresholds)

    return threshsum, APs, thresholds, APcats


        