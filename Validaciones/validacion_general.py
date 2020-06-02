import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import json
import os
from tqdm import tqdm
from PIL import Image

from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer
from lyft_dataset_sdk.utils.geometry_utils import BoxVisibility, box_in_image, view_points  # NOQA
import Validaciones.Lyft_Evaluation as Lyft_Evaluation

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode

categories = []

def box3Dto2D(corners):
    return np.array([[np.min(corners[:][0]) , np.max(corners[:][0])],
                    [ np.min(corners[:][1]) , np.max(corners[:][1])]])

def clipDetections(corners, size):
    corners[:][0] = np.clip(corners[:][0],0,size[0] )
    corners[:][1] = np.clip(corners[:][1],0,size[1] )
    return corners

def getSampleData(sample_token,dataLyft3D):
    data_path, boxes, camera_intrinsic = dataLyft3D.get_sample_data(sample_token, selected_anntokens=None)
    newBoxes = []
    im = Image.open(data_path)
    labels = list(map(lambda x : x.name,boxes))
    for box in boxes:
        # import pdb; pdb.set_trace()
        corners = view_points(box.corners(), camera_intrinsic, normalize=True)[:2, :]
        corners = clipDetections(corners,im.size)
        newBoxes.append(box3Dto2D(corners))
    return data_path , newBoxes, boxes, camera_intrinsic


os.environ['CUDA_VISIBLE_DEVICES'] = '2'
cfg = get_cfg()

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
cfg.DATASETS.TRAIN = ("lyft_train",)
cfg.DATASETS.TEST = ("lyft_val", )
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = os.path.join("/media","user_home2","vision2020_01","Data","LYFT","modelos","model_final.pth")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 3000    
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9  

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)



def validar(sample_data_val,valLyft):
    threshsums = 0
    APss = 0
    APcatss = 0
    for t in tqdm(sample_data_val):
        dets = []
        gts = []
        im, boxes, labels, cam = getSampleData(t['token'],valLyft)
        im = np.array(Image.open(im))
        outputs = predictor(im)
        categories = ['car', 'pedestrian', 'truck', 'bicycle', 'bus', 'other_vehicle', 'motorcycle', 'emergency_vehicle', 'animal']
        for i, det in enumerate(outputs["instances"].pred_boxes):
            
            box = det.cpu().numpy()
            box.reshape(-1)
            cam.reshape(-1)
            cat = outputs["instances"].pred_classes[i].cpu().numpy()
            detec = np.concatenate((box,cam,np.array([cat])),axis=None)
            dets.append(detec)
        for i, lab in enumerate(labels):
            label = np.concatenate((np.array(lab.center),np.array(lab.wlh),lab.orientation.q),axis=None)
            obj={
                "cat": lab.name,
                "value": label
            }
            gts.append(obj)
        if len(dets) + len(gts)>0:
            threshsum, APs, thresholds, APcats = Lyft_Evaluation.validar(dets,gts)
            threshsums += threshsum
            if APss ==0:
                APss = APs
            else:
                for a, AP in enumerate(APs):
                    APss[a] += AP
            if APcatss ==0:
                APcatss = APcats
            else:
                for a, AP in enumerate(APcats):
                    APcatss[a] += AP
    print(f"AP: {threshsums}")

    for i, thresh in enumerate(thresholds):
        print(f"AP {thresh}: {APss[i]}")
    for i, cat in enumerate(categories):
        print(f"AP {cat}: {APcatss[i]}")