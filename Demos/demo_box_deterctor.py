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
import Demos.demo3D as demo3D
import matplotlib.pyplot as plt 
from pyquaternion import Quaternion
from typing import List, Tuple
from matplotlib.axes import Axes

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode
from lyft_dataset_sdk.utils.data_classes import Box
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer


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

def get_color(category_name: str) -> Tuple[int, int, int]:
    """Provides the default colors based on the category names.
    This method works for the general Lyft Dataset categories, as well as the Lyft Dataset detection categories.

    Args:
        category_name:

    Returns:

    """
    if "bicycle" in category_name or "motorcycle" in category_name:
        return 255, 61, 99  # Red
    elif "vehicle" in category_name or category_name in ["bus", "car", "construction_vehicle", "trailer", "truck"]:
        return 255, 158, 0  # Orange
    elif "pedestrian" in category_name:
        return 0, 0, 230  # Blue
    elif "cone" in category_name or "barrier" in category_name:
        return 0, 0, 0  # Black
    else:
        return 255, 0, 255  # Magenta

def render_sample_data(
        sample_data_token: str,
        with_anns: bool = True,
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        axes_limit: float = 40,
        ax: Axes = None,
        num_sweeps: int = 1,
        out_path: str = None,
        underlay_map: bool = False,
        detections: list = [],
        categories: list =[],
        valLyft:LyftDataset=None
    ):
    """Render sample data onto axis.

    Args:
        sample_data_token: Sample_data token.
        with_anns: Whether to draw annotations.
        box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        axes_limit: Axes limit for lidar and radar (measured in meters).
        ax: Axes onto which to render.
        num_sweeps: Number of sweeps for lidar and radar.
        out_path: Optional path to save the rendered figure to disk.
        underlay_map: When set to true, LIDAR data is plotted onto the map. This can be slow.

    """

    # Get sensor modality.
    sd_record = valLyft.get("sample_data", sample_data_token)
    sensor_modality = sd_record["sensor_modality"]

    
    if sensor_modality == "camera":
        # Load boxes and image.
        data_path, _, camera_intrinsic = valLyft.get_sample_data(
            sample_data_token, box_vis_level=box_vis_level
        )
        data = Image.open(data_path)

        # Init axes.
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(9, 16))

        # Show image.
        ax.imshow(data)
        #categories = ['car', 'pedestrian', 'truck', 'bicycle', 'bus', 'other_vehicle', 'motorcycle', 'emergency_vehicle', 'animal']
        # Show boxes.
        if with_anns:
            boxes =[]
            for c1, detection in enumerate(detections):
                #print(categories)
                cat = categories[c1]
                #print(cat)
                #import pdb; pdb.set_trace()
                box = Box(detection[:3],detection[3:6],Quaternion(np.array(detection[6:10])),name=cat)
                boxes.append(box)
            for box in boxes:
                c = np.array(get_color(box.name)) / 255.0
                box.render(ax, view=camera_intrinsic, normalize=True, colors=(c, c, c))

        # Limit visible range.
        ax.set_xlim(0, data.size[0])
        ax.set_ylim(data.size[1], 0)

    ax.axis("off")
    ax.set_title(sd_record["channel"])
    ax.set_aspect("equal")

    if out_path is not None:
        num = len([name for name in os.listdir(out_path)])
        out_path = out_path + str(num).zfill(5) + "_" + sample_data_token + ".png"
        plt.savefig(out_path)
        plt.close("all")
        return out_path


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

threshsums = 0
APss=0
image = 23456

def predict(numimage,valLyft,sample_data_val):
    for d in ["train", "val"]:
        MetadataCatalog.get("lyft_" + d).set(thing_classes=['car', 'pedestrian', 'truck', 'bicycle', 'bus', 'other_vehicle', 'motorcycle', 'emergency_vehicle', 'animal'])
    lyft_metadata = MetadataCatalog.get("lyft_train")
    num = 0
    for t in tqdm(sample_data_val[numimage:numimage+1]):
        dets = []
        im, boxes, labels, cam = getSampleData(t['token'],valLyft)
        im = np.array(Image.open(im))
        outputs = predictor(im)
        categories = ['car', 'pedestrian', 'truck', 'bicycle', 'bus', 'other_vehicle', 'motorcycle', 'emergency_vehicle', 'animal']
        cats=[]
        outputs = predictor(im)
        v = Visualizer(im[:, :, :],
                    metadata=lyft_metadata, 
                    scale=0.8, 
                    instance_mode=ColorMode.IMAGE   # remove the colors of unsegmented pixels
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.imshow(v.get_image())
        plt.savefig(f"./results/{num}.png")
        num+=1
        plt.close("all")
        for i, det in enumerate(outputs["instances"].pred_boxes):
            
            box = det.cpu().numpy()
            box.reshape(-1)
            cam.reshape(-1)
            cat = outputs["instances"].pred_classes[i].cpu().numpy()
            #print(cat)
            detec = np.concatenate((box,cam,np.array([cat])),axis=None)
            det = demo3D.predecir(detec)
            cats.append(categories[cat])
            dets.append(det)
        render_sample_data(t['token'],out_path="./results/",detections=dets, categories = cats,valLyft=valLyft)
