# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

import os
import numpy as np
import json
from detectron2.structures import BoxMode
from tqdm import tqdm
categories = []

 
# CUDA_VISIBLE_DEVICES=1 ./cuda_executable
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


MetadataCatalog

def get_lyft_dicts(dataset): #dir: mini_data, set: train o val 
    dir="data"
    print(dataset)
    json_file = os.path.join("..",dir,"detections.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)[dataset]
    dataset_dicts = []
    for idx, v in tqdm(enumerate(imgs_anns)):
        record = {}
        
        filename = v[list(v.keys())[0]]['path']
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v[list(v.keys())[0]]['detections']
        objs = []
        for anno in annos:
            px = anno["box"][0]
            py = anno["box"][1]
            if anno["name"] not in categories:
                categories.append(anno["name"])
            cat = categories.index(anno["name"])
            obj = {
                "bbox": [px[0], py[0], px[1], py[1]],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": cat,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts
    


from detectron2.data import DatasetCatalog, MetadataCatalog
#for d in ["train", "val"]:
d = "val"
DatasetCatalog.register("lyft_" + d, lambda d=d: get_lyft_dicts(d))
MetadataCatalog.get("lyft_" + d).set(thing_classes=categories)
lyft_metadata = MetadataCatalog.get("lyft_val")

print(categories)

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg


cfg = get_cfg()

cfg.MODEL.WEIGHTS = os.path.join("..","output","model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
cfg.DATASETS.TEST = ("lyft_val", )
predictor = DefaultPredictor(cfg)


from detectron2.utils.visualizer import ColorMode
dataset_dicts = get_lyft_dicts("val")
i= 0
for d in random.sample(dataset_dicts, 10):    
    im = cv2.imread(d["file_name"])
    #cv2.imshow(im)
    cv2.imwrite("/imagen"+str(i)+".png", im) 
    print(d["file_name"])
    outputs = predictor(im)
    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)
    i=i+1