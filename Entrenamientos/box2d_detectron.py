import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import os
import numpy as np
import json
from detectron2.structures import BoxMode
from tqdm import tqdm

categories=[]
 
# CUDA_VISIBLE_DEVICES=1 ./cuda_executable
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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
print(categories)
get_lyft_dicts("val")
#METADATA
from detectron2.data import DatasetCatalog, MetadataCatalog
for d in ["train", "val"]:
    DatasetCatalog.register("lyft_" + d, lambda d=d: get_lyft_dicts(d))
    MetadataCatalog.get("lyft_" + d).set(thing_classes=categories)
lyft_metadata = MetadataCatalog.get("lyft_train")


# dataset_dicts = get_lyft_dicts("train")
# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=lyft_metadata, scale=0.5)
#     vis = visualizer.draw_dataset_dict(d)

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

print("inicio train")
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
cfg.DATASETS.TRAIN = ("lyft_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.000025  # pick a good LR
cfg.SOLVER.MAX_ITER = 500    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9  # only has one class (ballon)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
