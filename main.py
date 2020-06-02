# -*- coding: utf-8 -*-
import os
import time
import argparse
import numpy as np
import json
from tqdm import tqdm

from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer
from lyft_dataset_sdk.utils.geometry_utils import BoxVisibility, box_in_image, view_points  # NOQA
import Validaciones.Lyft_Evaluation as Lyft_Evaluation
import Demos.demo_box_deterctor as box_detector

import Validaciones.validacion_general as validacion_general


def main():
    # SET THE PARAMETERS
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="test",
                        help='Mode of the code (default: test)')
    parser.add_argument('--img', type=int, default=0,
                        help='Number of the validation image from 0 to 47627 (default: 0)')
    args = parser.parse_args()

    mode = args.mode
    image = args.img
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    filedata = os.path.join("/media","user_home2","vision2020_01","Data","LYFT","data","val")
    filejson = os.path.join("/media","user_home2","vision2020_01","Data","LYFT","data","val","json")
    valLyft = LyftDataset(data_path=filedata, json_path=filejson, verbose=True)
    sample_dataVal = list(filter(lambda x:x['sensor_modality']=='camera',valLyft.sample_data))

    filedata = os.path.join("/media","user_home2","vision2020_01","Data","LYFT","data","samples_split.json")
    data = json.load(open(filedata))

    val_indices = data['val']
    sample_data_val = []

    for i in tqdm(val_indices):
        sample_data_val.append(sample_dataVal[i])

    if mode == "test":
        validacion_general.validar(sample_data_val,valLyft)
    elif mode=="demo":
        box_detector.predict(image,valLyft,sample_data_val)

    print(mode, image)

if __name__ == '__main__':
    main()
