from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer
from lyft_dataset_sdk.utils.geometry_utils import BoxVisibility, box_in_image, view_points  # NOQA
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm

import json
import random

trainLyft = LyftDataset(data_path='../data/train/', json_path='../data/train/json', verbose=True)
sample_dataTrain = list(filter(lambda x:x['sensor_modality']=='camera',trainLyft.sample_data))

valLyft = LyftDataset(data_path='../data/val/', json_path='../data/val/json', verbose=True)
sample_dataVal = list(filter(lambda x:x['sensor_modality']=='camera',valLyft.sample_data))

def render_ann(ann_token,dataLyft3D):
    ann_record = dataLyft3D.get("sample_annotation", ann_token)
    sample_record = dataLyft3D.get("sample", ann_record["sample_token"])

    fig, axes = plt.subplots(1, 1, figsize=(12, 6))

    # Figure out which camera the object is fully visible in (this may return nothing)
    boxes, cam = [], []
    cams = [key for key in sample_record["data"].keys() if "CAM" in key]
    for cam in cams:
        _, boxes, _ = dataLyft3D.get_sample_data(
            sample_record["data"][cam], box_vis_level=BoxVisibility.ANY, selected_anntokens=[ann_token]
        )
        if len(boxes) > 0:
            break  # We found an image that matches. Let's abort.
    cam = sample_record["data"][cam]
    data_path, boxes, camera_intrinsic = dataLyft3D.get_sample_data(cam, selected_anntokens=[ann_token])
    im = Image.open(data_path)
    axes.imshow(im)
    axes.set_title(dataLyft3D.get("sample_data", cam)["channel"])
    axes.axis("off")
    axes.set_aspect("equal")
    for box in boxes:
            c = np.array(LyftDatasetExplorer.get_color(box.name)) / 255.0
            render_box(box,axes, view=camera_intrinsic, normalize=True, colors=(c, c, c),im=im)

def render_sample(sample_token,dataLyft3D):

    fig, axes = plt.subplots(1, 1, figsize=(12, 6))

    cam = dataLyft3D.get("sample_data", sample_token)
    data_path, boxes, camera_intrinsic = dataLyft3D.get_sample_data(sample_token, selected_anntokens=None)
    im = Image.open(data_path)
    axes.imshow(im)
    axes.set_title(cam["channel"])
    axes.axis("off")
    axes.set_aspect("equal")
    for box in boxes:
            c = np.array(LyftDatasetExplorer.get_color(box.name)) / 255.0
            render_box(box,axes, view=camera_intrinsic, normalize=True, colors=(c, c, c),im=im)

def render_box(box,axis,view,normalize,colors,im,linewidth=2):
    corners = view_points(box.corners(), view, normalize=normalize)[:2, :]
    corners = clipDetections(corners,im.size)
    # print([np.min(corners[:][0]), np.max(corners[:][0])],
    #       [np.min(corners[:][1]), np.max(corners[:][1])])
    render_corners(axis,corners,linewidth)

def render_corners(axis,corners,linewidth):
    axis.plot(
        [np.min(corners[:][0]), np.min(corners[:][0])],
        [np.min(corners[:][1]), np.max(corners[:][1])],
        color=(0,0.4,0.8,1),
        linewidth=linewidth,
    )
    axis.plot(
        [np.min(corners[:][0]), np.max(corners[:][0])],
        [np.max(corners[:][1]), np.max(corners[:][1])],
        color=(0,0.4,0.8,1),
        linewidth=linewidth,
    )
    axis.plot(
        [np.max(corners[:][0]), np.max(corners[:][0])],
        [np.max(corners[:][1]), np.min(corners[:][1])],
        color=(0,0.4,0.8,1),
        linewidth=linewidth,
    )
    axis.plot(
        [np.max(corners[:][0]), np.min(corners[:][0])],
        [np.min(corners[:][1]), np.min(corners[:][1])],
        color=(0,0.4,0.8,1),
        linewidth=linewidth,
    )

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

data = json.load(open('../data/samples_split.json'))

train_indices = data['train']
val_indices = data['val']


sample_data_train = []
sample_data_val = []

for i in train_indices:
    sample_data_train.append(sample_dataTrain[i])
    # print(sample_data[i])
    # render_sample(sample_data[i]['token'])
    # plt.show()

    # im, boxes = getSampleData(sample_data_train[-1]['token'])
    # fig, axes = plt.subplots(1, 1, figsize=(12, 6))
    # axes.imshow(im)
    # for box in boxes:
    #     render_corners(axes,box,2)
    # plt.show()

for i in val_indices:
    sample_data_val.append(sample_dataVal[i])
    # print(sample_data[i])
    # render_sample(sample_data[i]['token'])
    # plt.show()

print('Inicia división')

# subtrain = random.sample(sample_data_train,500)
# subval = random.sample(sample_data_val,150)
subSamplingDict = {
    'train' : [],
    'val'   : []
}

print('acaba el subsampling')

categories=[]

for t in tqdm(sample_data_train):
    im, boxes, labels, cam = getSampleData(t['token'],trainLyft)
    for i, box in enumerate(boxes):
        print(box.shape, cam.shape)
        box.reshape(-1)
        cam.reshape(-1)
        if labels[i].name not in categories:
            categories.append(labels[i].name)
            print(categories)
        features = np.concatenate((box,cam,np.array([categories.index(labels[i].name)])),axis=None)
        label = np.concatenate((np.array(labels[i].center),np.array(labels[i].wlh),labels[i].orientation.q),axis=None)
        obj={
            "features": features.tolist(),
            "label": label.tolist()
        }
        subSamplingDict['train'].append(obj)

print('acaba de guardar el training')

for v in tqdm(sample_data_val):
    im, boxes, labels,cam = getSampleData(v['token'],valLyft)
    for i, box in enumerate(boxes):
        box.reshape(-1)
        cam.reshape(-1)
        if labels[i].name not in categories:
            categories.append(labels[i].name)
        features = np.concatenate((box,cam,np.array([categories.index(labels[i].name)])),axis=None)
        label = np.concatenate((np.array(labels[i].center),np.array(labels[i].wlh),labels[i].orientation.q),axis=None)
        obj={
            "features": features.tolist(),
            "label": label.tolist()
        }
        subSamplingDict['val'].append(obj)

print('acaba de guardar el validation')

json.dump(subSamplingDict, open('./data/dataset2Dto3D.json', 'w'))

# i=0
# while True:
#     render_ann(sample_ann[i]['token'])
#     i+=1
#     plt.show()