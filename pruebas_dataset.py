from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer
from lyft_dataset_sdk.utils.geometry_utils import BoxVisibility, box_in_image, view_points  # NOQA
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

dataLyft3D = LyftDataset(data_path='./data/', json_path='./data/json', verbose=True)
sample_data = list(filter(lambda x:x['sensor_modality']=='camera',dataLyft3D.sample_data))
sample_ann = dataLyft3D.sample_annotation
sample_ann.pop(4)

def render_ann(ann_token):
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
            render_box(box,axes, view=camera_intrinsic, normalize=True, colors=(c, c, c))

def render_box(box,axis,view,normalize,colors,linewidth=2):
    corners = view_points(box.corners(), view, normalize=normalize)[:2, :]
    print([np.min(corners[:][0]), np.max(corners[:][0])],
          [np.min(corners[:][1]), np.max(corners[:][1])])
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

i=0
while True:
    render_ann(sample_ann[i]['token'])
    i+=1
    plt.show()