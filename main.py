<<<<<<< HEAD
import detectron2
=======
from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer
from lyft_dataset_sdk.utils.geometry_utils import BoxVisibility, box_in_image, view_points  # NOQA
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import json

level5data = LyftDataset(data_path='./data/', json_path='./data/json', verbose=True)

my_scene = level5data.scene[0]
my_scene

my_sample_token = my_scene["first_sample_token"]
# my_sample_token = level5data.get("sample", my_sample_token)["next"]  # proceed to next sample

level5data.render_sample(my_sample_token)
plt.show()
>>>>>>> 95a7d59c57143224d70e22a18248f9329adfcda3
