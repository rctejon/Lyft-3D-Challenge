from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer
import random
import json

dataLyft3D = LyftDataset(data_path='./data/', json_path='./data/json', verbose=True)
sample_data = list(filter(lambda x:x['sensor_modality']=='camera',dataLyft3D.sample_data))

# print(len(sample_data))

# train_indices = random.sample(range(len(sample_data)),k=int(len(sample_data)*0.7))
# val_indices = [x for x in range(len(sample_data)) if x not in train_indices]

# print(len(train_indices))
# print(len(val_indices))

# IndexDict = {
#     'train': train_indices,
#     'val'  : val_indices 
# }

# json.dump(IndexDict, open('./data/samples_split.json', 'w'))

data = json.load(open('./data/samples_split.json'))

print(list(map(lambda x:(x[0],len(x[1])),list(data.items()))))

train_indices = data['train']

sample_data_train = []

for i in train_indices:
    sample_data_train.append(sample_data[i])