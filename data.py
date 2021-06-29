"""
Initialize dataset for training and testing
"""

import torch
from torchvision import transforms, datasets
from PIL import Image
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, sampler
import glob, os 
import pandas as pd 
# from skimage import io 
import numpy as np 

COVIDX_LABELS = ["normal", "COVID-19", "pneumonia"]


# Balanced data sampler 
# repo: https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/torchsampler/imbalanced.py
class BalancedSampler(sampler.Sampler):
    def __init__(self, dataset, indices=None, num_samples=None, sampling=None):
        """
        dataset: dataset applying this balanced data sampler
        indices: a list of indices of given datset
        num_samples: number of samples to draw 
        sampling: "over" or "down" for oversampling or downsampling
        """
        self.dataset = dataset
        if indices is None:
            self.indices = list(range(len(dataset))) # indexing over dataset
        else:
            self.indices = indices 
        # statistic of class distributions
        label_count = {}
        for idx in self.indices:
            _, label = self.dataset[idx]
            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1
        # assign weights to each label and index
        weights = []
        for idx in self.indices:
            _, label = self.dataset[idx]
            weights.append(1.0/label_count[label])
        self.weights = torch.tensor(weights, dtype=torch.float32)
        # check sampling method
        if sampling == "down":
            # down-sampling: the number of sampling refers to the minimum level
            base_count = min(label_count.values())
        elif sampling == "over":
            base_count = max(label_count.values())
        else:
            base_count = len(dataset) / len(label_count)
        if num_samples is None:
            self.num_samples = int(base_count*len(label_count))  # sampling all dataset 
        else:
            # sampling with given numbers. small value for down-sampling and large value for over-sampling
            self.num_samples = num_samples   

    def __iter__(self):
        return (self.indices[idx] for idx in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


def image_sampler(dataset, training=True, balanced_sampling=None, num_samples=None):
    """Data sampler for training and testing"""
    if balanced_sampling is None:
        sp = None
    else:
        if balanced_sampling == "down":
            sampling = "down"
        elif balanced_sampling == "up":
            sampling = "over"
        else: 
            sampling = None
        sp = BalancedSampler(dataset, sampling=sampling, num_samples=num_samples)
    if not training:
        sp = None 
    return sp


class COVIDX_dataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        self._img_dir = img_dir
        df = pd.read_csv(csv_file, sep=" ", header=None)
        if len(df.columns) == 4:
            df.columns = ["case", "img", "label", "source"]
        elif len(df.columns) == 3:
            df.columns = ["case", "img", "label"]
        else:
            print("unexpected number of input columns", len(df.columns))
        self._img_files = df["img"].tolist()
        self._img_labels = df["label"].tolist()
        self._transform = transform


    def __len__(self):
        return len(self._img_files)

    def __getitem__(self, idx):
        image_name = os.path.join(self._img_dir, self._img_files[idx])
        assert os.path.exists(image_name), "Image file not found!"
        img = Image.open(image_name)
        img = img.convert("RGB")
        # print("img size: ", img.size)
        label = self._img_labels[idx]
        label_id = COVIDX_LABELS.index(label)
        #onehot_id = torch.nn.functional.one_hot(torch.Tensor(label_id), len(LABELS))
        if self._transform:
            img = self._transform(img)
        return img, label_id
    

def prepare_data(input_dir, config):
    """
    config: 
        input_size,
        train: train img file list
        test: test img file list
        dataset: name of dataset to use: covidx or HN
    """
    data_transforms = {
        'train': transforms.Compose([   
            # transforms.Lambda(lambda x: x.crop((0, int(x.height*0.08), x.width, x.height))),  # remove up offset
            transforms.Resize(int(config["image_size"]*1.12)),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(config["image_size"], scale=(0.85, 1.0), ratio=(0.85, 1.15)),
            # transforms.ColorJitter(0.12, 0.12, 0.08, 0.08),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            # transforms.Lambda(lambda x: x.crop((0, int(x.height*0.08), x.width, x.height))),  # remove up offset
            transforms.Resize(config["image_size"]),
            transforms.CenterCrop(config["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    image_datasets = {x: COVIDX_dataset(os.path.join(input_dir, x), config[x], data_transforms[x]) for x in ["train", "test"]}
    # class_names = image_datasets["train"].classes 
    data_sizes = {x: len(image_datasets[x]) for x in ["train", "test"]}
    return image_datasets, data_sizes

# resize image within the min-max sizes
class MinMaxResize(object):
    def __init__(self, min_size, max_size):
        self._min_size = min_size
        self._max_size = max_size
    
    def __call__(self, x):
        w, h = x.size
        min_edge = min(h, w)
        max_edge = max(h, w)
        if min_edge < self._min_size:
            scale = self._min_size / min_edge
            x = x.resize((int(w*scale+1), int(h*scale+1)))
        if max_edge > self._max_size:
            scale = self._max_size / max_edge
            x = x.resize((int(w*scale+1), int(h*scale+1)))
        return x


if __name__ == "__main__":
    import cv2
    image_dir = "/Users/zongfan/Projects/data/covidx"
    image_dir = "/Users/zongfan/Projects/data/covidx_test"
    config = {"image_size": 256, "train": "train_sample.txt", "test": "train_sample.txt"}
    ds, _ = prepare_data(image_dir, config)
    batch_size = 2
    dataloader = torch.utils.data.DataLoader(ds["train"], batch_size=batch_size)
    for imgs, label in dataloader:
        print(imgs.shape)
        for img in imgs:
            x = np.transpose(img.numpy(), (1, 2, 0))
            x = (x + 1.) / 2. * 255
            x = x.astype(np.uint8)
            x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
            cv2.imshow("image", x)
            # press q to exit window
            if cv2.waitKey(0) == ord("q"):
                break




