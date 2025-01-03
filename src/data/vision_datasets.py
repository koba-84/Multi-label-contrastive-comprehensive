import os
import csv
import requests
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
from xml.etree import ElementTree as ET
import tarfile
from matplotlib import pyplot as plt
import zipfile
import json
from tqdm import tqdm
from torch.utils.data import Subset


from torchvision.transforms import InterpolationMode
from torchvision.transforms import RandAugment

#from utils.utils import set_seed

####### VOC DATASET  ########

def download_and_extract_voc(download_path):
    VOCURL = [
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"
    ]

    if not os.path.exists(download_path):
        os.makedirs(download_path)
        for url in VOCURL:
            tar_path = os.path.join(download_path, url.split('/')[-1])

            # Download the dataset
            print(f"Downloading {url} to {tar_path}...")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(tar_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            # Extract the dataset
            print(f"Extracting {tar_path} to {download_path}...")
            with tarfile.open(tar_path, 'r') as tar_ref:
                tar_ref.extractall(download_path)
            print("Download and extraction complete.")
    else:
        print(
            f"Dataset folder {download_path} already exists. Skipping download and extraction.")


class VOCdataset(Dataset):
    def __init__(self, root_dir, image_set='trainval', transform=None, split='train'):
        self.root_dir = root_dir
        self.image_set = image_set
        self.transform = transform
        self.split = split
        self.labels_file = os.path.join(
            self.root_dir, f'voc_2007_{image_set}_{split}_labels.csv')
        self.class_name = [
            "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]

        self.images = self.load_voc_dataset()

        if not os.path.exists(self.labels_file):
            print("Labels file does not exist. Creating new.")
            self.create_labels_csv()
        else:
            print("Labels file exists. Loading from CSV.")

        self.labels = pd.read_csv(self.labels_file, header=None).values

        if image_set == 'test':
            split_file = os.path.join(self.root_dir, 'val_test_splits.csv')
            val_indices, test_indices = self.load_or_create_splits(split_file)
            if split == 'val':
                self.indices = val_indices
            else:
                self.indices = test_indices
        else:
            self.indices = list(range(len(self.images)))

    def load_voc_dataset(self):
        images = []
        image_dir = os.path.join(
            self.root_dir, 'VOCdevkit', 'VOC2007', 'JPEGImages')

        image_set_file = os.path.join(
            self.root_dir, 'VOCdevkit', 'VOC2007', 'ImageSets', 'Main', f'{self.image_set}.txt')
        with open(image_set_file, 'r') as file:
            for line in file:
                image_id = line.strip()
                images.append(os.path.join(image_dir, f'{image_id}.jpg'))

        return images

    def extract_labels_from_xml(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        labels = [0] * 20
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name in self.class_name:
                labels[self.class_name.index(name)] = 1
        return labels

    def create_labels_csv(self):
        annotations_dir = os.path.join(
            self.root_dir, 'VOCdevkit', 'VOC2007', 'Annotations')

        with open(self.labels_file, 'w', newline='') as file:
            writer = csv.writer(file)
            for image_path in self.images:
                image_id = os.path.basename(image_path).split('.')[0]
                xml_file = os.path.join(annotations_dir, f'{image_id}.xml')
                labels = self.extract_labels_from_xml(xml_file)
                writer.writerow(labels)

    def load_or_create_splits(self, file_path):
        if os.path.exists(file_path):
            print("Loading existing splits for val/test.")
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                indices = list(reader)
            val_indices = list(map(int, indices[0]))
            test_indices = list(map(int, indices[1]))
        else:
            print("Creating a new split for val/test.")
            num_images = len(self.images)
            indices = list(range(num_images))
            val_indices, test_indices = train_test_split(
                indices, test_size=0.5, random_state=42)
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(val_indices)
                writer.writerow(test_indices)
        return val_indices, test_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        image_path = self.images[actual_idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        labels = torch.tensor(self.labels[actual_idx], dtype=torch.float32)
        return {'input_ids': image, 'labels': labels}


######### COCO DATASET #############

def download_and_extract_coco(download_path):
    COCOURL = {
        "train": "http://images.cocodataset.org/zips/train2014.zip",
        "val": "http://images.cocodataset.org/zips/val2014.zip",
        "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    }

    if not os.path.exists(download_path):
        os.makedirs(download_path)
    
        for split, url in COCOURL.items():
            zip_path = os.path.join(download_path, url.split('/')[-1])

            # Download the dataset or annotations
            if not os.path.exists(zip_path):
                print(f"Downloading {url} to {zip_path}...")
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with open(zip_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
            else:
                print(f"{zip_path} already exists. Skipping download.")

            # Extract the dataset or annotations
            extract_path = os.path.join(download_path, split)
            if not os.path.exists(extract_path):
                print(f"Extracting {zip_path} to {download_path}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(download_path)
                print(f"Extraction of {zip_path} complete.")
            else:
                print(f"{extract_path} already exists. Skipping extraction.")
    else:
        print(
            f"Dataset folder {download_path} already exists. Skipping download and extraction.")


def load_coco_annotations(json_file):
    with open(json_file, 'r') as f:
        annotations = json.load(f)
    
    # Extract categories
    categories = {cat['id']: cat['name'] for cat in annotations['categories']}
    category_id_to_index = {cat['id']: idx for idx, cat in enumerate(annotations['categories'])}
    index_to_category_id = {idx: cat['id'] for idx, cat in enumerate(annotations['categories'])}
    
    # Extract image to category mapping
    image_to_categories = {}
    for ann in annotations['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id']
        if image_id not in image_to_categories:
            image_to_categories[image_id] = []
        image_to_categories[image_id].append(category_id)
    
    return categories, image_to_categories, category_id_to_index, index_to_category_id

def save_val_test_split_coco(root_dir, val_annotations_file, split_file='val_test_splits.csv'):
    categories, image_to_categories, _, _ = load_coco_annotations(os.path.join(root_dir, val_annotations_file))
    image_ids = list(image_to_categories.keys())
    
    train_ids, test_ids = train_test_split(image_ids, test_size=0.5, random_state=42)
    val_ids, test_ids = train_test_split(test_ids, test_size=0.5, random_state=42)
    
    df = pd.DataFrame({'image_id': val_ids + test_ids,
                       'split': ['val'] * len(val_ids) + ['test'] * len(test_ids)})
    
    df.to_csv(os.path.join(root_dir, split_file), index=False)
    return df

def precompute_labels_coco(root_dir, annotations_file, output_file):
    print(f"Precomputing labels for {annotations_file}...")
    categories, image_to_categories, category_id_to_index, _ = load_coco_annotations(os.path.join(root_dir, annotations_file))
    num_classes = len(category_id_to_index)
    
    data = []
    for image_id, category_ids in image_to_categories.items():
        label = torch.zeros(num_classes)
        for category_id in category_ids:
            label[category_id_to_index[category_id]] = 1
        data.append([image_id] + label.tolist())
    
    columns = ['image_id'] + [f'class_{i}' for i in range(num_classes)]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_file, index=False)
    print(f"Labels saved to {output_file}.")

class COCODataset(Dataset):
    def __init__(self, root_dir, split, transform=None, labels_file='precomputed_labels.csv'):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform 
        
        if split == 'train':
            annotations_file = 'annotations/instances_train2014.json'
            self.img_dir = os.path.join(root_dir, 'train2014')
            self.prefix = 'COCO_train2014_'
            labels_file = os.path.join(root_dir, 'precomputed_train_labels.csv')
        else:
            annotations_file = 'annotations/instances_val2014.json'
            self.img_dir = os.path.join(root_dir, 'val2014')
            self.prefix = 'COCO_val2014_'
            labels_file = os.path.join(root_dir, 'precomputed_val_labels.csv')
        
        self.categories, _, self.category_id_to_index, self.index_to_category_id = load_coco_annotations(os.path.join(root_dir, annotations_file))
        self.num_classes = len(self.category_id_to_index)
        
        if split in ['val', 'test']:
            split_df = pd.read_csv(os.path.join(root_dir, 'val_test_splits.csv'))
            self.image_ids = split_df[split_df['split'] == split]['image_id'].tolist()
        else:
            self.image_ids = list(pd.read_csv(labels_file)['image_id'])
        
        # Load precomputed labels
        print(f"Loading labels from {labels_file}...")
        labels_df = pd.read_csv(labels_file)
        labels_df.set_index('image_id', inplace=True)
        self.labels = labels_df.loc[self.image_ids].iloc[:, 0:].to_numpy(dtype=np.float32)
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, f'{self.prefix}{image_id:012d}.jpg')
        
        image = Image.open(img_path).convert('RGB')
        
        # Get the precomputed label
        label = torch.tensor(self.labels[idx])
        
        if self.transform:
            image = self.transform(image)
        
        return {'input_ids': image, 'labels': label}

category_map = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11, "13": 12, "14": 13, "15": 14, "16": 15, "17": 16, "18": 17, "19": 18, "20": 19, "21": 20, "22": 21, "23": 22, "24": 23, "25": 24, "27": 25, "28": 26, "31": 27, "32": 28, "33": 29, "34": 30, "35": 31, "36": 32, "37": 33, "38": 34, "39": 35, "40": 36, "41": 37, "42": 38, "43": 39, "44": 40, "46": 41, "47": 42, "48": 43, "49": 44, "50": 45, "51": 46, "52": 47, "53": 48, "54": 49, "55": 50, "56": 51, "57": 52, "58": 53, "59": 54, "60": 55, "61": 56, "62": 57, "63": 58, "64": 59, "65": 60, "67": 61, "70": 62, "72": 63, "73": 64, "74": 65, "75": 66, "76": 67, "77": 68, "78": 69, "79": 70, "80": 71, "81": 72, "82": 73, "84": 74, "85": 75, "86": 76, "87": 77, "88": 78, "89": 79, "90": 80}
    




# Main data loader function
def get_vision_loaders(dataset, batch_size, img_size, augmentation, seed, collate='simple', num_workers=8, pin_memory=True, fraction=1.0):
    def set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    set_seed(seed)

    # Define normalization transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Initialize variables
    train_transform = None
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])

    # Define transformations based on augmentation type
    if augmentation == 'basic':
        train_transform = val_transform
    elif augmentation == 'randaugment':
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            RandAugment(num_ops=2, magnitude=9, num_magnitude_bins=31, interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
            normalize
        ])
    elif augmentation == 'randaugment_strong':
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            RandAugment(num_ops=3, magnitude=15, num_magnitude_bins=31, interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
            normalize
        ])
    else:
        raise ValueError(f"Augmentation {augmentation} not supported.")

    if dataset == 'voc2007':
        root_dir = './data/data/voc2007'
        download_and_extract_voc(root_dir)
        train_dataset = VOCdataset(root_dir=root_dir, image_set='trainval', transform=train_transform, split='train')
        val_dataset = VOCdataset(root_dir=root_dir, image_set='test', transform=val_transform, split='val')
        test_dataset = VOCdataset(root_dir=root_dir, image_set='test', transform=val_transform, split='test')
        nb_labels = 20

    elif dataset == 'coco2014':
        root_dir = './data/data/coco2014'
        download_and_extract_coco(root_dir)
        split_file = os.path.join(root_dir, 'val_test_splits.csv')
        train_labels_file = os.path.join(root_dir, 'precomputed_train_labels.csv')
        val_labels_file = os.path.join(root_dir, 'precomputed_val_labels.csv')
        
        if not os.path.exists(split_file):
            save_val_test_split_coco(root_dir, 'annotations/instances_val2014.json')
        if not os.path.exists(train_labels_file):
            precompute_labels_coco(root_dir, 'annotations/instances_train2014.json', train_labels_file)
        if not os.path.exists(val_labels_file):
            precompute_labels_coco(root_dir, 'annotations/instances_val2014.json', val_labels_file)

        train_dataset = COCODataset(root_dir=root_dir, split='train', transform=train_transform, labels_file=train_labels_file)
        val_dataset = COCODataset(root_dir=root_dir, split='val', transform=val_transform, labels_file=val_labels_file)
        test_dataset = COCODataset(root_dir=root_dir, split='test', transform=val_transform, labels_file=val_labels_file)

        nb_labels = 80
    

    else:
        raise ValueError(f"Dataset {dataset} not supported.")

    effective_batch_size = batch_size // 2 if collate == 'double' else batch_size
    collate_fn = collate_fn_double if collate == 'double' else None #test de mettre les exemples en doubles pour garantir l'existence d'un positif (pas de gros changement en pratique a premiere vue)

    if fraction < 1.0:
        num_train_samples = int(len(train_dataset) * fraction)
        indices = list(range(len(train_dataset)))
        np.random.shuffle(indices)  # Ensure reproducibility
        subset_indices = indices[:num_train_samples]
        train_dataset = Subset(train_dataset, subset_indices)


    train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader, nb_labels




