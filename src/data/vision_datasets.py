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
class COCODataset2(Dataset):
    def __init__(self, image_dir, anno_path, input_transform=None, 
                    labels_path=None,
                    used_category=-1):
        self.coco = torchvision.datasets.CocoDetection(root=image_dir, annFile=anno_path)
        # with open('./data/coco/category.json','r') as load_category:
        #     self.category_map = json.load(load_category)
        self.category_map = category_map
        self.input_transform = input_transform
        self.labels_path = labels_path
        self.used_category = used_category
	
        self.labels = []
        if os.path.exists(self.labels_path):
            self.labels = np.load(self.labels_path).astype(np.float64)
            self.labels = (self.labels > 0).astype(np.float64)
        else:
            print("No preprocessed label file found in {}.".format(self.labels_path))
            l = len(self.coco)
            for i in tqdm(range(l)):
                item = self.coco[i]
                # print(i)
                categories = self.getCategoryList(item[1])
                label = self.getLabelVector(categories)
                self.labels.append(label)
            self.save_datalabels(labels_path)
        # import ipdb; ipdb.set_trace()

    def __getitem__(self, index):
        input = self.coco[index][0]
        if self.input_transform:
            input = self.input_transform(input)
        #return input, self.labels[index]

        return {'input_ids': input, 'labels': self.labels[index]}


    def getCategoryList(self, item):
        categories = set()
        for t in item:
            categories.add(t['category_id'])
        return list(categories)

    def getLabelVector(self, categories):
        label = np.zeros(80)
        # label_num = len(categories)
        for c in categories:
            index = self.category_map[str(c)]-1
            label[index] = 1.0 # / label_num
        return label

    def __len__(self):
        return len(self.coco)

    def save_datalabels(self, outpath):
        """
            Save datalabels to disk.
            For faster loading next time.
        """
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        labels = np.array(self.labels)
        np.save(outpath, labels)



#################################### Flickr Dataset ####################################


def filter_images_with_labels(df, labels_file):
    labels_df = pd.read_csv(labels_file)
    labels_df.set_index('image_id', inplace=True)
    
    valid_image_ids = labels_df[labels_df.sum(axis=1) > 0].index.tolist()
    filtered_df = df[df['image_id'].isin(valid_image_ids)]
    
    return filtered_df


def download_and_extract_mirflickr25k(download_dir="mirflickr25k"):
    # URLs for the dataset and annotations
    image_collection_url = "http://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k.zip"
    annotations_url = "http://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k_annotations_v080.zip"

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        # Create directory to save the downloaded files
        os.makedirs(download_dir, exist_ok=True)

        def download_file(url, dest):
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

            with open(dest, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()

            if total_size != 0 and progress_bar.n != total_size:
                print("ERROR: Something went wrong while downloading", url)
            else:
                print("Downloaded", url)

        def unzip_file(file_path, extract_to):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
                print(f"Unzipped {file_path} to {extract_to}")

        # Download the image collection
        image_collection_dest = os.path.join(download_dir, "mirflickr25k.zip")
        download_file(image_collection_url, image_collection_dest)

        # Download the annotations
        annotations_dest = os.path.join(download_dir, "mirflickr25k_annotations_v080.zip")
        download_file(annotations_url, annotations_dest)

        # Unzip the downloaded files
        unzip_file(image_collection_dest, download_dir)
        unzip_file(annotations_dest, download_dir)

        print("Download and extraction complete. Files are saved in:", download_dir)
    else:
        print(f"Dataset folder {download_dir} already exists. Skipping download and extraction.")

# Function to precompute labels for Flickr dataset
def precompute_labels_flickr(root_dir, labels_file):
    print(f"Precomputing labels for Flickr dataset...")
    
    label_names = [
        'sky', 'clouds', 'water', 'sea', 'river', 'lake', 'people', 'portrait',
        'male', 'female', 'baby', 'night', 'plant_life', 'tree', 'flower', 'animals',
        'dog', 'bird', 'structures', 'sunset', 'indoor', 'transport', 'car'
    ]
    
    # Load the labels from text files
    labels = {label: load_label_file(os.path.join(root_dir, f'{label}.txt')) for label in label_names}
    num_classes = len(label_names)
    
    data = []
    for image_id in range(1, 25001):  # Assuming image IDs from 1 to 25000
        label = torch.zeros(num_classes)
        for idx, label_name in enumerate(label_names):
            if image_id in labels[label_name]:
                label[idx] = 1
        data.append([image_id] + label.tolist())
    
    columns = ['image_id'] + label_names
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(labels_file, index=False)
    print(f"Labels saved to {labels_file}.")

def load_label_file(filepath):
    with open(filepath, 'r') as f:
        return [int(line.strip()) for line in f]

# Function to save the split information to CSV
def save_flickr_splits(root_dir, train_size=20000, val_size=2500, test_size=2500, random_state=42):
    all_image_ids = list(range(1, 25001))  # Assuming image IDs from 1 to 25000
    train_ids, temp_ids = train_test_split(all_image_ids, train_size=train_size, random_state=random_state)
    val_ids, test_ids = train_test_split(temp_ids, test_size=test_size, random_state=random_state)

    df = pd.DataFrame({'image_id': train_ids + val_ids + test_ids,
                       'split': ['train'] * len(train_ids) + ['val'] * len(val_ids) + ['test'] * len(test_ids)})
    
    labels_file = os.path.join(root_dir, 'precomputed_labels_flickr.csv')
    df = filter_images_with_labels(df, labels_file)

    df.to_csv(os.path.join(root_dir, 'flickr25k_splits.csv'), index=False)
    return df

class FlickrDataset(Dataset):
    label_names = [
        'sky', 'clouds', 'water', 'sea', 'river', 'lake', 'people', 'portrait',
        'male', 'female', 'baby', 'night', 'plant_life', 'tree', 'flower', 'animals',
        'dog', 'bird', 'structures', 'sunset', 'indoor', 'transport', 'car'
    ]

    def __init__(self, root_dir, split, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_dir = os.path.join(root_dir, 'mirflickr')
        self.split = split

        # Load split information
        split_df = pd.read_csv(os.path.join(root_dir, 'flickr25k_splits.csv'))
        self.image_ids = split_df[split_df['split'] == split]['image_id'].tolist()

        # Load precomputed labels
        labels_file = os.path.join(root_dir, 'precomputed_labels_flickr.csv')
        labels_df = pd.read_csv(labels_file)
        self.labels = labels_df.set_index('image_id').loc[self.image_ids].to_numpy(dtype=np.float32)

        self.num_classes = len(self.label_names)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, f'im{image_id}.jpg')
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx])
        return {'input_ids': image, 'labels': label}
    
    
    
######### NUS-WIDE DATASET #############

def download_and_extract_nuswide(download_dir="nuswide"):
    if not os.path.exists(download_dir):
        print(f"Downloading NUS-WIDE dataset to {download_dir}...")
        #dl from this kaggle repo https://www.kaggle.com/datasets/xinleili/nuswide
        raise NotImplementedError("Download the dataset from previous  Kaggle link pls")
        pass
    else:
        print(f"Dataset folder {download_dir} already exists. Skipping download and extraction.")



        

def save_nuswide_splits(root_dir, val_size=0.5, random_state=42):
    """
    Save the splits for the NUS-WIDE dataset into validation and test sets,
    filtering out examples without labels and creating separate CSV files for
    filtered train, validation, and test datasets.
    """
    def filter_images_with_labels(df):
        # Filter out examples that don't have any labels
        filtered_df = df[df.iloc[:, 2:].sum(axis=1) > 0]
        return filtered_df

    train_csv = os.path.join(root_dir, 'train.csv')
    test_csv = os.path.join(root_dir, 'test.csv')

    # Read the train and test CSV files
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Filter out examples without labels
    train_df = filter_images_with_labels(train_df)
    test_df = filter_images_with_labels(test_df)

    # Save the filtered training set
    filtered_train_csv = os.path.join(root_dir, 'filtered_train.csv')
    train_df.to_csv(filtered_train_csv, index=False)

    # Save the filtered test set
    filtered_test_csv = os.path.join(root_dir, 'filtered_test.csv')
    test_df.to_csv(filtered_test_csv, index=False)

    # Split the filtered test data into validation and test sets
    test_indices = test_df.index.tolist()
    val_indices, test_indices = train_test_split(test_indices, test_size=val_size, random_state=random_state)

    # Save the splits
    splits = {
        'val': val_indices,
        'test': test_indices
    }

    split_file = os.path.join(root_dir, 'val_test_splits.csv')
    pd.DataFrame(splits).to_csv(split_file, index=False)

    print(f"Filtered training set saved to {filtered_train_csv}.")
    print(f"Filtered test set saved to {filtered_test_csv}.")
    print(f"Validation and test splits saved to {split_file}.")

class NuswideDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        if split == 'train':
            csv_file = os.path.join(root_dir, 'filtered_train.csv')
        else:
            csv_file = os.path.join(root_dir, 'test.csv')
            split_file = os.path.join(root_dir, 'val_test_splits.csv')

            # Load split indices
            split_indices = pd.read_csv(split_file)
            self.indices = split_indices[split].dropna().astype(int).tolist()

        # Load the data from CSV
        self.data = pd.read_csv(csv_file)

        if split != 'train':
            self.data = self.data.iloc[self.indices]

        self.images = self.data['imageid'].values
        self.labels = self.data.drop(columns=['imageid', 'phase', 'num_label']).values

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 'images', self.images[idx] + '.jpg')
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return {'input_ids': image, 'labels': label}


def collate_fn_double(batch):
    # These two lines are how the default collate function works for tensors 
    inputs = torch.stack([item['input_ids'] for item in batch]) 
    labels = torch.stack([item['labels'] for item in batch])

    # We double the inputs and labels for contrastive loss
    inputs = torch.cat([inputs, inputs], dim=0)
    labels = torch.cat([labels, labels], dim=0)
    
    return {'input_ids': inputs, 'labels': labels}
    




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
    elif augmentation == 'randaugment2':
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
    elif dataset == 'coco2': 
        root_dir = './data/data/coco2014'
        train_transform_list = [transforms.Resize((img_size, img_size)),
                                            RandAugment(),
                                               transforms.ToTensor(),
                                               normalize]

        train_transform = transforms.Compose(train_transform_list)
        test_transform = transforms.Compose([
                                            transforms.Resize((img_size, img_size)),
                                            transforms.ToTensor(),
                                            normalize])
        train_dataset = COCODataset2(
            image_dir=os.path.join(root_dir, 'train2014'),
            anno_path=os.path.join(root_dir, 'annotations/instances_train2014.json'),
            input_transform=train_transform,
            labels_path='data/coco/train_label_vectors_coco14.npy',
        )
        val_dataset = COCODataset2(
            image_dir= os.path.join(root_dir, 'val2014'),
            anno_path=os.path.join(root_dir, 'annotations/instances_val2014.json'),
            input_transform=test_transform,
            labels_path='data/coco/val_label_vectors_coco14.npy',
        ) 
        test_dataset = COCODataset2(
            image_dir= os.path.join(root_dir, 'val2014'),
            anno_path=os.path.join(root_dir, 'annotations/instances_val2014.json'),
            input_transform=test_transform,
            labels_path='data/coco/val_label_vectors_coco14.npy',
        )   
            
       
        nb_labels = 80
    elif dataset == 'flickr25k':
        root_dir = './data/data/mirflickr25k'
        download_and_extract_mirflickr25k(root_dir)
        labels_file = os.path.join(root_dir, 'precomputed_labels_flickr.csv')
        if not os.path.exists(labels_file):
            precompute_labels_flickr(root_dir, labels_file)
        
        save_flickr_splits(root_dir)

        train_dataset = FlickrDataset(root_dir=root_dir, split='train', transform=train_transform)
        val_dataset = FlickrDataset(root_dir=root_dir, split='val', transform=val_transform)
        test_dataset = FlickrDataset(root_dir=root_dir, split='test', transform=val_transform)

        nb_labels = len(FlickrDataset.label_names)
        print(f"Number of labels in Flickers: {nb_labels}") 
        
    elif dataset == 'nuswide':
        root_dir = './data/data/nuswide'
        download_and_extract_nuswide(root_dir)
        if not os.path.exists(os.path.join(root_dir, 'val_test_splits.csv')) or not os.path.exists(os.path.join(root_dir, 'filtered_train.csv')) or not os.path.exists(os.path.join(root_dir, 'filtered_test.csv')):
            save_nuswide_splits(root_dir)
        else:
            print("Splits already created.")

        train_dataset = NuswideDataset(root_dir=root_dir, split='train', transform=train_transform)
        val_dataset = NuswideDataset(root_dir=root_dir, split='val', transform=val_transform)
        test_dataset = NuswideDataset(root_dir=root_dir, split='test', transform=val_transform)

        nb_labels = train_dataset.labels.shape[1]
        print(f"Number of labels in NUS-WIDE: {nb_labels}") #81

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

    first_image_tensor = train_dataset[0]['input_ids']
    first_image = first_image_tensor.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)

    # Compute the average pixel values across the color channels (RGB)
    avg_pixel_values = first_image.mean(axis=(0, 1))

    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(f"Number of training samples: {len(train_dataset)} (fraction {fraction})")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")
    print(f"Number of labels: {nb_labels}")
    print(f"Average pixel values of the first image: {avg_pixel_values}")
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    # plot the average of the pixel value of the first image on the dataset to make sure that this image is the same for different call of the function


    


    train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader, nb_labels



label_names = [
    'airport', 'animal', 'beach', 'bear', 'birds', 'boats', 'book', 'bridge', 'buildings', 'cars', 'castle', 'cat',
    'cityscape', 'clouds', 'computer', 'coral', 'cow', 'dancing', 'dog', 'earthquake', 'elk', 'fire', 'fish', 'flags',
    'flowers', 'food', 'fox', 'frost', 'garden', 'glacier', 'grass', 'harbor', 'horses', 'house', 'lake', 'leaf', 'map',
    'military', 'moon', 'mountain', 'nighttime', 'ocean', 'person', 'plane', 'plants', 'police', 'protest', 'railroad',
    'rainbow', 'reflection', 'road', 'rocks', 'running', 'sand', 'sign', 'sky', 'snow', 'soccer', 'sports', 'statue',
    'street', 'sunset', 'sun', 'surf', 'swimmers', 'tattoo', 'temple', 'tiger', 'tower', 'town', 'toy', 'train', 'tree',
    'valley', 'vehicle', 'waterfall', 'water', 'wedding', 'whales', 'window', 'zebra'
]


def show_images_from_loader(data_loader, num_images=5):
    data_iter = iter(data_loader)
    images_shown = 0
    
    fig, axes = plt.subplots(1, num_images, figsize=(20, 5))
    while images_shown < num_images:
        batch = next(data_iter)
        images, labels = batch['input_ids'], batch['labels']
        for i in range(min(num_images - images_shown, len(images))):
            image = images[i].permute(1, 2, 0).numpy()  # Change to HWC for visualization
            label_indices = labels[i].nonzero().flatten().tolist()
            label_names_display = [label_names[idx] for idx in label_indices]
            axes[images_shown].imshow(image)
            axes[images_shown].set_title(f"Labels: {label_names_display}")
            axes[images_shown].axis('off')
            images_shown += 1
            if images_shown >= num_images:
                break
    plt.show()


def check_loss_function(data_loader, criterion):
    for batch in data_loader:
        inputs, labels = batch['input_ids'], batch['labels']
        outputs = torch.randn_like(labels)  # Example model outputs
        loss = criterion(outputs, labels)
        print(f"Loss: {loss.item()}")
        print("Labels shape:", labels.shape)  # Should be [batch_size, num_labels]
        print("Outputs shape:", outputs.shape)  # Should be [batch_size, num_labels]
        print("Unique label values:", labels.unique())
        break

def check_dataset():
    batch_size = 64
    img_size = 224
    augmentation = 'randaugment'  # or 'basic', 'randaugment2'
    seed = 42

    train_loader, val_loader, test_loader, nb_labels = get_vision_loaders('nuswide', batch_size, img_size, augmentation, seed)

    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    test_dataset = test_loader.dataset

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")
    print(f"Number of labels in NUS-WIDE: {nb_labels}")

   
    # Check loss function
    criterion = torch.nn.BCEWithLogitsLoss()
    print("\nChecking loss function on training set:")
    check_loss_function(train_loader, criterion)

    print("\nChecking loss function on validation set:")
    check_loss_function(val_loader, criterion)

    print("\nChecking loss function on test set:")
    check_loss_function(test_loader, criterion)

    # Display some images and their labels from each dataset
    print("\nDisplaying some training images and their labels after augmentation:")
    show_images_from_loader(train_loader)

    print("\nDisplaying some validation images and their labels:")
    show_images_from_loader(val_loader)

    print("\nDisplaying some test images and their labels:")
    show_images_from_loader(test_loader)
    
    
if __name__ == '__main__':
    # print values of images in nuswide: 
    #dataset = NuswideDataset(root_dir='./data/data/nuswide', split='train', transform=None)
    # same for voc 
    train_loader, val_loader, test_loader, num_labels = get_vision_loaders('nuswide', 32, 224, 'randaugment', 42)
    for batch in train_loader:
        image_tensor = batch['input_ids']
        labels = batch['labels']
        break
    
    # image = dataset[0]['input_ids']
    # transform = transforms.ToTensor()

    # # Apply the transform to convert the image to a tensor
    # image_tensor = transform(image)
    # print(image_tensor)
    # # #print the mean of the image tensor and std across 3 dimensions
    # print(image_tensor.mean(dim=(1, 2)))
    # print(image_tensor.std(dim=(1, 2)))
    print(image_tensor.shape) # torch.Size([32, 3, 224, 224])
    #print mean for all images of the bach for each channel
    print(image_tensor.mean(dim=(0, 2, 3)))
    #print std 
    print(image_tensor.std(dim=(0, 2, 3)))


    
    
    
    # def show_images(images, labels, train_loader):
    #     plt.figure(figsize=(15, 7))
    #     for i, (image, label) in enumerate(zip(images, labels)):
    #         present_indices = np.where(label.numpy() == 1)[0]
    #         label_names = [FlickrDataset.label_names[idx] for idx in present_indices]

    #         plt.subplot(2, 5, i + 1)
    #         plt.imshow(image.permute(1, 2, 0))
    #         plt.title("Labels: " + ", ".join(label_names))
    #         plt.axis('off')
    #     plt.show()

    # train_loader, val_loader, test_loader, nb_labels = get_vision_loaders('flickr25k', 10, 128, 'basic', seed=38)

    # Print labels examples
    # for batch in train_loader:
    #     images = batch['input_ids']
    #     labels = batch['labels']
    #     show_images(images, labels, train_loader)
    #     break
    
    # add a function that display the number of examples in all three datasets with 0 labels
    # for loader, name in zip([train_loader, val_loader, test_loader], ['train', 'val', 'test']):
    #     zero_labels = 0
    #     for batch in loader:
    #         labels = batch['labels']
    #         zero_labels += (labels.sum(dim=1) == 0).sum().item()
    #     print(f"Number of examples with 0 labels in {name} set: {zero_labels}")


    # # make a function that call every download and extract function for all datasets
    # download_and_extract_coco('./data/data/coco2014')
    # download_and_extract_voc('./data/data/voc2007')
    # download_and_extract_mirflickr25k('./data/data/mirflickr25k')
    # download_and_extract_nuswide('./data/data/nuswide')
    


# script to dl voc
# import requests
# import os
# import tarfile
# download_path = './data/data/voc2007'
# VOCURL = [
#         "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
#         "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"
#     ]
# if not os.path.exists(download_path):
#     os.makedirs(download_path)
# for url in VOCURL:
#     tar_path = os.path.join(download_path, url.split('/')[-1])

#     # Download the dataset
#     print(f"Downloading {url} to {tar_path}...")
#     with requests.get(url, stream=True) as r:
#         r.raise_for_status()
#         with open(tar_path, 'wb') as f:
#             for chunk in r.iter_content(chunk_size=8192):
#                 f.write(chunk)

#     # Extract the dataset
#     print(f"Extracting {tar_path} to {download_path}...")
#     with tarfile.open(tar_path, 'r') as tar_ref:
#         tar_ref.extractall(download_path)
#     print("Download and extraction complete.")