import sys
import os
import random
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from collections import defaultdict
import numpy as np
import torch
import lmdb
import six
import time
from tqdm import tqdm
import json
import base64

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from vietocr.loader.s3_download import S3StorageService, ConfigS3, b64_2_img, b64_2_str
from vietocr.tool.translate import process_image
from vietocr.tool.create_dataset import createDataset
from vietocr.tool.translate import resize

class OCRDataset(Dataset):
    def __init__(self, lmdb_path, root_dir, annotation_path, vocab, label_studio_path=None,  image_height=32, image_min_width=32, image_max_width=512, transform=None, check_field=None):
        self.root_dir = root_dir
        if annotation_path:
            self.annotation_path = os.path.join(root_dir, annotation_path)
        else:
            self.annotation_path = None
        if label_studio_path:
            self.label_studio_path = os.path.join(root_dir, label_studio_path)
        else:
            self.label_studio_path = None
        self.vocab = vocab
        self.transform = transform
        self.check_field = check_field

        self.image_height = image_height
        self.image_min_width = image_min_width
        self.image_max_width = image_max_width

        self.lmdb_path =  lmdb_path

        self.s3 = S3StorageService()

        self.init_an()
    
    def init_an(self):
        self.annotations = []
        if self.label_studio_path:
            b64 = self.s3.download_file(ConfigS3.S3_BUCKET_NAME, self.label_studio_path)['b64']
            paths = b64_2_str(b64).split("\n")
            for path in paths:
                path = os.path.join(self.root_dir, path)
                self.annotations.extend(self.get_label(path))
        
        if self.annotation_path:
            b64 = self.s3.download_file(ConfigS3.S3_BUCKET_NAME, self.annotation_path)['b64']
            lines = b64_2_str(b64).split("\n")
            for line in lines:
                line = line.replace("\r", "")
                annotation = line.split("\t")
                self.annotations.append(annotation)
        
        self.nSamples = len(self.annotations)

        self.build_cluster_indices()


    def build_cluster_indices(self):
        self.cluster_indices = defaultdict(list)

        for i in range(self.__len__()):
            bucket = self.get_bucket(i)
            self.cluster_indices[bucket].append(i)
            
    def get_label(self, path):
        res = []
        
        json_file = self.s3.download_file(ConfigS3.S3_BUCKET_NAME, path)['b64']
        data = base64.b64decode(json_file)
        data = data.decode('utf8')
        data = json.loads(data)
        
        id = data["task"]["id"]

        for i in range(2, len(data['result']), 3):
            d = data['result'][i]
            field = data['result'][i-1]['value']['labels'][0]
            if self.check_field != None and field != self.check_field: continue
            
            original_width = d['original_width']
            original_height = d['original_height']
            x = d['value']['x']*original_width / 100
            y = d['value']['y']*original_height / 100
            w = d['value']['width']*original_width / 100
            h = d['value']['height']*original_height / 100
            
            xmin = int(x)
            ymin = int(y)
            xmax = int(x+w) + 1
            ymax = int(y+h) + 1

            label = d['value']['text'][0]
            
            img_path = data["task"]["data"]["ocr"].replace("s3://datalake/dataset/OCR/", "")
            res.append([os.path.join(self.root_dir, img_path), label, xmin, ymin, xmax, ymax, field, id])

        return res
    
    def get_crop_img(self, img, coors):
        xmin, ymin, xmax, ymax = coors
        img = np.array(img)
        crop_img = img[ymin:ymax, xmin:xmax]
        return Image.fromarray(crop_img)
        

    
    def get_bucket(self, idx):
        img, label, _ = self.read_buffer(idx)
        imgW, imgH = img.size
        new_w, image_height = resize(imgW, imgH, self.image_height, self.image_min_width, self.image_max_width)

        return new_w

    def read_buffer(self, idx):
#         print(self.annotations[idx])
        if self.annotations[idx].__len__() == 2:
            img_path, label = self.annotations[idx]
            img_path = os.path.join(self.root_dir, img_path)
            img_b64 = self.s3.download_file(ConfigS3.S3_BUCKET_NAME, img_path)['b64']
            img = b64_2_img(img_b64)
        else:
            img_path, label, xmin, ymin, xmax, ymax, field, id = self.annotations[idx]
#             if field == "mrz" and len(label) != 44:
#                 print(id, label, len(label))
            img_b64 = self.s3.download_file(ConfigS3.S3_BUCKET_NAME, img_path)['b64']
            img = b64_2_img(img_b64)  
            img = self.get_crop_img(img, [xmin, ymin, xmax, ymax])
            

        return img, label, img_path #buf, label, img_path

    def read_data(self, idx):
        img, label, img_path = self.read_buffer(idx)
       
        if self.transform:
            img = self.transform(img)

        img_bw = process_image(img, self.image_height, self.image_min_width, self.image_max_width)
            
        word = self.vocab.encode(label)

        return img_bw, word, img_path

    def __getitem__(self, idx):
        img, word, img_path = self.read_data(idx)
        
        img_path = os.path.join(self.root_dir, img_path)
        
        sample = {'img': img, 'word': word, 'img_path': img_path}

        return sample

    def __len__(self):
        return self.nSamples

class ClusterRandomSampler(Sampler):
    
    def __init__(self, data_source, batch_size, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle     
    

    def flatten_list(self, lst):
        return [item for sublist in lst for item in sublist]

    def __iter__(self):
        batch_lists = []
        for cluster, cluster_indices in self.data_source.cluster_indices.items():
            if self.shuffle:
                random.shuffle(cluster_indices)

            batches = [cluster_indices[i:i + self.batch_size] for i in range(0, len(cluster_indices), self.batch_size)]
            batches = [_ for _ in batches if len(_) == self.batch_size]
            if self.shuffle:
                random.shuffle(batches)

            batch_lists.append(batches)

        lst = self.flatten_list(batch_lists)
        if self.shuffle:
            random.shuffle(lst)

        lst = self.flatten_list(lst)

        return iter(lst)

    def __len__(self):
        return len(self.data_source)

class Collator(object):
    def __init__(self, masked_language_model=True):
        self.masked_language_model = masked_language_model

    def __call__(self, batch):
        filenames = []
        img = []
        target_weights = []
        tgt_input = []
        max_label_len = max(len(sample['word']) for sample in batch)
        for sample in batch:
            img.append(sample['img'])
            filenames.append(sample['img_path'])
            label = sample['word']
            label_len = len(label)
            
            
            tgt = np.concatenate((
                label,
                np.zeros(max_label_len - label_len, dtype=np.int32)))
            tgt_input.append(tgt)

            one_mask_len = label_len - 1

            target_weights.append(np.concatenate((
                np.ones(one_mask_len, dtype=np.float32),
                np.zeros(max_label_len - one_mask_len,dtype=np.float32))))
            
        img = np.array(img, dtype=np.float32)


        tgt_input = np.array(tgt_input, dtype=np.int64).T
        tgt_output = np.roll(tgt_input, -1, 0).T
        tgt_output[:, -1]=0
        
        # random mask token
        if self.masked_language_model:
            mask = np.random.random(size=tgt_input.shape) < 0.05
            mask = mask & (tgt_input != 0) & (tgt_input != 1) & (tgt_input != 2)
            tgt_input[mask] = 3

        tgt_padding_mask = np.array(target_weights)==0

        rs = {
            'img': torch.FloatTensor(img),
            'tgt_input': torch.LongTensor(tgt_input),
            'tgt_output': torch.LongTensor(tgt_output),
            'tgt_padding_mask': torch.BoolTensor(tgt_padding_mask),
            'filenames': filenames
        }   
        
        return rs

    
    
if __name__ == "__main__":
    s3 = S3StorageService()
    json_file = s3.download_file(ConfigS3.S3_BUCKET_NAME, "dataset/OCR/labeling/cmqd_passport/labels/202")['b64']
    data = base64.b64decode(json_file)
    data = data.decode('utf8')
    data = json.loads(data)

#     with open("test.json", encoding="utf-8") as f:
#         data = json.load(f)
    
    for i in range(2, len(data['result']), 3):
        d = data['result'][i]
        field = data['result'][i-1]['value']['labels'][0]
        x = d['value']['x']
        y = d['value']['y']
        w = d['value']['width']
        h = d['value']['height']
        
        xmin = int(x)
        ymin = int(y)
        xmax = int(x+w) + 1
        ymax = int(y+h) + 1
        
        label = d['value']['text'][0]
        print(xmin,ymin,xmax,ymax,label, field)
    
    print(data["task"]["id"])
    print(data["task"]["data"]["ocr"].replace("s3://datalake/dataset/OCR/", ""))