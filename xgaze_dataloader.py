from itertools import count
import numpy as np
import h5py
from requests import head
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import json
import random
import cv2
from typing import List

trans_train = transforms.Compose([
        #transforms.ToPILImage(),
        #transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                     std=[0.229, 0.224, 0.225]),
        transforms.Resize(size=(128,128)),
    ])

trans = transforms.Compose([
        #transforms.ToPILImage(),
        #transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                     std=[0.229, 0.224, 0.225]),
        transforms.Resize(size=(128,128)),
    ])


def get_train_loader(data_dir,
                           batch_size,
                           num_workers=0,
                           is_shuffle=True,
                           subject = None):
    # load dataset
    refer_list_file = 'data/eth_xgaze/train_test_split.json'
    print('load the train file list from: ', refer_list_file)

    with open(refer_list_file, 'r') as f:
        datastore = json.load(f)

    # there are three subsets for ETH-XGaze dataset: train, test and test_person_specific
    # train set: the training set includes 80 participants data
    # test set: the test set for cross-dataset and within-dataset evaluations
    # test_person_specific: evaluation subset for the person specific setting
    sub_folder_use = 'train'
    train_set = GazeDataset(dataset_path=data_dir, keys_to_use=datastore[sub_folder_use], sub_folder=sub_folder_use,
                            transform=trans, is_shuffle=is_shuffle, is_load_label=True, subject=subject)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers,drop_last=True)

    return train_set,train_loader

def get_val_loader(data_dir,
                           batch_size,
                           num_val_images,
                           num_workers=0,
                           is_shuffle=True,
                           subject = None):
    # load dataset
    refer_list_file = 'data/eth_xgaze/train_test_split.json'
    print('load the val file list from: ', refer_list_file)

    with open(refer_list_file, 'r') as f:
        datastore = json.load(f)

    # there are three subsets for ETH-XGaze dataset: train, test and test_person_specific
    # train set: the training set includes 80 participants data
    # test set: the test set for cross-dataset and within-dataset evaluations
    # test_person_specific: evaluation subset for the person specific setting
    sub_folder_use = 'val'
    val_set = GazeDataset(dataset_path=data_dir, keys_to_use=datastore[sub_folder_use], sub_folder=sub_folder_use,
                            transform=trans, is_shuffle=is_shuffle, is_load_label=True,subject=subject, num_val_images = num_val_images)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers,drop_last=True)

    return val_loader


def get_test_loader(data_dir,
                           batch_size,
                           num_workers=4,
                           is_shuffle=True):
    # load dataset
    refer_list_file = os.path.join(data_dir, 'train_test_split.json')
    print('load the train file list from: ', refer_list_file)

    with open(refer_list_file, 'r') as f:
        datastore = json.load(f)

    # there are three subsets for ETH-XGaze dataset: train, test and test_person_specific
    # train set: the training set includes 80 participants data
    # test set: the test set for cross-dataset and within-dataset evaluations
    # test_person_specific: evaluation subset for the person specific setting
    sub_folder_use = 'test'
    test_set = GazeDataset(dataset_path=data_dir, keys_to_use=datastore[sub_folder_use], sub_folder=sub_folder_use,
                           transform=trans, is_shuffle=is_shuffle, is_load_label=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers,drop_last=True)

    return test_loader


class GazeDataset(Dataset):
    def __init__(self, dataset_path: str, keys_to_use: List[str] = None, sub_folder='', transform=None, is_shuffle=True,
                 index_file=None, is_load_label=True, get_second_sample = True, subject = None, num_val_images=100):
        self.path = dataset_path
        self.hdfs = {}
        self.sub_folder = sub_folder
        self.is_load_label = is_load_label
        self.get_second_sample = get_second_sample
        self.is_bgr = True

        # assert len(set(keys_to_use) - set(all_keys)) == 0
        # Select keys
        # TODO: select only people with sufficient entries?
        if subject is None:
            self.selected_keys = [k for k in keys_to_use]
        else:
            self.selected_keys = [subject]
        self.prefixes = keys_to_use
        assert len(self.selected_keys) > 0

        for num_i in range(0, len(self.selected_keys)):
            file_path = os.path.join(self.path,"xgaze_" + self.selected_keys[num_i])
            self.hdfs[num_i] = h5py.File(file_path, 'r', swmr=True)
            # print('read file: ', os.path.join(self.path, self.selected_keys[num_i]))
            assert self.hdfs[num_i].swmr_mode

        # Construct mapping from full-data index to key and person-specific index
        if index_file is None:
            self.idx_to_kv = []
            for num_i in range(0, len(self.selected_keys)):
                if sub_folder == "val":
                    n = num_val_images
                else:
                    n= 50*18
                    #n = self.hdfs[num_i]["face_patch"].shape[0]
                
                #self.idx_to_kv += [
                #    (num_i, i) for i in range(n)
                #]  
                self.idx_to_kv += [
                    (num_i, i) for i in range(43*18)
                ]  
                self.idx_to_kv += [
                    (num_i, i) for i in range(self.hdfs[num_i]["face_patch"].shape[0]-1, self.hdfs[num_i]["face_patch"].shape[0] - 1 - 7*18, -1)
                ] 
                
        else:
            print('load the file: ', index_file)
            self.idx_to_kv = np.loadtxt(index_file, dtype=np.int)

        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

        if is_shuffle:
            random.shuffle(self.idx_to_kv)  # random the order to stable the training

        self.hdf = None
        self.transform = transform
        self.n = n
        self.target_idx = np.loadtxt("evaluation_target_single_subject.txt", dtype=np.int)

    def __len__(self):
        return len(self.idx_to_kv)

    def __del__(self):
        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

    def preprocess_image(self, image):
        if self.is_bgr:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        else:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        #ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        image = np.transpose(image, [2, 0, 1])  # Colour image
        image = 2.0 * image / 255.0 - 1
        return image

    def preprocess_entry(self, val):
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val.astype(np.float32))
        elif isinstance(val, int):
            # NOTE: maybe ints should be signed and 32-bits sometimes
            val = torch.tensor(val, dtype=torch.long, requires_grad=False)
        return val

    def __getitem__(self, idx):
        key, idx = self.idx_to_kv[idx]

        #self.hdf = h5py.File(os.path.join(self.path,"xgaze_" + self.selected_keys[key]), 'r', swmr=True)

        self.hdf_nerf = h5py.File(os.path.join(self.path,"xgaze_" + self.selected_keys[key]), 'r', swmr=True) #TODO check the path
        assert self.hdf_nerf.swmr_mode

        # Get face image
        image = self.hdf_nerf['face_patch'][idx, :]
        #image = image[:, :, [2, 1, 0]]  # from BGR to RGB
        image = self.preprocess_image(image)
        image = self.preprocess_entry(image)
        image = self.transform(image)

        # Get labels
        if self.is_load_label:
            gaze_label = self.hdf_nerf["pitchyaw_head"][idx, :]
            #gaze_label = self.hdf["face_gaze"][idx, :]
            gaze_label = gaze_label.astype(np.float32)
            head_label = self.hdf_nerf['face_head_pose'][idx, :]
            head_label = head_label.astype(np.float32)
            entry = {
            'key': key,
            'image_a': image,
            'gaze_a': gaze_label,
            'head_a': head_label,
        }
            if self.get_second_sample:
                all_indices = [i for i in range(self.n) if i != idx]
                if len(all_indices) == 1:
                    # If there is only one sample for this person, just return the same sample.
                    idx_b = idx
                elif self.sub_folder != 'val':
                    idx_b = np.random.choice(all_indices)
                elif self.sub_folder == 'val':
                    idx_b = self.target_idx[idx]

                
                # Grab 2nd entry from same person
                # Get face image
                image = self.hdf_nerf['face_patch'][idx_b, :]
                #image = image[:, :, [2, 1, 0]]  # from BGR to RGB
                image = self.preprocess_image(image)
                image = self.preprocess_entry(image)
                image = self.transform(image)

                gaze_label = self.hdf_nerf["pitchyaw_head"][idx_b, :]
                #gaze_label = self.hdf["face_gaze"][idx_b, :]
                gaze_label = gaze_label.astype(np.float32)
                head_label = self.hdf_nerf['face_head_pose'][idx_b, :]
                head_label = head_label.astype(np.float32)

                face_mask = self.hdf_nerf["head_mask"][idx_b, :]
                kernel_2 = np.ones((3, 3), dtype=np.uint8)
                face_mask = cv2.erode(face_mask, kernel_2, iterations=2)

                left_eye_mask = self.hdf_nerf["left_eye_mask"][idx_b, :]
                right_eye_mask = self.hdf_nerf["right_eye_mask"][idx_b, :]

                entry['image_b'] = image
                entry['gaze_b'] = gaze_label
                entry['head_b'] = head_label
                entry['mask_b'] = face_mask
                entry['left_eye_b'] = left_eye_mask
                entry['right_eye_b'] = right_eye_mask
                entry['cam_ind_b'] = self.hdf_nerf["cam_index"][idx_b, :]
                entry['ldms_b'] = self.hdf_nerf["facial_landmarks"][idx_b, :]


            return entry
        else:
            return 