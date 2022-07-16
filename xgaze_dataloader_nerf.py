import json
import os
import random
from typing import List
from xml.dom import INDEX_SIZE_ERR

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from random import randint

trans_train = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

trans = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_train_loader(
    data_dir, batch_size, num_workers=0, is_shuffle=True, subject=None,  evaluate = None, index_file = None
):
    """
    Create the dataloader for the train dataset, takes the subjects from train_test_split.json .
    :data_dir: Path to the subjects files with all the information
    :batch_size: Batch size that will be used by the dataloader
    :num_workers: Num of workers to utilize
    :is_shuffle: Boolean value that indicates if images will be shuffled
    """

    # load dataset
    refer_list_file = os.path.join("data/eth_xgaze", "train_test_split.json")
    print("load the train file list from: ", refer_list_file)

    with open(refer_list_file, "r") as f:
        datastore = json.load(f)

    # there are three subsets for ETH-XGaze dataset: train, test and test_person_specific
    # train set: the training set includes 80 participants data
    # test set: the test set for cross-dataset and within-dataset evaluations
    # test_person_specific: evaluation subset for the person specific setting
    sub_folder_use = "train"
    train_set = GazeDataset(
        dataset_path=os.path.join(data_dir, "subjects"),
        keys_to_use=datastore[sub_folder_use],
        sub_folder=sub_folder_use,
        transform=trans,
        is_shuffle=is_shuffle,
        subject=subject,
        evaluate= evaluate,
        index_file=index_file,
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)

    return train_loader


def get_val_loader_processed(
    data_dir, batch_size, num_val_images, num_workers=1, is_shuffle=True, subject=None,  evaluate = None, index_file = None
):
    """
    Create the dataloader for the validation dataset, takes the subjects from train_test_split.json .
    :data_dir: Path to the subjects files with all the information
    :batch_size: Batch size that will be used by the dataloader
    :num_workers: Num of workers to utilize
    :is_shuffle: Boolean value that indicates if images will be shuffled
    """

    # load dataset
    refer_list_file = os.path.join("data/eth_xgaze", "train_test_split.json")
    print("load the val file list from: ", refer_list_file)

    with open(refer_list_file, "r") as f:
        datastore = json.load(f)

    # there are three subsets for ETH-XGaze dataset: train, test and test_person_specific
    # train set: the training set includes 80 participants data
    # test set: the test set for cross-dataset and within-dataset evaluations
    # test_person_specific: evaluation subset for the person specific setting
    sub_folder_use = "val"
    val_set = GazeDataset(
        dataset_path=os.path.join(data_dir, "subjects"),
        keys_to_use=datastore[sub_folder_use],
        sub_folder=sub_folder_use,
        num_val_images=num_val_images,
        transform=trans,
        is_shuffle=is_shuffle,
        subject=subject,
        evaluate= evaluate,
        index_file=index_file,
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)

    return val_loader


def get_test_loader(data_dir, batch_size, num_workers=4, is_shuffle=True, subject=None,  evaluate = None, index_file = None):
    """
    Create the dataloader for the test dataset, takes the subjects from train_test_split.json .
    :data_dir: Path to the subjects files with all the information
    :batch_size: Batch size that will be used by the dataloader
    :num_workers: Num of workers to utilize
    :is_shuffle: Boolean value that indicates if images will be shuffled
    """

    # load dataset
    refer_list_file = os.path.join("data/eth_xgaze", "train_test_split.json")
    print("load the train file list from: ", refer_list_file)

    with open(refer_list_file, "r") as f:
        datastore = json.load(f)

    # there are three subsets for ETH-XGaze dataset: train, test and test_person_specific
    # train set: the training set includes 80 participants data
    # test set: the test set for cross-dataset and within-dataset evaluations
    # test_person_specific: evaluation subset for the person specific setting
    sub_folder_use = "test"
    test_set = GazeDataset(
        dataset_path=os.path.join(data_dir, "subjects"),
        keys_to_use=datastore[sub_folder_use],
        sub_folder=sub_folder_use,
        transform=trans,
        is_shuffle=is_shuffle,
        subject=subject,
        evaluate= evaluate,
        index_file=index_file,
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

    return test_loader


class GazeDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        keys_to_use: List[str] = None,
        sub_folder="",
        num_val_images=50,
        transform=None,
        is_shuffle=True,
        index_file=None,
        subject=None,
        evaluate = None,
    ):
        """
        Init function for the ETH-XGaze dataset, create key pairs to shuffle the dataset.
        :dataset_path: Path to the subjects files with all the information
        :keys_to_use: The subjects ID to use for the dataset
        :sub_folder: Indicate if it has to create the train,validation or test dataset
        :num_val_images: Used only for the validation dataset, indicate how many images to include in the validation dataset
        :transform: All the transformations to apply to the images
        :is_shuffle: Boolean value that indicates if images will be shuffled
        :index_file: Path to a specific key pairs file
        """

        self.path = dataset_path
        self.hdfs = {}
        self.sub_folder = sub_folder
        self.evaluate = evaluate

        # Select keys
        if subject is None:
            self.selected_keys = [k for k in keys_to_use]
        else:
            self.selected_keys = [subject]  # ["subject0000.h5"]
        assert len(self.selected_keys) > 0

        for num_i in range(0, len(self.selected_keys)):
            file_path = os.path.join(self.path, self.selected_keys[num_i])
            print(self.selected_keys[num_i])
            self.hdfs[num_i] = h5py.File(file_path, "r", swmr=True)
            # print('read file: ', os.path.join(self.path, self.selected_keys[num_i]))
            assert self.hdfs[num_i].swmr_mode

        # Construct mapping from full-data index to key and person-specific index
        if index_file is None:
            self.idx_to_kv = []
            for num_i in range(0, len(self.selected_keys)):
                if sub_folder == "val":
                    n = num_val_images
                else:
                    n = self.hdfs[num_i]["face_patch"].shape[0]
                self.idx_to_kv += [(num_i, i) for i in range(n)]
        else:
            print("load the file: ", "configs/config_files/evaluation_input.txt")
            self.idx_to_kv = np.loadtxt("configs/config_files/evaluation_input.txt", dtype=np.int)
            print("load the file: ", "configs/config_files/evaluation_target.txt")
            self.target_idx = np.loadtxt("configs/config_files/evaluation_target.txt", dtype=np.int)

        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

        if is_shuffle and index_file is None:
            random.shuffle(self.idx_to_kv)  # random the order to stable the training
        #np.savetxt(fname="evaluation_input.txt",X=self.idx_to_kv,fmt ='%.0f')
    
        """
        keys = []
        for i in range(len(self.idx_to_kv)):
            key, idx = self.idx_to_kv[i]
            key_target = key +1
            while key_target != key:
                idx_target = randint(a = 0, b = len(self.idx_to_kv) -1)
                #print(idx_target)
                key_target,idx = self.idx_to_kv[idx_target]
            keys.append(idx_target)
        np.savetxt(fname="evaluation_target.txt",X=keys,fmt ='%.0f')
        """ 

        self.hdf = None
        self.transform = transform

    def __len__(self):
        """
        Function that returns the length of the dataset.
        :return: Returns the length of the dataset
        """

        return len(self.idx_to_kv)

    def __del__(self):
        """
        Close all the hdfs files of the subjects.
        """

        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

    def __getitem__(self, idx_input):
        """
        Return one sample from the dataset, the on in poistiongiven by idx.
        :idx: Indicate the position of the data sample to return
        :return: Returns one sample from the dataset
        """
        key, idx = self.idx_to_kv[idx_input]

        idx_input_image = idx
        key_input_image = key

        self.hdf = h5py.File(
            os.path.join(self.path, self.selected_keys[key]),
            "r",
            swmr=True,
        )
        assert self.hdf.swmr_mode

        # Get face image
        image = self.hdf["face_patch"][idx, :]
        # image = image[:, :, [2, 1, 0]]  # from BGR to RGB
        image = self.transform(image)

        face_mask = self.hdf["head_mask"][idx, :]
        eye_mask = self.hdf["eye_mask"][idx, :]

        kernel_2 = np.ones((3, 3), dtype=np.uint8)
        face_mask = cv2.erode(face_mask, kernel_2, iterations=2)

        ldms = self.hdf["facial_landmarks"][idx, :]
        cam_ind = self.hdf["cam_index"][idx, :]

        nl3dmm_para_dict = {}

        # nl3dmm_para_dict["code"] = self.hdf["latent_codes"][idx, :]
        nl3dmm_para_dict["code"] = self.hdf["latent_codes"][0, :]
        nl3dmm_para_dict["w2c_Rmat"] = self.hdf["w2c_Rmat"][idx, :]
        nl3dmm_para_dict["w2c_Tvec"] = self.hdf["w2c_Tvec"][idx, :]
        nl3dmm_para_dict["inmat"] = self.hdf["inmat"][idx, :]
        nl3dmm_para_dict["c2w_Rmat"] = self.hdf["c2w_Rmat"][idx, :]
        nl3dmm_para_dict["c2w_Tvec"] = self.hdf["c2w_Tvec"][idx, :]
        nl3dmm_para_dict["inv_inmat"] = self.hdf["inv_inmat"][idx, :]
        nl3dmm_para_dict["pitchyaw"] = self.hdf["pitchyaw"][idx, :]

        if self.evaluate == "target":
            #print(idx_input)
            key_target,idx = self.idx_to_kv[self.target_idx[idx_input]]
            #print(key_target,idx)

            idx_target_image = idx
            key_target_image = key_target

            self.hdf = h5py.File(
            os.path.join(self.path, self.selected_keys[key_target]),
            "r",
            swmr=True,
            )
            assert self.hdf.swmr_mode

            # Get face image
            image_target = self.hdf["face_patch"][idx, :]
            # image = image[:, :, [2, 1, 0]]  # from BGR to RGB
            image_target = self.transform(image_target)

            face_mask_target = self.hdf["head_mask"][idx, :]
            eye_mask_target = self.hdf["eye_mask"][idx, :]

            kernel_2_target = np.ones((3, 3), dtype=np.uint8)
            face_mask_target = cv2.erode(face_mask_target, kernel_2_target, iterations=2)

            ldms_target = self.hdf["facial_landmarks"][idx, :]
            cam_ind_target = self.hdf["cam_index"][idx, :]

            nl3dmm_para_dict_target = {}

            # nl3dmm_para_dict["code"] = self.hdf["latent_codes"][idx, :]
            nl3dmm_para_dict_target["code"] = self.hdf["latent_codes"][0, :]
            nl3dmm_para_dict_target["w2c_Rmat"] = self.hdf["w2c_Rmat"][idx, :]
            nl3dmm_para_dict_target["w2c_Tvec"] = self.hdf["w2c_Tvec"][idx, :]
            nl3dmm_para_dict_target["inmat"] = self.hdf["inmat"][idx, :]
            nl3dmm_para_dict_target["c2w_Rmat"] = self.hdf["c2w_Rmat"][idx, :]
            nl3dmm_para_dict_target["c2w_Tvec"] = self.hdf["c2w_Tvec"][idx, :]
            nl3dmm_para_dict_target["inv_inmat"] = self.hdf["inv_inmat"][idx, :]
            nl3dmm_para_dict_target["pitchyaw"] = self.hdf["pitchyaw"][idx, :]
                
            return image, face_mask, eye_mask, nl3dmm_para_dict,ldms, cam_ind,idx_input_image, key_input_image, image_target, face_mask_target, eye_mask_target, nl3dmm_para_dict_target,ldms_target, cam_ind_target, idx_target_image, key_target_image
        elif self.evaluate == "landmark":
            return image, face_mask, eye_mask, nl3dmm_para_dict,ldms, cam_ind,
        else:
            return image, face_mask, eye_mask, nl3dmm_para_dict