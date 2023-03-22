# --------------------------------------------------------
# Copyright (C) 2019 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Seonwook Park, Shalini De Mello, Yufeng Zheng.
# --------------------------------------------------------
from dataclasses import is_dataclass
import numpy as np
from collections import OrderedDict
import gc
import json
import os
import cv2
import losses
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader, Subset
import logging
from dataset import HDFDataset
from models.gazeheadResnet import GazeHeadResNet
from models.gazeheadnet import GazeHeadNet
from utils import save_images, worker_init_fn, send_data_dict_to_gpu, recover_images, def_test_list, RunningStatistics,\
    adjust_learning_rate, script_init_common, get_example_images, save_model, load_model
from core import DefaultConfig
from models.xgaze_baseline_head import gaze_network_head
from models import STED
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
import wandb
from PIL import Image
from torchvision import transforms
from xgaze_dataloader import get_train_loader
from xgaze_dataloader import get_val_loader as xgaze_get_val_loader
from mpii_face_dataloader import get_val_loader as mpii_get_val_loader
from columbia_dataloader import get_val_loader as columbia_get_val_loader
from gaze_capture_dataloader import get_val_loader as gaze_capture_get_val_loader
from standard_image_dataset import get_data_loader as image_get_data_loader
from piq import ssim, psnr, LPIPS,DISTS, FID
import torch.nn.functional as F
from gaze_estimation_utils import normalize
import scipy.io
from logging_utils import log_evaluation_image, log_one_subject_evaluation_results, log_all_datasets_evaluation_results
from face_recognition.evaluation_similarity import evaluation_similarity
import h5py

trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.Resize(size=(224,224)),
    ])

trans_normalize = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  
    ])

trans_resize = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(size=(224, 224)),
    ]
)

torch.manual_seed(45)  # cpu
torch.cuda.manual_seed(55)  # gpu
np.random.seed(65)  # numpy
random.seed(75)  # random and transforms
torch.backends.cudnn.deterministic = True  # cudnn
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
# Set Configurations
config = DefaultConfig()
wandb.init(project="sted evalaution", config={"gpu_id": 0})
script_init_common()
# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import warnings
warnings.filterwarnings('ignore')

if not config.skip_training:
    if config.semi_supervised:
        assert config.num_labeled_samples != 0
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    # save configurations
    config.write_file_contents(config.save_path)

# Create the train and test datasets.
all_data = OrderedDict()
# Read GazeCapture train/val/test split
with open('./gazecapture_split.json', 'r') as f:
    all_gc_prefixes = json.load(f)
if not config.skip_training:
    # Define single training dataset
    train_prefixes = all_gc_prefixes['train']
    #train_dataset = HDFDataset(hdf_file_path=config.xgaze_file,
                               #prefixes=train_prefixes,
                               #is_bgr=False,
                               #get_2nd_sample=True,
                               #num_labeled_samples=config.num_labeled_samples if config.semi_supervised else None)
    
    # Define multiple val/test datasets for evaluation during training
    for tag, hdf_file, is_bgr, prefixes in [
        #('gc/val', config.gazecapture_file, False, all_gc_prefixes['val']),
        #('gc/test', config.gazecapture_file, False, all_gc_prefixes['test']),
        #('mpi', config.mpiigaze_file, False, None),
        #('xgaze_val', config.xgaze_val_file, False, None),
        #('columbia', config.columbia_file, True, None),
        #('eyediap', config.eyediap_file, True, None),
    ]:
        #dataset = HDFDataset(hdf_file_path=hdf_file,
        #                     prefixes=prefixes,
        #                     is_bgr=is_bgr,
        #                     get_2nd_sample=True,
        #                     pick_at_least_per_person=2)
        dataset,dataloader = xgaze_get_val_loader(data_dir = "/data/data2/aruzzi/train",batch_size=int(config.batch_size))
        if tag == 'gc/test':
            # test pair visualization:
            test_list = def_test_list()
            test_visualize = get_example_images(dataset, test_list)
            test_visualize = send_data_dict_to_gpu(test_visualize, device)

        subsample = config.test_subsample
        # subsample test sets if requested
        if subsample < (1.0 - 1e-6):
            dataset = Subset(dataset, np.linspace(
                start=0, stop=len(dataset),
                num=int(subsample * len(dataset)),
                endpoint=False,
                dtype=np.uint32,
            ))
        all_data[tag] = {
            'dataset': dataset,
            'dataloader': DataLoader(dataset,
                                     batch_size=config.eval_batch_size,
                                     shuffle=False,
                                     num_workers=config.num_data_loaders,  # args.num_data_loaders,
                                     pin_memory=True,
                                     ),
        }
    """
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=int(config.batch_size),
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=config.num_data_loaders,
                                  pin_memory=True,
                                  )
    """
    train_dataset, train_dataloader = get_train_loader(data_dir = "/data/data2/aruzzi/xgaze_subjects",batch_size=int(config.batch_size))
    all_data['gc/train'] = {'dataset': train_dataset, 'dataloader': train_dataloader}

    # Print some stats.
    logging.info('')
    for tag, val in all_data.items():
        tag = '[%s]' % tag
        dataset = val['dataset']
        original_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
        num_people = len(original_dataset.prefixes)
        num_original_entries = len(original_dataset)
        logging.info('%10s full set size:           %7d' % (tag, num_original_entries))
        logging.info('%10s current set size:        %7d' % (tag, len(dataset)))
        logging.info('')

    # Have dataloader re-open HDF to avoid multi-processing related errors.
    #for tag, data_dict in all_data.items():
    #    dataset = data_dict['dataset']
    #    original_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    #    original_dataset.close_hdf()

# Create redirection network
network = STED().to(device)
# Load weights if available
from checkpoints_manager import CheckpointsManager
print('---->> ',config.eval_gazenet_savepath)
saver = CheckpointsManager(network.GazeHeadNet_eval, config.eval_gazenet_savepath,device)
_ = saver.load_last_checkpoint()
del saver

saver = CheckpointsManager(network.GazeHeadNet_train, config.gazenet_savepath,device)
_ = saver.load_last_checkpoint(xgaze=True)
del saver

if config.load_step != 0:
    #load_model(network, os.path.join(config.save_path, "checkpoints", str(config.load_step) + '.pt'),device)
    load_model(network, os.path.join(config.save_path, "checkpoints", str(config.load_step) + '_final_green_10.pt'),device)
    logging.info("Loaded checkpoints from step " + str(config.load_step))

# Transfer on the GPU before constructing and optimizer
if torch.cuda.device_count() > 1:
    logging.info('Using %d GPUs!' % torch.cuda.device_count())
    network.encoder = nn.DataParallel(network.encoder)
    network.decoder = nn.DataParallel(network.decoder)
    network.discriminator = nn.DataParallel(network.discriminator)
    network.GazeHeadNet_eval = nn.DataParallel(network.GazeHeadNet_eval)
    network.GazeHeadNet_train = nn.DataParallel(network.GazeHeadNet_train)
    network.lpips = nn.DataParallel(network.lpips)


def execute_training_step(current_step):
    global train_data_iterator
    try:
        input = next(train_data_iterator)
    except StopIteration:
        np.random.seed()  # Ensure randomness
        # Some cleanup
        train_data_iterator = None
        torch.cuda.empty_cache()
        gc.collect()
        # Restart!
        global train_dataloader
        train_data_iterator = iter(train_dataloader)
        input = next(train_data_iterator)
    input = send_data_dict_to_gpu(input, device)
    network.train()
    # forward + backward + optimize
    loss_dict, generated = network.optimize(input, current_step)

    if current_step % 500 == 0:
        #print(input['image_a'].shape)
        #print(input['image_b'].shape)
        #print(generated.shape)
        img = np.concatenate([((input['image_a'].detach().cpu().permute(0, 2, 3, 1).numpy() +1) * 255.0/2.0).astype(np.uint8),((input['image_b'].detach().cpu().permute(0, 2, 3, 1).numpy() +1) * 255.0/2.0).astype(np.uint8),np.clip(((generated.detach().cpu().permute(0, 2, 3, 1).numpy()  +1) * 255.0/2.0),0,255).astype(np.uint8)],axis=2)
        #img = np.concatenate([(input['image_a'].detach().cpu().permute(0, 2, 3, 1).numpy()* 255.0).astype(np.uint8),(input['image_b'].detach().cpu().permute(0, 2, 3, 1).numpy() * 255.0).astype(np.uint8),(generated.detach().cpu().permute(0, 2, 3, 1).numpy() * 255.0).astype(np.uint8)],axis=2)
        img = Image.fromarray(img[0])
        log_image = wandb.Image(img)
        #log_image.show()
        wandb.log({"Sted Prediction": log_image})

    # If doing multi-GPU training, just take an average
    for key, value in loss_dict.items():
        if value.dim() > 0:
            value = torch.mean(value)
            loss_dict[key] = value
    # Store values for logging later
    for key, value in loss_dict.items():
        loss_dict[key] = value.detach().cpu()
    for key, value in loss_dict.items():
        running_losses.add(key, value.numpy())



def variance_of_laplacian(image):
    return cv2.Laplacian(image,cv2.CV_64F).var()

def select_dataloader(name, subject, idx, img_dir, batch_size, num_images, num_workers, is_shuffle):
    if name == "eth_xgaze":
        return (name, subject, idx, xgaze_get_val_loader(data_dir=img_dir, batch_size=batch_size, num_val_images= num_images, num_workers= num_workers, is_shuffle= is_shuffle, subject=subject))
    elif name == "mpii_face_gaze":
        return (name, subject, idx, mpii_get_val_loader(data_dir=img_dir, batch_size=batch_size, num_val_images= num_images, num_workers= num_workers, is_shuffle= is_shuffle, subject=subject))
    elif name == "columbia":
        return (name, subject, idx, columbia_get_val_loader(data_dir=img_dir, batch_size=batch_size, num_val_images= num_images, num_workers= num_workers, is_shuffle= is_shuffle, subject=subject))
    elif name == "gaze_capture":
        return (name, subject, idx, gaze_capture_get_val_loader(data_dir=img_dir, batch_size=batch_size, num_val_images= num_images, num_workers= num_workers, is_shuffle= is_shuffle, subject=subject))
    else:
        print("Dataset not supported")

def select_cam_matrix(name,cam_matrix,cam_distortion,cam_ind, subject):
    if name == "eth_xgaze":
        return cam_matrix[name][cam_ind], cam_distortion[name][cam_ind]
    elif name == "mpii_face_gaze":
        camera_matrix = cam_matrix[name][int(subject[-5:-3])]
        camera_matrix[0, 2] = 256.0
        camera_matrix[1, 2] = 256.0
        return camera_matrix, cam_distortion[name][int(subject[-5:-3])]
    elif name == "columbia":
        return cam_matrix[name], cam_distortion[name]
    elif name == "gaze_capture":
        return cam_matrix[name], cam_distortion[name]
    else:
        print("Dataset not supported")

def load_cams():
    cam_matrix = {}
    cam_distortion = {}
    cam_translation = {}
    cam_rotation = {}

    for name in config.data_names:
        cam_matrix[name] = []
        cam_distortion[name] = []
        cam_translation[name] = []
        cam_rotation[name] = []
    
    
    for cam_id in range(18):
        cam_file_name = "data/eth_xgaze/cam/cam" + str(cam_id).zfill(2) + ".xml"
        fs = cv2.FileStorage(cam_file_name, cv2.FILE_STORAGE_READ)
        cam_matrix["eth_xgaze"].append(fs.getNode("Camera_Matrix").mat())
        cam_distortion["eth_xgaze"].append(fs.getNode("Distortion_Coefficients").mat())
        cam_translation["eth_xgaze"].append(fs.getNode("cam_translation"))
        cam_rotation["eth_xgaze"].append(fs.getNode("cam_rotation"))
        fs.release()

    for i in range(15):
        file_name = os.path.join(
        "data/mpii_face_gaze/cam", "Camera" + str(i).zfill(2) + ".mat"
        )
        mat = scipy.io.loadmat(file_name)
        cam_matrix["mpii_face_gaze"].append(mat.get("cameraMatrix"))
        cam_distortion["mpii_face_gaze"].append(mat.get(
            "distCoeffs"
        ))

    cam_file_name = "data/columbia/cam/cam" + str(0).zfill(2) + ".xml"
    fs = cv2.FileStorage(cam_file_name, cv2.FILE_STORAGE_READ)
    cam_matrix["columbia"] = fs.getNode("Camera_Matrix").mat()
    cam_distortion["columbia"] = fs.getNode("Distortion_Coefficients").mat()

    cam_file_name = "data/gaze_capture/cam/cam" + str(0).zfill(2) + ".xml"
    fs = cv2.FileStorage(cam_file_name, cv2.FILE_STORAGE_READ)
    cam_matrix["gaze_capture"] = fs.getNode("Camera_Matrix").mat()
    cam_distortion["gaze_capture"] = fs.getNode("Distortion_Coefficients").mat()

    return cam_matrix,cam_distortion, cam_translation, cam_rotation

def calculate_FID(gt_images, pred_images):
    first_dl, second_dl = image_get_data_loader(gt_images), image_get_data_loader(pred_images)
    fid_metric = FID()
    first_feats = fid_metric.compute_feats(first_dl)
    second_feats = fid_metric.compute_feats(second_dl)
    fid: torch.Tensor = fid_metric(first_feats, second_feats)
    return fid

def execute_test(log, current_step):

    face_model_load =  np.loadtxt('data/eth_xgaze/face_model.txt')  # Generic face model with 3D facial landmarks
    val_keys = {}
    for name in config.data_names:
        file_path = os.path.join("data", name, "train_test_split.json")
        with open(file_path, "r") as f:
            datastore = json.load(f)
        val_keys[name] = datastore["val"]

    dataloader_all = []

    for idx,name in enumerate(config.data_names):
        for subject in val_keys[name]:
            dataloader_all.append(select_dataloader(name, subject, idx, config.img_dir[idx], 1, config.num_images[idx], 0, is_shuffle=False))   

    cam_matrix, cam_distortion, cam_translation, cam_rotation = load_cams()


    path = "sted/checkpoints/epoch_16_resnet_correct_ckpt.pth.tar"
    #path = "sted/checkpoints/epoch_60_512_ckpt.pth.tar"
    #path = "baseline_estimator_resnet/checkpoints/at_step_0140000.pth.tar"
    model = gaze_network_head().to(device)
    #model = GazeHeadResNet().to(device)
    #odel = GazeHeadNet().to(device)

    state_dict = torch.load(path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict=state_dict["model_state"])
    #model.load_state_dict(state_dict=state_dict)
    model.eval()

    print("Done")

    dict_angular_loss = {}
    dict_angular_head_loss = {}
    dict_ssim_loss = {}
    dict_psnr_loss = {}
    dict_lpips_loss = {}
    dict_l1_loss = {}
    dict_num_images = {}

    dict_similarity = {}

    dict_fid = {}
    dict_gt_images = {}
    dict_pred_images = {}
    full_images_gt_list = []
    full_images_pred_list = []

    for name in config.data_names:
        dict_angular_loss[name] = 0.0
        dict_angular_head_loss[name] = 0.0
        dict_ssim_loss[name] = 0.0
        dict_psnr_loss[name] = 0.0
        dict_lpips_loss[name] = 0.0
        dict_l1_loss[name] = 0.0
        dict_num_images[name] = 0

        dict_similarity[name] = 0.0

        dict_fid[name] = 0.0
        dict_gt_images[name] = []
        dict_pred_images[name] = []
    
    for name, subject, index_dataset, dataloader in dataloader_all:
    
        angular_loss = 0.0
        angular_head_loss = 0.0
        ssim_loss = 0.0
        psnr_loss = 0.0
        lpips_loss = 0.0
        l1_loss = 0.0
        num_images = 0

        similarity = 0.0

        fid = 0.0
        gt_list = []
        pred_list = []

        for index,entry in enumerate(dataloader):
            print(index)
            ldms = entry["ldms_b"][0]
            batch_head_mask = torch.reshape(entry["mask_b"], (1, 1, 512, 512))
            cam_ind = entry["cam_ind_b"]

            camera_matrix, camera_distortion = select_cam_matrix(name, cam_matrix,cam_distortion, cam_ind, subject)

            input_dict = send_data_dict_to_gpu(entry, device)
            output_dict, loss_dict = network(input_dict)

            image_gt = ((input_dict['image_b'].detach().cpu().permute(0, 2, 3, 1).numpy() +1) * 255.0/2.0).astype(np.uint8)
            image_gen = np.clip(((output_dict['image_b_hat'].detach().cpu().permute(0, 2, 3, 1).numpy() +1) * 255.0/2.0),0,255).astype(np.uint8)

            batch_images_gt = trans_normalize(image_gt[0,:])
            
            nonhead_mask = batch_head_mask < 0.5   
            nonhead_mask_c3b = nonhead_mask.expand(-1, 3, -1, -1)  
            batch_images_gt = torch.reshape(batch_images_gt,(1,3,512,512))      
            batch_images_gt[nonhead_mask_c3b] = 1.0

            target_image_quality = torch.reshape(
                batch_images_gt , (1, 3, 512, 512)
            ).to(device)
            #target_image_quality = batch_images_gt
            

            batch_images_gt_norm = normalize(
                (batch_images_gt.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(
                    np.uint8
                )[0],
                camera_matrix,
                camera_distortion,
                face_model_load,
                ldms,
                config.img_dim,
            )
            target_normalized_log = torch.reshape(trans_normalize(batch_images_gt_norm),(1,3,224,224)).to(device)
            batch_images_gt_norm = torch.reshape(
                trans(batch_images_gt_norm), (1, 3, config.img_dim, config.img_dim)
            ).to(
                device
            )  
            pitchyaw_gt, head_gt = model(batch_images_gt_norm)



            batch_images_gen = trans_normalize(image_gen[0,:])
            
            nonhead_mask = batch_head_mask < 0.5
            nonhead_mask_c3b = nonhead_mask.expand(-1, 3, -1, -1)
            batch_images_gen = torch.reshape(batch_images_gen,(1,3,512,512))
            batch_images_gen[nonhead_mask_c3b] = 1.0

            pred_image_quality = torch.reshape(
                 batch_images_gen, (1, 3, 512, 512)
            ).to(device)
            #pred_image_quality = batch_images_gen

            batch_images_gen_norm = normalize(
                (batch_images_gen.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(
                    np.uint8
                )[0],
                camera_matrix,
                camera_distortion,
                face_model_load,
                ldms,
                config.img_dim,
            )

            pred_normalized_log = torch.reshape(trans_normalize(batch_images_gen_norm),(1,3,224,224)).to(device)
            
            batch_images_norm = torch.reshape(
                trans(batch_images_gen_norm), (1, 3, config.img_dim, config.img_dim)
            ).to(device)
            pitchyaw_gen, head_gen = model(batch_images_norm)


            loss = losses.gaze_angular_loss(pitchyaw_gt,pitchyaw_gen).detach().cpu().numpy()
            angular_loss += loss
            num_images += 1
            dict_angular_loss[name] += loss
            dict_num_images[name] += 1
            print("Gaze Angular Error: ",angular_loss/num_images,loss,num_images)

            loss = losses.gaze_angular_loss(head_gt,head_gen).detach().cpu().numpy()
            angular_head_loss += loss
            dict_angular_head_loss[name] += loss
            print("Head Angular Error: ",angular_head_loss/num_images,loss,num_images)

            sim_gt = ( torch.reshape(
                trans_resize(batch_images_gt[0,:]) , (1, 3, config.img_dim, config.img_dim)
            ).to(device).detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)[0]

            sim_gen = ( torch.reshape(
                trans_resize(batch_images_gen[0,:]) , (1, 3, config.img_dim, config.img_dim)
            ).to(device).detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)[0]
            
            try:
                loss = evaluation_similarity(sim_gt, sim_gen)
            except:
                loss = -0.1
            similarity += loss
            dict_similarity[name] += loss
            print("Similarity Score: ", similarity / num_images, loss, num_images)

            gt_list.append(target_image_quality[0,:])
            pred_list.append(pred_image_quality[0,:])

            #dict_gt_images[name].append(target_image_quality[0,:])
            #dict_pred_images[name].append(pred_image_quality[0,:])

            #full_images_gt_list.append(target_image_quality[0,:])
            #full_images_pred_list.append(pred_image_quality[0,:])

            loss = ssim(target_image_quality, pred_image_quality, data_range=1.).detach().cpu().numpy()
            ssim_loss += loss
            dict_ssim_loss[name] += loss
            print("SSIM: ",ssim_loss/num_images,loss,num_images)

            loss = psnr(target_image_quality, pred_image_quality, data_range=1.).detach().cpu().numpy()
            psnr_loss += loss
            dict_psnr_loss[name] += loss
            print("PSNR: ",psnr_loss/num_images,loss,num_images)

            lpips_metric = LPIPS()
            loss = lpips_metric(target_image_quality, pred_image_quality).detach().cpu().numpy()
            lpips_loss += loss
            dict_lpips_loss[name] += loss
            print("LPIPS: ",lpips_loss/num_images,loss,num_images)

            loss = torch.nn.functional.l1_loss(target_image_quality, pred_image_quality).detach().cpu().numpy()
            l1_loss += loss
            dict_l1_loss[name] += loss
            print("L1 Distance: ", l1_loss/num_images,loss, num_images)

            if index % log == 0:
                log_evaluation_image(pred_normalized_log, target_normalized_log, ((input_dict['image_a'].detach().cpu().permute(0, 2, 3, 1).numpy() +1) * 255.0/2.0).astype(np.uint8), target_image_quality, pred_image_quality)

        fid = calculate_FID(gt_images= gt_list, pred_images= pred_list)

        if index % log == 0:
            log_one_subject_evaluation_results(current_step, angular_loss, angular_head_loss, ssim_loss, psnr_loss, lpips_loss,
                                                l1_loss, num_images, fid , similarity)

    for name in config.data_names:
         dict_fid[name]  = calculate_FID(gt_images= dict_gt_images[name], pred_images= dict_pred_images[name])
         #dict_fid[name]  = 0.0
        
    full_fid = calculate_FID(gt_images= full_images_gt_list, pred_images= full_images_pred_list)    
    #full_fid = 0.0
                            
    if index % log == 0:
        log_all_datasets_evaluation_results(current_step, config.data_names, dict_angular_loss, dict_angular_head_loss, dict_ssim_loss, dict_psnr_loss, dict_lpips_loss,
                                                dict_l1_loss, dict_num_images,dict_fid, full_fid, dict_similarity)

def personal_calibration(num_images):
    refer_list_file = os.path.join("data/eth_xgaze", "train_test_split.json")

    with open(refer_list_file, "r") as f:
        datastore = json.load(f)

    val_keys = datastore["val"]

    face_model_load =  np.loadtxt('data/eth_xgaze/face_model.txt') 

    cam_matrix, cam_distortion, cam_translation, cam_rotation = load_cams()

    for t, subject in enumerate(val_keys):

        for iter in range(1):
            train_dataloader = xgaze_get_val_loader(data_dir="/data/data2/aruzzi/xgaze_subjects", batch_size=1, num_val_images=200, num_workers= 0, is_shuffle= False, subject=subject)

            random_fit_images_num = []
            for i in range(num_images):
                random_fit_images_num.append(random.randint(0,199))
            
            print(random_fit_images_num)

            save_path = "/local/home/aruzzi/personal_calibration_files_sted/" 
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            hdf_path = save_path + subject[:-3] + "_nsample_" +  str(num_images) + "_iter_" + str(iter) + ".h5"
            output_h5_id = h5py.File(hdf_path, "w")

            output_face_patch = []


            ## predict images to fine tune the pre traind gaze estimator

            if not output_face_patch:
                output_face_patch = output_h5_id.create_dataset(
                "face_patch",
                shape=(200, 224, 224, 3),
                compression="lzf",
                dtype=np.uint8,
                chunks=(1, 224, 224, 3),
            )     

            counter_save_index = 0
               
            for i,entry in enumerate(train_dataloader):
                ldms = entry["ldms_b"][0]
                batch_head_mask = torch.reshape(entry["mask_b"], (1, 1, 512, 512))
                cam_ind = entry["cam_ind_b"]
                camera_matrix, camera_distortion = select_cam_matrix("eth_xgaze", cam_matrix,cam_distortion, cam_ind, subject)

                if i not in random_fit_images_num:

                    input_dict = send_data_dict_to_gpu(entry, device)
                    output_dict, loss_dict = network(input_dict)

                    image_gen = np.clip(((output_dict['image_b_hat'].detach().cpu().permute(0, 2, 3, 1).numpy() +1) * 255.0/2.0),0,255).astype(np.uint8)

                    batch_images_gen = trans_normalize(image_gen[0,:])
                    nonhead_mask = batch_head_mask < 0.5
                    nonhead_mask_c3b = nonhead_mask.expand(-1, 3, -1, -1)
                    batch_images_gen = torch.reshape(batch_images_gen,(1,3,512,512))
                    batch_images_gen[nonhead_mask_c3b] = 1.0

                    batch_images_norm_pre = normalize(
                        (batch_images_gen.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(
                            np.uint8
                        )[0],
                        camera_matrix,
                        camera_distortion,
                        face_model_load,
                        ldms,
                        224,
                    )

                    image = cv2.cvtColor(batch_images_norm_pre, cv2.COLOR_RGB2BGR)

                    output_face_patch[counter_save_index] = image
                    counter_save_index +=1

                    if i%1 == 0:
                        res_img = np.concatenate(
                            [
                                #batch_images_norm_gt.reshape(1, 224, 224, 3),
                                batch_images_norm_pre.reshape(1, 224, 224, 3),
                            ],
                            axis=2,
                        )
                        img = Image.fromarray(res_img[0])
                        log_image = wandb.Image(img)
                        wandb.log(
                            {" Target Normalized | Prediction Normalized ": log_image}
                        )
                else:

                    image_gt = ((input_dict['image_b'].detach().cpu().permute(0, 2, 3, 1).numpy() +1) * 255.0/2.0).astype(np.uint8)
                    batch_images_gt = trans_normalize(image_gt[0,:])
                    nonhead_mask = batch_head_mask < 0.5   
                    nonhead_mask_c3b = nonhead_mask.expand(-1, 3, -1, -1)  
                    batch_images_gt = torch.reshape(batch_images_gt,(1,3,512,512))      
                    batch_images_gt[nonhead_mask_c3b] = 1.0
                    batch_images_norm = normalize(
                        (batch_images_gt.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(
                            np.uint8
                        )[0],
                        camera_matrix,
                        camera_distortion,
                        face_model_load,
                        ldms,
                        224,
                    )

                    image = cv2.cvtColor(batch_images_norm, cv2.COLOR_RGB2BGR)
                    #cv2.imwrite(os.path.join(save_path, "image_" + str(i).zfill(4) + ".jpg"), image)

                    output_face_patch[counter_save_index] = image
                    counter_save_index +=1


            output_h5_id.close()


def execute_visualize(data):
    output_dict, losses_dict = network(test_visualize)
    keys = data['key'].cpu().numpy()
    for i in range(len(keys)):
        path = os.path.join(config.save_path, 'samples', str(keys[i]))
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(os.path.join(path, 'redirect_' + str(current_step) + '.png'),
                    recover_images(output_dict['image_b_hat'][i]))
        cv2.imwrite(os.path.join(path, 'redirect_all_' + str(current_step) + '.png'),
                    recover_images(output_dict['image_b_hat_all'][i]))
    walks = network.latent_walk(test_visualize)
    save_images(os.path.join(config.save_path, 'samples'), walks, keys, cycle=True)


if config.use_tensorboard and ((not config.skip_training) or config.compute_full_result):
    from tensorboardX import SummaryWriter
    tensorboard = SummaryWriter(logdir=config.save_path)
current_step = config.load_step

#for i in range(10):
#    print("here")
#    personal_calibration(i+1)

if not config.skip_training:
    logging.info('Training')
    running_losses = RunningStatistics()
    train_data_iterator = iter(train_dataloader)
    # main training loop
    for current_step in range(config.load_step, config.num_training_steps):
        # Save model
        if current_step % config.save_interval == 0 and current_step != config.load_step:
            save_model(network, current_step)
        # lr decay
        if (current_step % config.decay_steps == 0) or current_step == config.load_step:
            lr = adjust_learning_rate(network.optimizers, config.decay, int(current_step /config.decay_steps), config.lr)
            if config.use_tensorboard:
                tensorboard.add_scalar('train/lr', lr, current_step)
        # Testing loop: every specified iterations compute the test statistics
        if current_step % config.print_freq_test == 0 and current_step != 0 :
            network.eval()
            network.clean_up()
            torch.cuda.empty_cache()
            execute_test(1, current_step)
            # This might help with memory leaks
            torch.cuda.empty_cache()
        # Visualization loop
        """
        if (current_step != 0 and current_step % config.save_freq_images == 0) or current_step == config.num_training_steps - 1:
            network.eval()
            torch.cuda.empty_cache()
            with torch.no_grad():
                # save redirected, style modified samples
                execute_visualize(test_visualize)
            torch.cuda.empty_cache()
        """
        # Training step
        execute_training_step(current_step)
        # Print training loss
        if current_step != 0 and (current_step % config.print_freq_train == 0):
            running_loss_means = running_losses.means()
            logging.info('Losses at [%7d]: %s' %
                         (current_step,
                          ', '.join(['%s: %.5f' % v
                                     for v in running_loss_means.items()])))
            if config.use_tensorboard:
                for k, v in running_loss_means.items():
                    tensorboard.add_scalar('train/' + k, v, current_step)
            running_losses.reset()
    logging.info('Finished Training')
    # Save model parameters
    save_model(network, config.num_training_steps)
    del all_data
# Compute evaluation results on complete test sets
if config.compute_full_result:
    
    logging.info('Computing complete test results for final model...')

    network.eval()
    torch.cuda.empty_cache()
    execute_test(1,0)
    
    torch.cuda.empty_cache()

# Use Redirector to create new training data
if config.store_redirect_dataset:
    train_tag = 'gc/train'
    train_prefixes = all_gc_prefixes['train']
    train_dataset = HDFDataset(hdf_file_path=config.gazecapture_file,
                               prefixes=train_prefixes,
                               num_labeled_samples=config.num_labeled_samples,
                               sample_target_label=True
                               )
    train_dataset.close_hdf()
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.eval_batch_size,
                                  shuffle=False,
                                  num_workers=config.num_data_loaders,
                                  pin_memory=True,
                                  )
    current_person_id = None
    current_person_data = {}
    ofpath = os.path.join(config.save_path, 'Redirected_samples.h5')
    ofdir = os.path.dirname(ofpath)
    if not os.path.isdir(ofdir):
        os.makedirs(ofdir)
    import h5py

    h5f = h5py.File(ofpath, 'w')

    def store_person_predictions():
        global current_person_data
        if len(current_person_data) > 0:
            g = h5f.create_group(current_person_id)
            for key, data in current_person_data.items():
                g.create_dataset(key, data=data, chunks=tuple([1] + list(np.asarray(data).shape[1:])),
                                 compression='lzf', dtype=
                                 np.float32)
        current_person_data = {}

    with torch.no_grad():
        np.random.seed()
        num_batches = int(np.ceil(len(train_dataset) / config.eval_batch_size))
        for i, input_dict in enumerate(train_dataloader):
            batch_size = input_dict['image_a'].shape[0]
            input_dict = send_data_dict_to_gpu(input_dict, device)
            output_dict = network.redirect(input_dict)
            zipped_data = zip(
                input_dict['key'],
                input_dict['image_a'].cpu().numpy().astype(np.float32),
                input_dict['gaze_a'].cpu().numpy().astype(np.float32),
                input_dict['head_a'].cpu().numpy().astype(np.float32),
                output_dict['image_b_hat_r'].cpu().numpy().astype(np.float32),
                input_dict['gaze_b_r'].cpu().numpy().astype(np.float32),
                input_dict['head_b_r'].cpu().numpy().astype(np.float32)
            )

            for (person_id, image_a, gaze_a, head_a, image_b_r, gaze_b_r, head_b_r) in zipped_data:
                # Store predictions if moved on to next person
                if person_id != current_person_id:
                    store_person_predictions()
                    current_person_id = person_id
                # Now write it
                to_write = {
                    'real': True,
                    'gaze': gaze_a,
                    'head': head_a,
                    'image': image_a,
                }
                for k, v in to_write.items():
                    if k not in current_person_data:
                        current_person_data[k] = []
                    current_person_data[k].append(v)

                to_write = {
                    'real': False,
                    'gaze': gaze_b_r,
                    'head': head_b_r,
                    'image': image_b_r,
                }
                for k, v in to_write.items():
                    current_person_data[k].append(v)

            logging.info('processed batch [%04d/%04d] with %d entries.' %
                         (i + 1, num_batches, len(next(iter(input_dict.values())))))
        store_person_predictions()
    logging.info('Completed processing')
    logging.info('Done')
    del train_dataset, train_dataloader