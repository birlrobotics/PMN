import os
import ipdb
from tqdm import tqdm

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import utils.io as io
from datasets.hico_constants import HicoConstants
import h5py
import json

import torchvision
import torch
# from utils.vis_tool import vis_img

if __name__ == "__main__":
    # set up object detection model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, rpn_post_nms_top_n_test=200, \
                                                                 box_batch_size_per_image=128, box_score_thresh=0.1, box_nms_thresh=0.3)
    device = torch.device('cuda:0')
    model.cuda()
    model.eval()

    print('Begining...')
    data_const = HicoConstants()
    anno_list = io.load_json_object(data_const.anno_list_json)
    io.mkdir_if_not_exists(data_const.proc_dir, recursive=True)
    faster_rcnn_det_data = h5py.File(data_const.faster_det_fc7_feat, 'w')
    nms_keep_indices_dict = {}
    for ind in tqdm(range(len(anno_list))):
        root = 'datasets/hico/images/'
        image = Image.open(os.path.join(root, anno_list[ind]['image_path_postfix'])).convert('RGB')
        input = torchvision.transforms.functional.to_tensor(image)
        input = input.to(device)
        outputs = model([input], save_feat=True)

        # save object detection result data
        img_id = anno_list[ind]['global_id']
        faster_rcnn_det_data.create_group(str(img_id))
        faster_rcnn_det_data[str(img_id)].create_dataset(name='boxes', data=outputs[0]['boxes'].cpu().detach().numpy()) 
        faster_rcnn_det_data[str(img_id)].create_dataset(name='scores', data=outputs[0]['scores'].cpu().detach().numpy())  
        faster_rcnn_det_data[str(img_id)].create_dataset(name='fc7_feat', data=outputs[0]['fc7_feat'].cpu().detach().numpy()) 
        # faster_rcnn_det_data[str(img_id)].create_dataset(name='pool_feaet', data=outputs[0]['pool_feat'].cpu().detach().numpy())
        nms_keep_indices_dict[str(img_id)] = outputs[0]['labels']
    faster_rcnn_det_data.close()
    io.dump_json_object(nms_keep_indices_dict, os.path.join(data_const.proc_dir, 'nms_keep_indices.json'))

    faster_rcnn_det_data.close()
    print('Make detection data successfully!')

    # set up the human pose detection model
    model_pose = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True, rpn_post_nms_top_n_test=200, box_batch_size_per_image=128, \
                                                                 box_score_thresh=0.1, box_nms_thresh=0.3)
    model_pose.to(device)
    model_pose.eval()

    faster_rcnn_pose_data = h5py.File(data_const.faster_rcnn_pose_feat, 'w') 
    for ind in tqdm(range(len(anno_list))):
        root = 'datasets/hico/images/'
        image = Image.open(os.path.join(root, anno_list[ind]['image_path_postfix'])).convert('RGB')
        input = torchvision.transforms.functional.to_tensor(image)
        input = input.to(device)
        outputs_pose = model_pose([input])

        img_id = anno_list[ind]['global_id']
        faster_rcnn_pose_data.create_group(str(img_id))
        faster_rcnn_pose_data[str(img_id)].create_dataset(name='boxes', data=outputs_pose[1][0]["boxes"].cpu().detach().numpy())
        faster_rcnn_pose_data[str(img_id)].create_dataset(name='keypoints', data=outputs_pose[1][0]["keypoints"].cpu().detach().numpy())
        # ipdb.set_trace()
    faster_rcnn_pose_data.close()