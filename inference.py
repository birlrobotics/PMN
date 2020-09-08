from __future__ import print_function
import sys
import os
import numpy as np
import argparse
import ipdb
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib
from PIL import Image, ImageDraw, ImageFont

from model.vsgats.model import AGRNN as vsgat_hico
from model.pgception import PGception
from datasets.hico_constants import HicoConstants
from datasets.hico_dataset import HicoDataset, collate_fn
from datasets import metadata
from utils.vis_tool import vis_img, vis_img_vcoco

from datasets.vcoco_constants import VcocoConstants
from datasets.vcoco_dataset import VcocoDataset
from datasets.vcoco_dataset import collate_fn as vcoco_collate_fn
from model.vsgats.vcoco_model import AGRNN as vsgat_vcoco

matplotlib.use('TKAgg')

def main(args):
    # Load checkpoint and set up model
    try:
        # use GPU if available else revert to CPU
        device = torch.device('cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu')
        print("Testing on", device)

        # set up model and initialize it with uploaded checkpoint
        if args.dataset == 'hico':
            # load checkpoint
            checkpoint = torch.load(args.main_pretrained_hico, map_location=device)
            print('vsgats Checkpoint loaded!')
            pg_checkpoint = torch.load(args.pretrained_hico, map_location=device)
            data_const = HicoConstants(feat_type=checkpoint['feat_type'])
            vs_gats = vsgat_hico(feat_type=checkpoint['feat_type'], bias=checkpoint['bias'], bn=checkpoint['bn'], dropout=checkpoint['dropout'], multi_attn=checkpoint['multi_head'], layer=checkpoint['layers'], diff_edge=checkpoint['diff_edge']) #2 )
        if args.dataset == 'vcoco':
            # load checkpoint
            checkpoint = torch.load(args.main_pretrained_vcoco, map_location=device)
            print('vsgats Checkpoint loaded!')
            pg_checkpoint = torch.load(args.pretrained_vcoco, map_location=device)
            data_const = VcocoConstants()
            vs_gats = vsgat_vcoco(feat_type=checkpoint['feat_type'], bias=checkpoint['bias'], bn=checkpoint['bn'], dropout=checkpoint['dropout'], multi_attn=checkpoint['multi_head'], layer=checkpoint['layers'], diff_edge=checkpoint['diff_edge']) #2 )
        vs_gats.load_state_dict(checkpoint['state_dict'])
        vs_gats.to(device)
        vs_gats.eval()

        print(pg_checkpoint['o_c_l'], pg_checkpoint['b_l'], pg_checkpoint['attn'], pg_checkpoint['lr'], pg_checkpoint['dropout'])
        # pgception = PGception(action_num=24, classifier_mod='cat', o_c_l=[64,64,128,128], last_h_c=256, bias=pg_checkpoint['bias'], drop=pg_checkpoint['dropout'], bn=pg_checkpoint['bn'])
        pgception = PGception(action_num=pg_checkpoint['a_n'], layers=1, classifier_mod=pg_checkpoint['classifier_mod'], o_c_l=pg_checkpoint['o_c_l'], last_h_c=pg_checkpoint['last_h_c'], bias=pg_checkpoint['bias'], drop=pg_checkpoint['dropout'], bn=pg_checkpoint['bn'], agg_first=pg_checkpoint['agg_first'], attn=pg_checkpoint['attn'], b_l=pg_checkpoint['b_l'])
        # pgception = PGception(action_num=pg_checkpoint['a_n'], drop=pg_checkpoint['dropout'])
        pgception.load_state_dict(pg_checkpoint['state_dict'])
        pgception.to(device)
        pgception.eval()
        print('Constructed model successfully!')
    except Exception as e:
        print('Failed to load checkpoint or construct model!', e)
        sys.exit(1)

    # prepare for data 
    if args.dataset == 'hico':
        original_imgs_dir = os.path.join(data_const.infer_dir, 'original_imgs/hico')
        # original_imgs_dir = './datasets/hico/images/test2015'
        save_path = os.path.join(data_const.infer_dir,'processed_imgs/hico')
        test_dataset = HicoDataset(data_const=data_const, subset='test')
        dataloader = sorted(os.listdir(original_imgs_dir))
        # dataloader = ['HICO_test2015_00000128.jpg']
    else:
        original_imgs_dir = os.path.join(data_const.infer_dir, 'original_imgs/vcoco')
        # original_imgs_dir = './datasets/vcoco/coco/images/val2014'
        save_path = os.path.join(data_const.infer_dir,'processed_imgs/vcoco')
        test_dataset = VcocoDataset(data_const=data_const, subset='vcoco_test',  pg_only=False)
        # dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, collate_fn=vcoco_collate_fn)
        dataloader = sorted(os.listdir(original_imgs_dir))
        dataloader = ['COCO_val2014_000000150361.jpg']

    if not os.path.exists(original_imgs_dir):
        os.makedirs(original_imgs_dir)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print('result images will be kept here{}'.format(save_path))

    # ipdb.set_trace()
    for data in tqdm(dataloader):
        # load corresponding data
        # print("Testing on image named {}".format(img))
        if args.dataset == 'hico':
            img = data
            global_id = data.split('.')[0]
            test_data = test_dataset.sample_date(global_id)
            test_data = collate_fn([test_data])
            det_boxes = test_data['det_boxes'][0]
            roi_scores = test_data['roi_scores'][0]
            roi_labels = test_data['roi_labels'][0]
            keypoints = test_data['keypoints'][0]
            edge_labels = test_data['edge_labels']
            node_num = test_data['node_num']
            features = test_data['features'] 
            spatial_feat = test_data['spatial_feat'] 
            word2vec = test_data['word2vec']
            pose_normalized = test_data["pose_to_human"]
            pose_to_obj_offset = test_data["pose_to_obj_offset"]
        else:
            # global_id = data['global_id'][0]
            img = data
            global_id = str(int((data.split('.')[0].split('_')[-1])))
            test_data = test_dataset.sample_date(global_id)
            test_data = vcoco_collate_fn([test_data])
            # img = data['img_name'][0][:].astype(np.uint8).tostring().decode('ascii').split("/")[-1]
            # test_data = data
            det_boxes = test_data['det_boxes'][0]
            roi_scores = test_data['roi_scores'][0]
            roi_labels = test_data['roi_labels'][0]
            edge_labels = test_data['edge_labels']
            node_num = test_data['node_num']
            features = test_data['features'] 
            spatial_feat = test_data['spatial_feat']
            word2vec = test_data['word2vec']
            pose_normalized = test_data["pose_to_human"]
            pose_to_obj_offset = test_data["pose_to_obj_offset"]

        # inference
        pose_to_obj_offset, pose_normalized, features, spatial_feat, word2vec = pose_to_obj_offset.to(device), pose_normalized.to(device), features.to(device), spatial_feat.to(device), word2vec.to(device)
        outputs, attn, attn_lang = vs_gats(node_num, features, spatial_feat, word2vec, [roi_labels])    # !NOTE: it is important to set [roi_labels] 
        pg_outputs = pgception(pose_normalized, pose_to_obj_offset)
        # action_score = nn.Sigmoid()(outputs+pg_outputs)
        # action_score = action_score.cpu().detach().numpy()
        det_outputs = nn.Sigmoid()(outputs+pg_outputs)
        det_outputs = det_outputs.cpu().detach().numpy()

        # show result
        # import ipdb; ipdb.set_trace()
        if args.dataset == 'hico':
            image = Image.open(os.path.join('datasets/hico/images/test2015', img)).convert('RGB')
            image_temp = image.copy()
            gt_img = vis_img(image, det_boxes, roi_labels, roi_scores, edge_labels.cpu().numpy(), score_thresh=0.5)
            det_img = vis_img(image_temp, det_boxes, roi_labels, roi_scores, det_outputs, score_thresh=0.5)
        if args.dataset == 'vcoco':
            image = Image.open(os.path.join(data_const.original_image_dir, 'val2014', img)).convert('RGB')
            image_temp = image.copy()
            gt_img = vis_img_vcoco(image, det_boxes, roi_labels, roi_scores, edge_labels.cpu().numpy(), score_thresh=0.1)
            det_img = vis_img_vcoco(image_temp, det_boxes, roi_labels, roi_scores, det_outputs, score_thresh=0.5)
    
        # det_img.save('/home/birl/ml_dl_projects/bigjun/hoi/VS_GATs/inference_imgs/original_imgs'+'/'+img)
        det_img.save(save_path+'/'+img.split("/")[-1])
        # fig = plt.figure(figsize=(100,100))
        # fig.suptitle(img, fontsize=16)
        # ax1 = plt.subplot(1,2,1)
        # ax1.set_title('ground_truth')
        # plt.imshow(np.array(gt_img))
        # plt.axis('off')
        # ax2 = plt.subplot(1,2,2)
        # ax2.set_title('action_detection')
        # plt.imshow(np.array(det_img))
        # plt.axis('off')
        # plt.ion()
        # plt.pause(10)
        # plt.close()
        # ipdb.set_trace()

def str2bool(arg):
    arg = arg.lower()
    if arg in ['yes', 'true', '1']:
        return True
    elif arg in ['no', 'false', '0']:
        return False
    else:
        # raise argparse.ArgumentTypeError('Boolean value expected!')
        pass

if __name__ == "__main__":
    # set some arguments
    parser = argparse.ArgumentParser(description='inference of the model')

    parser.add_argument('--dataset', type=str, default='hico', choices=['hico', 'vcoco'],
                        help='which datasets you choose: [hico, vcoco]')

    parser.add_argument('--pretrained_hico', '-p_h', type=str, default='./checkpoints/hico/640064_mid128_offset_posetoh_trainval/checkpoint_160_epoch.pth', 
                        help='Location of the checkpoint file: ./checkpoints/hico/640064_mid128_offset_posetoh_trainval/checkpoint_160_epoch.pth')

    parser.add_argument('--pretrained_vcoco', '-p_v', type=str, default='./checkpoints/vcoco/v4_640064_mid128_offset_posetohuman_trainval/checkpoint_400_epoch.pth', 
                        help='Location of the checkpoint file: ./checkpoints/vcoco/v4_640064_mid128_offset_posetohuman_trainval/checkpoint_400_epoch.pth')

    parser.add_argument('--main_pretrained_hico', '--m_p_h', type=str, default='/home/birl/ml_dl_projects/bigjun/hoi/PGception/checkpoints/hico_vsgats/hico_checkpoint.pth',
                        help='Location of the checkpoint file of exciting method: /home/birl/ml_dl_projects/bigjun/hoi/PGception/checkpoints/hico_vsgats/hico_checkpoint.pth')

    parser.add_argument('--main_pretrained_vcoco', '--m_p_v', type=str, default='/home/birl/ml_dl_projects/bigjun/hoi/PGception/checkpoints/vcoco_vsgats/checkpoint_600_epoch.pth',
                        help='Location of the checkpoint file of exciting method: /home/birl/ml_dl_projects/bigjun/hoi/PGception/checkpoints/hico_vsgats/checkpoint_600_epoch.pth')

    parser.add_argument('--gpu', type=str2bool, default='true',
                        help='use GPU or not: true')

    parser.add_argument('--random_data', type=str2bool, default='false',
                        help='select data randomly from the test dataset: true')
    args = parser.parse_args()
    # inferencing
    main(args)