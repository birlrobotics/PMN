from __future__ import print_function
import sys
import os
import ipdb
import pickle
import h5py
import argparse
import numpy as np
from tqdm import tqdm

import dgl
import networkx as nx
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from model.vsgats.model import AGRNN
from model.pgception import PGception
# from model.no_frill_pose_net import fully_connect as PGception
from datasets.hico_constants import HicoConstants
from datasets.hico_dataset import HicoDataset, collate_fn
from datasets import metadata
import utils.io as io

def main(args):
    # use GPU if available else revert to CPU
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    print("Testing on", device)

    # Load checkpoint and set up model
    try:
        # load checkpoint
        checkpoint = torch.load(args.main_pretrained, map_location=device)
        print('vsgats Checkpoint loaded!')
        pg_checkpoint = torch.load(args.pretrained, map_location=device)

        # set up model and initialize it with uploaded checkpoint
        # ipdb.set_trace()
        if not args.exp_ver:
            args.exp_ver = args.pretrained.split("/")[-2]+"_"+args.pretrained.split("/")[-1].split("_")[-2]
            # import ipdb; ipdb.set_trace()
        data_const = HicoConstants(feat_type=checkpoint['feat_type'], exp_ver=args.exp_ver)
        vs_gats = AGRNN(feat_type=checkpoint['feat_type'], bias=checkpoint['bias'], bn=checkpoint['bn'], dropout=checkpoint['dropout'], multi_attn=checkpoint['multi_head'], layer=checkpoint['layers'], diff_edge=checkpoint['diff_edge']) #2 )
        vs_gats.load_state_dict(checkpoint['state_dict'])
        vs_gats.to(device)
        vs_gats.eval()

        print(pg_checkpoint['o_c_l'], pg_checkpoint['b_l'], pg_checkpoint['attn'], pg_checkpoint['lr'], pg_checkpoint['dropout'])
        pgception = PGception(action_num=pg_checkpoint['a_n'], layers=1, classifier_mod=pg_checkpoint['classifier_mod'], o_c_l=pg_checkpoint['o_c_l'], last_h_c=pg_checkpoint['last_h_c'], bias=pg_checkpoint['bias'], drop=pg_checkpoint['dropout'], bn=pg_checkpoint['bn'], agg_first=pg_checkpoint['agg_first'], attn=pg_checkpoint['attn'], b_l=pg_checkpoint['b_l'])

        pgception.load_state_dict(pg_checkpoint['state_dict'])
        pgception.to(device)
        pgception.eval()
        print('Constructed model successfully!')
    except Exception as e:
        print('Failed to load checkpoint or construct model!', e)
        sys.exit(1)
    
    print('Creating hdf5 file for predicting hoi dets ...')
    if not os.path.exists(data_const.result_dir):
        os.mkdir(data_const.result_dir)
    pred_hoi_dets_hdf5 = os.path.join(data_const.result_dir, 'pred_hoi_dets.hdf5')
    pred_hois = h5py.File(pred_hoi_dets_hdf5,'w')

    test_dataset = HicoDataset(data_const=data_const, subset='test', test=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    # for global_id in tqdm(test_list): 
    for data in tqdm(test_dataloader):
        global_id = data['global_id'][0]
        det_boxes = data['det_boxes'][0]
        roi_scores = data['roi_scores'][0]
        roi_labels = data['roi_labels'][0]
        node_num = data['node_num']
        features = data['features'] 
        spatial_feat = data['spatial_feat']
        word2vec = data['word2vec']
        pose_normalized = data["pose_to_human"]
        pose_to_obj_offset = data["pose_to_obj_offset"]

        # referencing
        pose_to_obj_offset, pose_normalized, features, spatial_feat, word2vec = pose_to_obj_offset.to(device), pose_normalized.to(device), features.to(device), spatial_feat.to(device), word2vec.to(device)
        outputs, attn, attn_lang = vs_gats(node_num, features, spatial_feat, word2vec, [roi_labels])    # !NOTE: it is important to set [roi_labels] 
        pg_outputs = pgception(pose_normalized, pose_to_obj_offset)
        action_scores = nn.Sigmoid()(outputs+pg_outputs)
        action_scores = action_scores.cpu().detach().numpy()
        # save detection result
        pred_hois.create_group(global_id)
        det_data_dict = {}
        h_idxs = np.where(roi_labels == 1)[0]
        for h_idx in h_idxs:
            for i_idx in range(len(roi_labels)):
                if i_idx == h_idx:
                    continue
                if h_idx > i_idx:
                    edge_idx = h_idx * (node_num[0] - 1) + i_idx
                else:
                    edge_idx = h_idx * (node_num[0] - 1) + i_idx - 1
                    
                score = roi_scores[h_idx] * roi_scores[i_idx] * action_scores[edge_idx]
                try:
                    hoi_ids = metadata.obj_hoi_index[roi_labels[i_idx]]
                except Exception as e:
                    ipdb.set_trace()
                for hoi_idx in range(hoi_ids[0]-1, hoi_ids[1]):
                    hoi_pair_score = np.concatenate((det_boxes[h_idx], det_boxes[i_idx], np.expand_dims(score[metadata.hoi_to_action[hoi_idx]], 0)), axis=0)
                    if str(hoi_idx+1).zfill(3) not in det_data_dict.keys():
                        det_data_dict[str(hoi_idx+1).zfill(3)] = hoi_pair_score[None,:]
                    else:
                        det_data_dict[str(hoi_idx+1).zfill(3)] = np.vstack((det_data_dict[str(hoi_idx+1).zfill(3)], hoi_pair_score[None,:]))
        for k, v in det_data_dict.items():
            pred_hois[global_id].create_dataset(k, data=v)

    pred_hois.close()

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
    parser = argparse.ArgumentParser(description='Evaluate the model')

    parser.add_argument('--pretrained', '-p', type=str, default='checkpoints/v3_2048/epoch_train/checkpoint_300_epoch.pth', 
                        help='Location of the checkpoint file: ./checkpoints/checkpoint_150_epoch.pth')

    parser.add_argument('--main_pretrained', '--m_p', type=str, default='./checkpoints/hico_vsgats/hico_checkpoint.pth',
                        help='Location of the checkpoint file of exciting method: ./checkpoints/hico_vsgats/hico_checkpoint.pth')

    parser.add_argument('--gpu', type=str2bool, default='true',
                        help='use GPU or not: true')

    # parser.add_argument('--feat_type', '--f_t', type=str, default='fc7', required=True, choices=['fc7', 'pool'],
    #                     help='if using graph head, here should be pool: default(fc7) ')

    parser.add_argument('--exp_ver', '--e_v', type=str, default=None, 
                        help='the version of code, will create subdir in log/ && checkpoints/ ')

    parser.add_argument('--rewrite', '-r', action='store_true', default=False,
                        help='overwrite the detection file')

    args = parser.parse_args()
    # data_const = HicoConstants(feat_type=args.feat_type, exp_ver=args.exp_ver)
    # inferencing
    main(args)