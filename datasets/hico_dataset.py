import torch
import torch.nn as nn
from torch.utils.data import Dataset

import h5py
import numpy as np
import utils.io as io 
from datasets.hico_constants import HicoConstants
from datasets import metadata

import sys
import random

class HicoDataset(Dataset):
    '''
    Args:
        subset: ['train', 'val', 'train_val', 'test']
    '''
    data_sample_count = 0   # record how many times to process data sampling 

    def __init__(self, data_const=HicoConstants(), subset='train', data_aug=False, sampler=None, test=False):
        super(HicoDataset, self).__init__()
        
        self.data_aug = data_aug
        self.data_const = data_const
        self.test = test
        self.subset_ids = self._load_subset_ids(subset, sampler)
        self.sub_app_data = self._load_subset_app_data(subset)
        self.sub_spatial_data = self._load_subset_spatial_data(subset)
        self.word2vec = h5py.File(self.data_const.word2vec, 'r')
        self.sub_pose_feat = self._load_subset_pose_data(subset)

    def _load_subset_ids(self, subset, sampler):
        global_ids = io.load_json_object(self.data_const.split_ids_json)
        bad_det_ids = io.load_json_object(self.data_const.bad_faster_rcnn_det_ids)
        # skip bad instance detection image with 0-1 det
        # !NOTE: How to reduce the number of bad instance detection images
        subset_ids = [id for id in global_ids[subset] if id not in bad_det_ids['0']+bad_det_ids["1"]]
        if sampler:
            # import ipdb; ipdb.set_trace()
            ''' when changing the model, use sub-dataset to quickly show if there is something wrong '''
            subset_ids = random.sample(subset_ids, int(len(subset_ids)*sampler))
        return subset_ids

    def _load_subset_app_data(self, subset):
        print(f'Using {self.data_const.feat_type} feature...')
        if subset == 'train' or subset == 'val' or subset == 'train_val':
            return h5py.File(self.data_const.hico_trainval_data, 'r')
        elif subset == 'test':
            return h5py.File(self.data_const.hico_test_data, 'r')
        else:
            print('Please double check the name of subset!!!')
            sys.exit(1)

    def _load_subset_spatial_data(self, subset):
        if subset == 'train' or subset == 'val' or subset == 'train_val':
            return h5py.File(self.data_const.trainval_spatial_feat, 'r')
        elif subset == 'test':
            return h5py.File(self.data_const.test_spatial_feat, 'r')
        else:
            print('Please double check the name of subset!!!')
            sys.exit(1)

    def _load_subset_pose_data(self, subset):
        if subset == 'train' or subset == 'val' or subset == 'train_val':
            return h5py.File(self.data_const.trainval_keypoints_feat, 'r')
        elif subset == 'test':
            return h5py.File(self.data_const.test_keypoints_feat, 'r')
        else:
            print('Please double check the name of subset!!!')
            sys.exit(1)

    def _get_obj_one_hot(self,node_ids):
        num_cand = len(node_ids)
        obj_one_hot = np.zeros([num_cand,80])
        for i, node_id in enumerate(node_ids):
            obj_idx = int(node_id)-1
            obj_one_hot[i,obj_idx] = 1.0
        return obj_one_hot

    def _get_word2vec(self,node_ids):
        word2vec = np.empty((0,300))
        for node_id in node_ids:
            vec = self.word2vec[metadata.coco_classes[node_id]]
            word2vec = np.vstack((word2vec, vec))
        return word2vec

    def _get_interactive_label(self, edge_label):
        interactive_label = np.zeros(edge_label.shape[0])  
        interactive_label = interactive_label[:, None]
        valid_idxs = list(set(np.where(edge_label==1)[0]))
        if len(valid_idxs) > 0:
            # import ipdb; ipdb.set_trace()
            interactive_label[valid_idxs,:] = 1
        return interactive_label
        
    @staticmethod
    def displaycount():
        print("total times to process data sampling:", HicoDataset.data_sample_count)

    # def get_verb_one_hot(self,hoi_ids):
    #     num_cand = len(hoi_ids)
    #     verb_one_hot = np.zeros([num_cand,len(self.verb_to_id)])
    #     for i, hoi_id in enumerate(hoi_ids):
    #         verb_id = self.verb_to_id[self.hoi_dict[hoi_id]['verb']]
    #         verb_idx = int(verb_id)-1
    #         verb_one_hot[i,verb_idx] = 1.0
    #     return verb_one_hot

    def __len__(self):
        return len(self.subset_ids)

    def __getitem__(self, idx):
        global_id = self.subset_ids[idx]

        data = {}
        single_app_data = self.sub_app_data[global_id]
        single_spatial_data = self.sub_spatial_data[global_id]
        single_pose_data = self.sub_pose_feat[str(global_id)]
        data['roi_labels'] = single_app_data['classes'][:]
        data['node_num'] = single_app_data['node_num'].value
        data['edge_labels'] = single_app_data['edge_labels'][:]
        data['features'] = single_app_data['feature'][:]
        data['spatial_feat'] = single_spatial_data[:]
        data['word2vec'] = self._get_word2vec(data['roi_labels'])
        # data['pose_feat'] = single_pose_data[:]
        data['pose_to_human'] = single_pose_data['pose_to_human'][:]
        data['pose_to_obj_offset'] = single_pose_data['pose_to_obj_offset'][:]
        if self.test:
            data['global_id'] = global_id
            data['img_name'] = global_id + '.jpg'
            data['det_boxes'] = single_app_data['boxes'][:]
            data['roi_scores'] = single_app_data['scores'][:]
        # import ipdb; ipdb.set_trace()
        if self.data_aug:
            thresh = random.random()
            if thresh > 0.5:
                data = self._data_sampler(data)
        return data

    # for inference
    def sample_date(self, global_id):
        data = {}
        single_app_data = self.sub_app_data[global_id]
        single_spatial_data = self.sub_spatial_data[global_id]
        single_pose_data = self.sub_pose_feat[str(global_id)]
        data['global_id'] = global_id
        data['img_name'] = global_id + '.jpg'
        data['det_boxes'] = single_app_data['boxes'][:]
        data['roi_labels'] = single_app_data['classes'][:]
        data['roi_scores'] = single_app_data['scores'][:]
        data['node_num'] = single_app_data['node_num'].value
        # data['node_labels'] = single_app_data['node_labels'][:]
        data['edge_labels'] = single_app_data['edge_labels'][:]
        data['features'] = single_app_data['feature'][:]
        data['spatial_feat'] = single_spatial_data[:]
        data['word2vec'] = self._get_word2vec(data['roi_labels'])
        data['pose_to_human'] = single_pose_data['pose_to_human'][:]
        data['pose_to_obj_offset'] = single_pose_data['pose_to_obj_offset'][:]
        data['keypoints'] = single_app_data['keypoints'][:]
    
        return data
# for DatasetLoader
def collate_fn(batch):
    '''
        Default collate_fn(): https://github.com/pytorch/pytorch/blob/1d53d0756668ce641e4f109200d9c65b003d05fa/torch/utils/data/_utils/collate.py#L43
    '''
    batch_data = {}
    batch_data['global_id'] = []
    batch_data['img_name'] = []
    batch_data['det_boxes'] = []
    batch_data['roi_labels'] = []
    batch_data['roi_scores'] = []
    batch_data['node_num'] = []
    batch_data['edge_labels'] = []
    batch_data['features'] = []
    batch_data['spatial_feat'] = []
    batch_data['word2vec'] = []
    # batch_data['pose_feat'] = []
    batch_data['pose_to_human'] = []
    batch_data['pose_to_obj_offset'] = []
    batch_data['keypoints'] = []
    for data in batch:
        batch_data['roi_labels'].append(data['roi_labels'])
        batch_data['node_num'].append(data['node_num'])
        batch_data['edge_labels'].append(data['edge_labels'])
        batch_data['features'].append(data['features'])
        batch_data['spatial_feat'].append(data['spatial_feat'])
        batch_data['word2vec'].append(data['word2vec'])
        # batch_data["pose_feat"].append(data["pose_feat"])
        batch_data["pose_to_human"].append(data["pose_to_human"])
        batch_data["pose_to_obj_offset"].append(data["pose_to_obj_offset"])
        if 'global_id' in data.keys():
            batch_data['global_id'].append(data['global_id'])
            batch_data['img_name'].append(data['img_name'])
            batch_data['det_boxes'].append(data['det_boxes'])
            batch_data['roi_scores'].append(data['roi_scores'])
        if 'keypoints' in data.keys():
            batch_data['keypoints'].append(data['keypoints'])

    # import ipdb; ipdb.set_trace()
    batch_data['edge_labels'] = torch.FloatTensor(np.concatenate(batch_data['edge_labels'], axis=0))
    batch_data['features'] = torch.FloatTensor(np.concatenate(batch_data['features'], axis=0))
    batch_data['spatial_feat'] = torch.FloatTensor(np.concatenate(batch_data['spatial_feat'], axis=0))
    batch_data['word2vec'] = torch.FloatTensor(np.concatenate(batch_data['word2vec'], axis=0))
    # batch_data['pose_feat'] = torch.FloatTensor(np.concatenate(batch_data['pose_feat'], axis=0))
    batch_data['pose_to_human'] = torch.FloatTensor(np.concatenate(batch_data['pose_to_human'], axis=0))
    batch_data['pose_to_obj_offset'] = torch.FloatTensor(np.concatenate(batch_data['pose_to_obj_offset'], axis=0))

    return batch_data