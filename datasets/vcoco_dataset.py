import torch
import torch.nn as nn
from torch.utils.data import Dataset

import h5py
import numpy as np
import utils.io as io
from datasets import vcoco_metadata
from datasets.vcoco import vsrl_utils as vu
from datasets.vcoco_constants import VcocoConstants

import os
import sys
import random

class VcocoDataset(Dataset):
    '''
    Args:
        subset: ['vcoco_train', 'vcoco_val', 'vcoco_test', 'vcoco_trainval']
    '''
    data_sample_count = 0   # record how many times to process data sampling 

    def __init__(self, data_const=VcocoConstants(), subset='vcoco_train', pg_only=False, data_aug=False, sampler=None):
        super(VcocoDataset, self).__init__()
        
        self.data_const = data_const
        self.pg_only = pg_only
        self.subset_ids = self._load_subset_ids(subset, sampler)
        self.sub_app_data = self._load_subset_app_data(subset)
        if not pg_only:
            self.sub_spatial_data = self._load_subset_spatial_data(subset)
            self.word2vec = h5py.File(self.data_const.word2vec, 'r')
        self.sub_pose_feat = self._load_subset_pose_data(subset)

    def _load_subset_ids(self, subset, sampler):
        # import ipdb; ipdb.set_trace()
        vcoco = vu.load_vcoco(subset)
        subset_ids = list(set(vcoco[0]['image_id'][:,0].astype(int).tolist()))
        if sampler:
            # import ipdb; ipdb.set_trace()
            ''' when changing the model, use sub-dataset to quickly show if there is something wrong '''
            subset_ids = random.sample(subset_ids, int(len(subset_ids)*sampler))
        return subset_ids

    def _load_subset_app_data(self, subset):
        return h5py.File(os.path.join(self.data_const.proc_dir, subset, 'vcoco_data.hdf5'), 'r')

    def _load_subset_spatial_data(self, subset):
        return h5py.File(os.path.join(self.data_const.proc_dir, subset, 'spatial_feat.hdf5'), 'r')

    def _load_subset_pose_data(self, subset):
        return h5py.File(os.path.join(self.data_const.proc_dir, subset, 'keypoints_feat.hdf5'), 'r')
    
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
            vec = self.word2vec[vcoco_metadata.coco_classes[node_id]]
            word2vec = np.vstack((word2vec, vec))
        return word2vec
        
    @staticmethod
    def displaycount():
        print("total times to process data sampling:", VcocoDataset.data_sample_count)


    def __len__(self):
        return len(self.subset_ids)

    def __getitem__(self, idx):
        global_id = self.subset_ids[idx]

        data = {}
        single_app_data = self.sub_app_data[str(global_id)]
        single_pose_data = self.sub_pose_feat[str(global_id)]
        if not self.pg_only:
            single_spatial_data = self.sub_spatial_data[str(global_id)]
            data['global_id'] = global_id
            data['img_name'] = single_app_data['img_name']
            data['det_boxes'] = single_app_data['boxes'][:]
            data['roi_labels'] = single_app_data['classes'][:]
            data['roi_scores'] = single_app_data['scores'][:]
            data['node_num'] = single_app_data['node_num'].value
            # data['node_labels'] = single_app_data['node_labels'][:]
            data['edge_labels'] = single_app_data['edge_labels'][:]
            # data['edge_num'] = data['edge_labels'].shape[0]
            data['features'] = single_app_data['feature'][:]
            data['spatial_feat'] = single_spatial_data[:]
            # data['node_one_hot'] = self._get_obj_one_hot(data['roi_labels'])
            data['word2vec'] = self._get_word2vec(data['roi_labels'])
            # data['interactive_label'] = self._get_interactive_label(data['edge_labels'])
        # data['pose_labels'] = single_app_data['pose_labels'][:]
        data['pose_to_human'] = single_pose_data['pose_to_human'][:]
        # data['pose_to_obj'] = single_pose_data['pose_to_obj'][:]
        # data['pose_to_human_tight'] = single_pose_data['pose_to_human_tight'][:]
        data['pose_to_obj_offset'] = single_pose_data['pose_to_obj_offset'][:]
        data['edge_labels'] = single_app_data['edge_labels'][:]
        # if self.pg_only:
        #     # delete invariable pose which is all zeros
        #     mask_list = []
        #     for i in range(data['pose_feat'].shape[0]):
        #         if np.all(data['pose_feat'][i] == 0):
        #             # import ipdb; ipdb.set_trace()
        #             mask_list.append(i)
        #     data['pose_labels'] = np.delete(data['pose_labels'], mask_list, 0)
        #     data['pose_feat'] = np.delete(data['pose_feat'], mask_list, 0)

        # mask_list = []
        # for i in range(data['pose_feat'].shape[0]):
        #     if np.all(data['pose_feat'][i] == 0):
        #         # import ipdb; ipdb.set_trace()
        #         mask_list.append(i)
        # # import ipdb; ipdb.set_trace()
        # mask = np.array(np.ones_like(data['edge_labels']))
        # mask[mask_list] = 0
        # data['mask'] = mask
        return data


    # for inference
    def sample_date(self, global_id):
        data = {}
        # import ipdb; ipdb.set_trace()
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
    # batch_data['edge_num'] = []
    # batch_data['node_labels'] = []
    batch_data['features'] = []
    batch_data['spatial_feat'] = []
    # batch_data['node_one_hot'] = []
    batch_data['word2vec'] = []
    # batch_data['interactive_label'] = []
    # batch_data['pose_feat'] = []
    batch_data['pose_to_human'] = []
    batch_data['pose_to_human_tight'] = []
    batch_data['pose_to_obj'] = []
    batch_data['pose_to_obj_offset'] = []
    # batch_data['pose_labels'] = []
    batch_data['mask'] = []
    for data in batch:
        batch_data['global_id'].append(data['global_id'])
        batch_data['img_name'].append(data['img_name'])
        batch_data['det_boxes'].append(data['det_boxes'])
        batch_data['roi_labels'].append(data['roi_labels'])
        batch_data['roi_scores'].append(data['roi_scores'])
        batch_data['node_num'].append(data['node_num'])
        # batch_data['node_labels'].append(data['node_labels'])
        batch_data['edge_labels'].append(data['edge_labels'])
        # batch_data['edge_num'].append(data['edge_num'])
        batch_data['features'].append(data['features'])
        batch_data['spatial_feat'].append(data['spatial_feat'])
        # batch_data['node_one_hot'].append(data['node_one_hot'])
        batch_data['word2vec'].append(data['word2vec'])
        # batch_data['interactive_label'].append(data['interactive_label'])
        # batch_data["pose_feat"].append(data["pose_feat"])
        batch_data["pose_to_human"].append(data["pose_to_human"])
        # batch_data["pose_to_obj"].append(data["pose_to_obj"])
        # batch_data["pose_to_human_tight"].append(data["pose_to_human_tight"])
        batch_data["pose_to_obj_offset"].append(data["pose_to_obj_offset"])
        # batch_data["pose_labels"].append(data["pose_labels"])
        # batch_data["mask"].append(data["mask"])

    # import ipdb; ipdb.set_trace()
    # batch_data['node_labels'] = torch.FloatTensor(np.concatenate(batch_data['node_labels'], axis=0))
    batch_data['edge_labels'] = torch.FloatTensor(np.concatenate(batch_data['edge_labels'], axis=0))
    batch_data['features'] = torch.FloatTensor(np.concatenate(batch_data['features'], axis=0))
    batch_data['spatial_feat'] = torch.FloatTensor(np.concatenate(batch_data['spatial_feat'], axis=0))
    # batch_data['node_one_hot'] = torch.FloatTensor(np.concatenate(batch_data['node_one_hot'], axis=0))
    batch_data['word2vec'] = torch.FloatTensor(np.concatenate(batch_data['word2vec'], axis=0))
    # batch_data['interactive_label'] = torch.FloatTensor(np.concatenate(batch_data['interactive_label'], axis=0))
    # batch_data['pose_feat'] = torch.FloatTensor(np.concatenate(batch_data['pose_feat'], axis=0))
    batch_data['pose_to_human'] = torch.FloatTensor(np.concatenate(batch_data['pose_to_human'], axis=0))
    # batch_data['pose_to_obj'] = torch.FloatTensor(np.concatenate(batch_data['pose_to_obj'], axis=0))
    # batch_data['pose_to_human_tight'] = torch.FloatTensor(np.concatenate(batch_data['pose_to_human_tight'], axis=0))
    batch_data['pose_to_obj_offset'] = torch.FloatTensor(np.concatenate(batch_data['pose_to_obj_offset'], axis=0))
    # batch_data['pose_labels'] = torch.FloatTensor(np.concatenate(batch_data['pose_labels'], axis=0))
    # batch_data['mask'] = torch.FloatTensor(np.concatenate(batch_data['mask'], axis=0))
    return batch_data