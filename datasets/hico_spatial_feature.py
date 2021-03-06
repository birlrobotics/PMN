import h5py
import numpy as np 
import scipy.io as scio
import utils.io as io
from datasets.hico_constants import HicoConstants
import ipdb
from tqdm import tqdm

# class BoxFeatures():
#     def __init__(self):
#         pass

#     def compute_bbox_center(self,bbox):
#         num_boxes = bbox.shape[0]
#         center = np.zeros([num_boxes,2])
#         center[:,0] = 0.5*(bbox[:,0] + bbox[:,2])
#         center[:,1] = 0.5*(bbox[:,1] + bbox[:,3])
#         return center

#     def normalize_center(self,c,im_wh):
#         return c / im_wh

#     def compute_l2_norm(self,v):
#         return np.sqrt(np.sum(v**2))
        
#     def compute_bbox_wh(self,bbox):
#         num_boxes = bbox.shape[0]
#         wh = np.zeros([num_boxes,2])
#         wh[:,0] = 0.5*(bbox[:,2]-bbox[:,0])
#         wh[:,1] = 0.5*(bbox[:,3]-bbox[:,1])
#         return wh

#     def compute_offset(self,c1,c2,wh1,normalize):
#         offset = c2 - c1
#         if normalize:
#             offset = offset / wh1
#         return offset

#     def compute_aspect_ratio(self,wh,take_log):
#         aspect_ratio = wh[:,0] / (wh[:,1] + 1e-6)
#         if take_log:
#             aspect_ratio = np.log2(aspect_ratio+1e-6)
#         return aspect_ratio

#     def compute_bbox_size_ratio(self,wh1,wh2,take_log):
#         ratio = (wh2[:,0]*wh2[:,1])/(wh1[:,0]*wh1[:,1])
#         if take_log:
#             ratio = np.log2(ratio+1e-6)
#         return ratio
            
#     def compute_bbox_area(self,wh,im_wh,normalize):
#         bbox_area = wh[:,0]*wh[:,1]
#         if normalize:
#             norm_factor = im_wh[:,0]*im_wh[:,1]
#         else:
#             norm_factor = 1
#         bbox_area = bbox_area / norm_factor
#         return bbox_area

#     def compute_im_center(self,im_wh):
#         return im_wh/2

def center_offset(box1, box2, im_wh):
    c1 = [(box1[2]+box1[0])/2, (box1[3]+box1[1])/2]
    c2 = [(box2[2]+box2[0])/2, (box2[3]+box2[1])/2]
    offset = np.array(c1)-np.array(c2)/np.array(im_wh)
    return offset

def box_with_respect_to_img(box, im_wh):
    '''
        To get [x1/W, y1/H, x2/W, y2/H, A_box/A_img]
    '''
    # ipdb.set_trace()
    feats = [box[0]/(im_wh[0]+ 1e-6), box[1]/(im_wh[1]+ 1e-6), box[2]/(im_wh[0]+ 1e-6), box[3]/(im_wh[1]+ 1e-6)]
    box_area = (box[2]-box[0])*(box[3]-box[1])
    img_area = im_wh[0]*im_wh[1]
    feats +=[ box_area/(img_area+ 1e-6) ]
    return feats

def box1_with_respect_to_box2(box1, box2):
    feats = [ (box1[0]-box2[0])/(box2[2]-box2[0]+1e-6),
              (box1[1]-box2[1])/(box2[3]-box2[1]+ 1e-6),
              np.log((box1[2]-box1[0])/(box2[2]-box2[0]+ 1e-6)),
              np.log((box1[3]-box1[1])/(box2[3]-box2[1]+ 1e-6))   
            ]
    return feats

def cal_pose_to_box(keypoint, box):
    feat = 2 * keypoint / np.array([box[0]+box[2], box[1]+box[3]])
    return feat

def cal_pose_to_img(keypoint, im_wh):
    feat = keypoint / im_wh
    return feat

def cal_pose_to_box_offset(keypoint, box, im_wh):
    feat = (keypoint - (np.array([box[0]+box[2], box[1]+box[3]])/2)) / np.array(im_wh)
    return feat

def calculate_spatial_feats(det_boxes, im_wh):
    spatial_feats = []
    for i in range(det_boxes.shape[0]):
        for j in range(det_boxes.shape[0]):
            if j == i:
                continue
            else:
                single_feat = []
                box1_wrt_img = box_with_respect_to_img(det_boxes[i], im_wh)
                box2_wrt_img = box_with_respect_to_img(det_boxes[j], im_wh)
                box1_wrt_box2 = box1_with_respect_to_box2(det_boxes[i], det_boxes[j])
                offset = center_offset(det_boxes[i], det_boxes[j], im_wh)
                single_feat = single_feat + box1_wrt_img + box2_wrt_img + box1_wrt_box2 + offset.tolist()
                # ipdb.set_trace()
                spatial_feats.append(single_feat)
    spatial_feats = np.array(spatial_feats)
    return spatial_feats

def calculate_spatial_pose_feats(det_boxes, keypoints, im_wh):
    spatial_feats = []
    pose_to_obj = []
    pose_to_human = []
    pose_to_obj_offset = []
    # ipdb.set_trace()
    for i in range(det_boxes.shape[0]):
        for j in range(det_boxes.shape[0]):
            if j == i:
                continue
            else:
                single_feat = []
                box1_wrt_img = box_with_respect_to_img(det_boxes[i], im_wh)
                box2_wrt_img = box_with_respect_to_img(det_boxes[j], im_wh)
                box1_wrt_box2 = box1_with_respect_to_box2(det_boxes[i], det_boxes[j])
                offset = center_offset(det_boxes[i], det_boxes[j], im_wh)
                single_feat = single_feat + box1_wrt_img + box2_wrt_img + box1_wrt_box2 + offset.tolist()
                # ipdb.set_trace()
                spatial_feats.append(single_feat)
                
                if i < keypoints.shape[0]:
                    pose_to_human.append(cal_pose_to_box(keypoints[i], det_boxes[i]))
                    # pose_to_obj.append(cal_pose_to_box(keypoints[i],det_boxes[j]))
                    pose_to_obj_offset.append(cal_pose_to_box_offset(keypoints[i], det_boxes[j], im_wh))
    spatial_feats = np.array(spatial_feats)
    pose_to_human = np.array(pose_to_human)
    pose_to_obj_offset = np.array(pose_to_obj_offset)
    # pose_to_img = np.array(pose_to_img)
    # pose_to_obj = np.array(pose_to_obj)
    return spatial_feats, pose_to_obj, pose_to_human, pose_to_obj_offset 
    # return spatial_feats, pose_to_obj

if __name__=="__main__":
    data_const = HicoConstants()

    print('Load seleced boxes data file successfully...')
    split_id = io.load_json_object(data_const.split_ids_json)
    anno_list = io.load_json_object(data_const.anno_list_json)
    anno_dict = {item['global_id']: item for item in anno_list}
    print('Load original data successfully!')

    for subset in ['train_val', 'test']:
        # create saving file
        if subset == 'train_val':
            print('Creating trainval_spatial_feat.hdf5 file....')
            select_feats = h5py.File(data_const.hico_trainval_data, 'r')
            save_data = h5py.File(data_const.trainval_spatial_feat, 'w')
            norm_keypoints = h5py.File(data_const.trainval_keypoints_feat, 'w')
        else:
            print('Creating test_spatial_feat.hdf5 file....')
            select_feats = h5py.File(data_const.hico_test_data, 'r')
            save_data = h5py.File(data_const.test_spatial_feat, 'w')
            norm_keypoints = h5py.File(data_const.test_keypoints_feat, 'w')

        for global_id in tqdm(split_id[subset]):
            det_boxes = select_feats[global_id]['boxes']
            keypoints = select_feats[global_id]['keypoints']
            # !NOTE: the saved sizes of image is [H,W], please refer to the hico_mat_to_json.py file 
            img_hw = np.array(anno_dict[global_id]["image_size"])[:2]
            img_wh = [img_hw[1], img_hw[0]]
            # spatial_feats = calculate_spatial_feats(det_boxes, img_wh)
            spatial_feats, pose_to_obj, pose_to_human, pose_to_obj_offset = calculate_spatial_pose_feats(det_boxes, keypoints, img_wh)
            save_data.create_dataset(global_id, data=spatial_feats)
            # norm_keypoints.create_dataset(global_id, data=pose_to_obj)
            # save feature related to pose
            norm_keypoints.create_group(str(global_id))
            norm_keypoints[str(global_id)].create_dataset('pose_to_human', data=pose_to_human)
            # norm_keypoints[str(global_id)].create_dataset('pose_to_obj', data=pose_to_obj)
            norm_keypoints[str(global_id)].create_dataset('pose_to_obj_offset', data=pose_to_obj_offset)
        save_data.close()
        norm_keypoints.close()
    print('Finished!!!')
