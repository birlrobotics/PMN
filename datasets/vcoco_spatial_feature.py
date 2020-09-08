import os
import h5py
import ipdb
import numpy as np
from tqdm import tqdm

import utils.io as io
from datasets.vcoco import vsrl_utils as vu
from datasets.vcoco_constants import VcocoConstants

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
    feats +=[ box_area/(img_area+ 1e-6)]
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

def cal_pose_to_box_tight(keypoint, box):
    # import ipdb; ipdb.set_trace()
    minXY = np.min(keypoint, axis=0)
    maxXY = np.max(keypoint, axis=0)
    box_wh = maxXY - minXY
    if np.any(box_wh==0):
        # import ipdb; ipdb.set_trace()
        return keypoint

    return ((keypoint-minXY)*2 - box_wh) / box_wh

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
    pose_to_human = []
    # pose_to_img = []
    pose_to_obj = []
    pose_to_obj_offset = []
    pose_to_human_tight = []
    # ipdb.set_trace()
    for i in range(det_boxes.shape[0]):
        # if i < keypoints.shape[0]:
        #     try:
        #         pose_to_human.append(cal_pose_to_box(keypoints[i], det_boxes[i]))
        #         pose_to_img.append(cal_pose_to_img(keypoints[i], im_wh))
        #     except Exception as e:
        #         ipdb.set_trace()
        #         print(e)
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
                    # pose_to_human_tight.append(cal_pose_to_box_tight(keypoints[i], det_boxes[i]))
    spatial_feats = np.array(spatial_feats)
    pose_to_human = np.array(pose_to_human)
    pose_to_obj_offset = np.array(pose_to_obj_offset)
    # pose_to_human_tight = np.array(pose_to_human_tight)
    # pose_to_img = np.array(pose_to_img)
    # pose_to_obj = np.array(pose_to_obj)
    return spatial_feats, pose_to_human, pose_to_obj_offset, pose_to_obj, pose_to_human_tight

if __name__=="__main__":
    data_const = VcocoConstants()

    for subset in ["vcoco_train", "vcoco_test", "vcoco_val"]:
        # create the folder/file to save corresponding spatial features
        print(f"construct spatial features and pose features for {subset}")
        io.mkdir_if_not_exists(os.path.join(data_const.proc_dir, subset), recursive=True) 
        save_data = h5py.File(os.path.join(data_const.proc_dir, subset, 'spatial_feat.hdf5'), 'w') 
        norm_keypoints = h5py.File(os.path.join(data_const.proc_dir, subset, 'keypoints_feat.hdf5'), 'w') 
        # load selected object detection result
        vcoco_data = h5py.File(os.path.join(data_const.proc_dir, subset, 'vcoco_data.hdf5'), 'r')
        vcoco_all = vu.load_vcoco(subset)
        image_ids = vcoco_all[0]['image_id'][:,0].astype(int).tolist()
        for img_id in tqdm(set(image_ids)):
            # ipdb.set_trace()
            det_boxes = vcoco_data[str(img_id)]['boxes']
            img_wh = vcoco_data[str(img_id)]['img_size']
            keypoints = vcoco_data[str(img_id)]['keypoints']
            # spatial_feats = calculate_spatial_feats(det_boxes, img_wh)
            spatial_feats, pose_to_human, pose_to_obj_offset, pose_to_obj, pose_to_human_tight = calculate_spatial_pose_feats(det_boxes, keypoints, img_wh)
            save_data.create_dataset(str(img_id), data=spatial_feats)
            # save feature related to pose
            norm_keypoints.create_group(str(img_id))
            norm_keypoints[str(img_id)].create_dataset('pose_to_human', data=pose_to_human)
            # norm_keypoints[str(img_id)].create_dataset('pose_to_obj', data=pose_to_obj)
            norm_keypoints[str(img_id)].create_dataset('pose_to_obj_offset', data=pose_to_obj_offset)
            # norm_keypoints[str(img_id)].create_dataset('pose_to_human_tight', data=pose_to_human_tight)
        save_data.close()
        norm_keypoints.close()
    print('Finished!')

