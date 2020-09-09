import os
import time
import argparse
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from datasets.hico_constants import HicoConstants
from datasets.hico_dataset import HicoDataset, collate_fn
from model.vsgats.model import AGRNN
from model.pgception import PGception
# from model.no_frill_pose_net import fully_connect as PGception
import utils.io as io

###########################################################################################
#                                     TRAIN MODEL                                         #
###########################################################################################
def run_model(args, data_const):
    # prepare data
    train_dataset = HicoDataset(data_const=data_const, subset='train_val')
    val_dataset = HicoDataset(data_const=data_const, subset='val')
    dataset = {'train': train_dataset, 'val': val_dataset}

    train_dataloader = DataLoader(dataset=dataset['train'], batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset=dataset['val'], batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=collate_fn)
    dataloader = {'train': train_dataloader, 'val': val_dataloader}
    print("Preparing data done!!!")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'training on {device}...')
    
    # load checkpoint
    checkpoint = torch.load(args.main_pretrained, map_location=device)
    print('vsgats Checkpoint loaded!')
    # set up model and initialize it with uploaded checkpoint
    vs_gats = AGRNN(feat_type=checkpoint['feat_type'], bias=checkpoint['bias'], bn=checkpoint['bn'], dropout=checkpoint['dropout'], multi_attn=checkpoint['multi_head'], layer=checkpoint['layers'], diff_edge=checkpoint['diff_edge']) #2 )
    vs_gats.load_state_dict(checkpoint['state_dict'])
    for param in vs_gats.parameters():
        param.requires_grad = False
    vs_gats.to(device)
    vs_gats.eval()

    # [64,64,128,128], [128,256,256,256]
    print(args.b_l, args.o_c_l)
    model = PGception(action_num=args.a_n, layers=args.n_layers, classifier_mod=args.c_m, o_c_l=args.o_c_l, b_l=args.b_l,
                      last_h_c=args.last_h_c, bias=args.bias, drop=args.d_p, bn=args.bn, agg_first=args.agg_first, attn=args.attn)
    # model = PGception(action_num=args.a_n, drop=args.d_p)
    # load pretrained model
    if args.pretrained:
        print(f"loading pretrained model {args.pretrained}")
        checkpoints = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(checkpoints['state_dict'])
    model.to(device)
    # # build optimizer && criterion  
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0)
    elif args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0, amsgrad=True)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.1) #the scheduler divides the lr by 10 every 150 epochs
    # set visualization and create folder to save checkpoints
    writer = SummaryWriter(log_dir=args.log_dir + '/' + args.exp_ver)
    io.mkdir_if_not_exists(os.path.join(args.save_dir, args.exp_ver), recursive=True)
    # start training
    for epoch in range(args.start_epoch, args.epoch):
        # each epoch has a training and validation step
        epoch_loss = 0
        # for phase in ['train', 'val']:
        for phase in ['train']:
            start_time = time.time()
            running_loss = 0
            # import ipdb; ipdb.set_trace()
            for data in tqdm(dataloader[phase]):
                roi_labels = data['roi_labels']
                node_num = data['node_num']
                features = data['features']
                spatial_feat = data['spatial_feat']
                word2vec = data['word2vec']
                edge_labels = data['edge_labels']
                # pose_feat = data["pose_feat"]
                pose_normalized = data["pose_to_human"]
                pose_to_obj_offset = data["pose_to_obj_offset"]
                features, spatial_feat, word2vec, edge_labels = features.to(device), spatial_feat.to(device), word2vec.to(device), edge_labels.to(device)
                pose_to_obj_offset, pose_normalized, edge_labels =  pose_to_obj_offset.to(device), pose_normalized.to(device), edge_labels.to(device)
                # mask = mask.to(device)
                if phase == "train":
                    model.train()
                    model.zero_grad()
                    # import ipdb; ipdb.set_trace()
                    outputs = vs_gats(node_num, features, spatial_feat, word2vec, roi_labels, validation=True) + model(pose_normalized, pose_to_obj_offset)
                    loss = criterion(outputs, edge_labels)
                    loss.backward()
                    optimizer.step()
                else:
                    model.eval()
                    with torch.no_grad():
                        outputs = vs_gats(node_num, features, spatial_feat, word2vec, roi_labels, validation=True) + model(pose_normalized, pose_to_obj_offset)
                        loss = criterion(outputs, edge_labels)

                running_loss += loss.item() * edge_labels.shape[0]

            epoch_loss = running_loss / len(dataset[phase])
            # if phase == 'train':
            #     train_loss = epoch_loss 
            # else:
            #     writer.add_scalars('trainval_loss_epoch', {'train': train_loss, 'val': epoch_loss}, epoch)
            writer.add_scalars('trainval_loss_epoch', {'train': epoch_loss}, epoch)
            # print data
            if epoch ==0 or (epoch % args.print_every) == 9:
                end_time = time.time()
                print("[{}] Epoch: {}/{} Loss: {} Execution time: {}".format(\
                        phase, epoch+1, args.epoch, epoch_loss, (end_time-start_time)))
        # if args.scheduler_step and epoch % 10 == 9 and epoch < 300:     
        if args.scheduler_step:  
            scheduler.step()
        # save model epoch_loss<0.29 or 
        if epoch % args.save_every == (args.save_every - 1) and epoch >= (100-1):
            checkpoint = { 
                            'lr': args.lr,
                           'b_s': args.batch_size,
                          'bias': args.bias, 
                            'bn': args.bn, 
                       'dropout': args.d_p,
                         'o_c_l': args.o_c_l,
                           'b_l': args.b_l,
                      'last_h_c': args.last_h_c,
                           'a_n': args.a_n,
                'classifier_mod': args.c_m,
                      'n_layers': args.n_layers,
                     'agg_first': args.agg_first,
                          'attn': args.attn,
                    'state_dict': model.state_dict()
            }
            save_name = "checkpoint_" + str(epoch+1) + '_epoch.pth'
            torch.save(checkpoint, os.path.join(args.save_dir, args.exp_ver, save_name))

    writer.close()
    print('Finishing training!')

###########################################################################################
#                                 SET SOME ARGUMENTS                                      #
###########################################################################################
# define a string2boolean type function for argparse
def str2bool(arg):
    arg = arg.lower()
    if arg in ['yes', 'true', '1']:
        return True
    elif arg in ['no', 'false', '0']:
        return False
    else:
        # raise argparse.ArgumentTypeError('Boolean value expected!')
        pass

parser = argparse.ArgumentParser(description="HOI DETECTION!")

parser.add_argument('--batch_size', '--b_s', type=int, default=32,
                    help='batch size: 1')
parser.add_argument('--lr', type=float, default=0.00003, 
                    help='learning rate: 0.001')
parser.add_argument('--d_p', type=float, default=0, 
                    help='dropout parameter: 0')
parser.add_argument('--bias', type=str2bool, default='true', 
                    help="add bias to fc layers or not: True")
parser.add_argument('--bn', type=str2bool, default='false', 
                    help='use batch normailzation or not: true')
parser.add_argument('--epoch', type=int, default=200,
                    help='number of epochs to train: 300') 
parser.add_argument('--scheduler_step', '--s_s', type=int, default=0,
                    help='number of epochs to train: 0')
parser.add_argument('--o_c_l', type=list, default= [64,64,64,64],     # [64,64,128,128] [64,64,128,128]
                    help='out channel in each branch in PGception layer: [128,256,256,256]')
parser.add_argument('--b_l', type=int, nargs='+', default= [0,1,2,3],     # [128,256,256,256] [64,64,128,128]
                    help='which branchs are in PGception layer: [0,1,2,3]')
parser.add_argument('--last_h_c', type=int, default=256,
                    help='the channel of last hidden layer: 512') 
parser.add_argument('--start_epoch', type=int, default=0,
                    help='number of beginning epochs : 0') 
parser.add_argument('--c_m',  type=str, default="cat", choices=['cat', 'mean'],
                    help='the model of last classifier: cat or mean')
parser.add_argument('--optim',  type=str, default='adam', choices=['sgd', 'adam', 'amsgrad'],
                    help='which optimizer to be use: sgd, adam, amsgrad')
parser.add_argument('--a_n',  type=int, default=117,
                    help='acition number: 117')
parser.add_argument('--n_layers', type=int, default=1,
                    help='number of inception blocks: 1')
parser.add_argument('--agg_first', '--a_f', type=str2bool, default='true', 
                    help="In gcn, aggregation first means Z=W(AX), whilc aggregation later means Z=AWX: true")
parser.add_argument('--attn',  type=str2bool, default='false', 
                    help="In gcn, leverage attention mechamism or not: false")

parser.add_argument('--pretrained', '-p', type=str, default=None,
                    help='location of the pretrained model file for training: None')
parser.add_argument('--main_pretrained', '--m_p', type=str, default='./checkpoints/hico_vsgats/hico_checkpoint.pth',
                    help='Location of the checkpoint file of exciting method: ./checkpoints/hico_vsgats/hico_checkpoint.pth')
parser.add_argument('--log_dir', type=str, default='./log/hico',
                    help='path to save the log data like loss\accuracy... : ./log') 
parser.add_argument('--save_dir', type=str, default='./checkpoints/hico',
                    help='path to save the checkpoints: ./checkpoints/vcoco')
parser.add_argument('--exp_ver', '--e_v', type=str, default='v1', 
                    help='the version of code, will create subdir in log/ && checkpoints/ ')

parser.add_argument('--print_every', type=int, default=10,
                    help='number of steps for printing training and validation loss: 10')
parser.add_argument('--save_every', type=int, default=10,
                    help='number of steps for saving the model parameters: 50') 

args = parser.parse_args() 

if __name__ == "__main__":
    data_const = HicoConstants()
    run_model(args, data_const)
