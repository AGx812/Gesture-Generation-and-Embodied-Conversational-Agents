import argparse
import math
import os
import random
import warnings
import lmdb
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import pyarrow
import loader_v2 as loader
import processor_v2 as processor
from utils.ted_db_utils import *

from os.path import join as jn

from config.parse_args import parse_args

warnings.filterwarnings('ignore')


# base_path = os.path.dirname(os.path.realpath(__file__))
base_path = '.'
data_path = jn(base_path, 'data')

models_s2ag_path = jn(base_path, 'models', 's2ag_v2_mfcc_run_3')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Speech to Emotive Gestures')
parser.add_argument('--dataset-s2ag', type=str, default='ted_db', metavar='D-S2G',
                    help='dataset to train and evaluate speech to emotive gestures (default: ted_db)')
parser.add_argument('--dataset-test', type=str, default='ted_db', metavar='D-TST',
                    help='dataset to test emotive gestures (options: ted_db, genea_challenge_2020)')
parser.add_argument('-dap', '--dataset-s2ag-already-processed',
                    help='Optional. Set to True if dataset has already been processed.' +
                         'If not, or if you are not sure, set it to False.',
                    type=str2bool, default=True)
parser.add_argument('-c', '--config', required=True, is_config_file=True, help='Config file path')
parser.add_argument('--frame-drop', type=int, default=2, metavar='FD',
                    help='frame down-sample rate (default: 2)')
parser.add_argument('--train_s2ag', type=bool, default=False, metavar='T-s2ag',
                    help='train the s2ag model (default: False)')
parser.add_argument('--use-multiple-gpus', type=bool, default=False, metavar='T',
                    help='use multiple GPUs if available (default: False)')
parser.add_argument('--s2ag-load-last-best', type=bool, default=True, metavar='s2ag-LB',
                    help='load the most recent best model for s2ag (default: True)')
parser.add_argument('--batch-size', type=int, default=512, metavar='B',
                    help='input batch size for training (default: 32)')
parser.add_argument('--num-worker', type=int, default=4, metavar='W',
                    help='number of threads? (default: 4)')
parser.add_argument('--s2ag-start-epoch', type=int, default=290, metavar='s2ag-SE',
                    help='starting epoch of training of s2ag (default: 0)')
parser.add_argument('--s2ag-num-epoch', type=int, default=500, metavar='s2ag-NE',
                    help='number of epochs to train s2ag (default: 1000)')
# parser.add_argument('--window-length', type=int, default=1, metavar='WL',
#                     help='max number of past time steps to take as input to transformer decoder (default: 60)')
parser.add_argument('--base-tr', type=float, default=1., metavar='TR',
                    help='base teacher rate (default: 1.0)')
parser.add_argument('--step', type=list, default=0.05 * np.arange(20), metavar='[S]',
                    help='fraction of steps when learning rate will be decreased (default: [0.5, 0.75, 0.875])')
parser.add_argument('--lr-s2ag-decay', type=float, default=0.999, metavar='LRD-s2ag',
                    help='learning rate decay for s2ag (default: 0.999)')
parser.add_argument('--gradient-clip', type=float, default=0.1, metavar='GC',
                    help='gradient clip threshold (default: 0.1)')
parser.add_argument('--nesterov', action='store_true', default=True,
                    help='use nesterov')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='D',
                    help='Weight decay (default: 5e-4)')
parser.add_argument('--upper-body-weight', type=float, default=1., metavar='UBW',
                    help='loss weight on the upper body joint motions (default: 2.05)')
parser.add_argument('--affs-reg', type=float, default=0.8, metavar='AR',
                    help='regularization for affective features loss (default: 0.01)')
parser.add_argument('--quat-norm-reg', type=float, default=0.1, metavar='QNR',
                    help='regularization for unit norm constraint (default: 0.01)')
parser.add_argument('--quat-reg', type=float, default=1.2, metavar='QR',
                    help='regularization for quaternion loss (default: 0.01)')
parser.add_argument('--recons-reg', type=float, default=1.2, metavar='RCR',
                    help='regularization for reconstruction loss (default: 1.2)')
parser.add_argument('--eval-interval', type=int, default=1, metavar='EI',
                    help='interval after which model is evaluated (default: 1)')
parser.add_argument('--log-interval', type=int, default=200, metavar='LI',
                    help='interval after which log is printed (default: 100)')
parser.add_argument('--save-interval', type=int, default=10, metavar='SI',
                    help='interval after which model is saved (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA training')
parser.add_argument('--pavi-log', action='store_true', default=False,
                    help='pavi log')
parser.add_argument('--print-log', action='store_true', default=True,
                    help='print log')
parser.add_argument('--save-log', action='store_true', default=True,
                    help='save log')
# TO ADD: save_result

args = parser.parse_args()
args.data_path = data_path
randomized = False
args.train_s2ag = False

s2ag_config_args = parse_args()

args.work_dir_s2ag = jn(models_s2ag_path, args.dataset_s2ag)
os.makedirs(args.work_dir_s2ag, exist_ok=True)

args.video_save_path = jn(base_path, 'outputs', args.dataset_test, 'videos_trimodal_style')
os.makedirs(args.video_save_path, exist_ok=True)
args.quantitative_save_path = jn(base_path, 'outputs', 'quantitative')
os.makedirs(args.quantitative_save_path, exist_ok=True)


train_data_ted, eval_data_ted, test_data_ted = loader.load_ted_db_data(data_path, s2ag_config_args)
# print(train_data_ted.n_poses, train_data_ted.n_samples)
# exit()
data_loader = dict(train_data_s2ag=train_data_ted, eval_data_s2ag=eval_data_ted, test_data_s2ag=test_data_ted)
pose_dim = 27
coords = 3
audio_sr = 16000


# Load main processor and models
pr = processor.Processor(args, s2ag_config_args, data_loader, pose_dim, coords, audio_sr)
pr.mean_dir_vec = np.squeeze(np.array(pr.s2ag_config_args.mean_dir_vec))
pr.device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
loaded_vars = torch.load(jn('models', 's2ag_v2_mfcc_run_3', 'ted_db', "epoch_290_loss_-0.0048_model.pth.tar"))
pr.s2ag_generator.load_state_dict(loaded_vars['gen_model_dict'])

# trimodal_checkpoint = torch.load(jn('outputs', 'trimodal_gen.pth.tar'))
# pr.trimodal_generator.load_state_dict(trimodal_checkpoint['trimodal_gen_dict'])

# pr.trimodal_generator.to(pr.device)
pr.s2ag_generator.to(pr.device)

# pr.trimodal_generator.eval()
pr.s2ag_generator.eval()


# Load data from preprocessed lmdb_test
data_params = {'env_file': jn('ted_db/lmdb_test'),
               'clip_duration_range': [5, 30],
               'audio_sr': 16000}


# Video parameters
check_duration = False
fade_out = False
make_video = True
save_pkl = True


# Prepare data and inference
lmdb_env = lmdb.open(data_params['env_file'], readonly=True, lock=False)
with lmdb_env.begin(write=False) as txn:
    keys = [key for key, _ in txn.cursor()]
    print('Total samples to generate: {}'.format(len(keys)))
    for sample_idx in range(len(keys)):
        key = keys[sample_idx]
        buf = txn.get(key)
        video = pyarrow.deserialize(buf)
        vid_name = video['vid']
        clips = video['clips']
        n_clips = len(clips)
        if n_clips == 0:
            continue

        for clip_idx in range(n_clips):
            if clip_idx != 4:
                continue
            speaker_vid_idx = 158

            clip_poses = clips[clip_idx]['skeletons_3d']
            clip_audio = clips[clip_idx]['audio_raw']
            clip_words = clips[clip_idx]['words']
            clip_time = [clips[clip_idx]['start_time'], clips[clip_idx]['end_time']]

            sentence = " ".join([ll[0] for ll in clip_words])
            print(sample_idx, clip_idx, sentence)

            # Prepare seed seq from clip_poses
            clip_poses_resampled = resample_pose_seq(clip_poses, clip_time[1] - clip_time[0], pr.s2ag_config_args.motion_resampling_framerate)
            target_dir_vec = convert_pose_seq_to_dir_vec(clip_poses_resampled)
            target_dir_vec = target_dir_vec.reshape(target_dir_vec.shape[0], -1)
            target_dir_vec -= pr.mean_dir_vec
            seed_seq = target_dir_vec[0:pr.s2ag_config_args.n_pre_poses]

            # render the s2ag
            pr.render_one_clip(data_params, vid_name, sample_idx,
                            len(keys), seed_seq, clip_audio,
                            data_params['audio_sr'], clip_words, clip_time, unit_time=None,
                            clip_idx=clip_idx, speaker_vid_idx=speaker_vid_idx,
                            check_duration=check_duration, fade_out=fade_out,
                            make_video=make_video, save_pkl=save_pkl)

            # render all
            # pr.render_clip(data_params, vid_name, sample_idx,
            #                 len(keys), clip_poses, clip_audio,
            #                 data_params['audio_sr'], clip_words, clip_time, unit_time=None,
            #                 clip_idx=clip_idx, speaker_vid_idx=speaker_vid_idx,
            #                 check_duration=check_duration, fade_out=fade_out,
            #                 make_video=make_video, save_pkl=save_pkl)
            exit()

