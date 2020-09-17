# Copyright 2020
# Author: Christian Leininger <info2016frei@gmail.com>

import os
import sys

import time
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
from datetime import datetime
from train import train_agent


def main(arg):
    """ Starts different tests
    Args:
        param1(args): args
    """
    path = arg.locexp
    # experiment_name = args.experiment_name
    res_path = os.path.join(path, "results")
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    dir_model = os.path.join(path, "pytorch_models")
    if arg.save_model and not os.path.exists(dir_model):
        os.makedirs(dir_model)
    print("Created model dir {} ".format(dir_model))
    train_agent(arg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="kuka_block_grasping-v0", type=str, help='Name of a environment (set it to any Continous environment you want')
    parser.add_argument('--seed', default=2, type=int, help='Random seed')
    parser.add_argument('--start_timesteps', default=100, type=int)
    parser.add_argument('--eval_freq', default=1e4, type=int)  # How often the evaluation step is performed (after how many timesteps)
    parser.add_argument('--max_timesteps', default=7e6, type=int)               # Total number of iterations/timesteps
    parser.add_argument('--buffer_size', default=1e5, type=int)               # 
    parser.add_argument('--save_model', default=True, type=bool)     # Boolean checker whether or not to save the pre-trained model
    parser.add_argument('--lr_alpha', default=3e-4, type=float)
    parser.add_argument('--lr_actor', default=1e-4, type=float)      # Exploration noise - STD value of exploration Gaussian noise
    parser.add_argument('--lr_critic', default=1e-4, type=float)      # Exploration noise - STD value of exploration Gaussian noise
    parser.add_argument('--lr_decoder', default=1e-4, type=float)      # Divide by 5
    parser.add_argument('--batch_size', default= 512, type=int)      # Size of the batch
    parser.add_argument('--discount', default=0.99, type=float)      # Discount factor gamma, used in the calculation of the total discounted reward
    parser.add_argument('--tau', default=0.005, type= float)        # Target network update rate
    parser.add_argument("--n_quantiles", default=25, type=int)
    parser.add_argument("--top_quantiles_to_drop_per_net", default=2, type=int)
    parser.add_argument("--n_nets", default=5, type=int)
    parser.add_argument('--tensorboard_freq', default=500, type=int)    # every nth episode write in to tensorboard
    parser.add_argument('--device', default='cuda', type=str)    # amount of qtarget nets
    parser.add_argument('--reward_scalling', default=1, type=int)    # amount of qtarget nets
    parser.add_argument('--max_episode_steps', default=50, type=int)    # amount of qtarget nets
    parser.add_argument('--history_length', default=3, type=int)     # Maximum value of the Gaussian noise added to the actions (policy)
    parser.add_argument('--image_pad', default=4, type=int)     # Maximum value of the Gaussian noise added to the actions (policy)
    parser.add_argument('--size', default=84, type=int, help="size of the image of the simulator")     #  Idx for the replay buffer for the samples of the actor update 
    parser.add_argument('--actor_clip_gradient', default=1., type=float)     # Maximum value of the Gaussian noise added to the actions (policy)
    parser.add_argument('--locexp', type=str)     # Maximum value of the Gaussian noise added to the actions (policy)
    parser.add_argument('--debug', default=False, type=bool)     # Maximum value of the Gaussian noise added to the actions (policy)
    parser.add_argument('--policy', default="TCQ", type=str)
    parser.add_argument('--repeat_update', default=1, type=int)         # Number of iterations to wait before the policy network (Actor model) is updated
    parser.add_argument('--update_beta_freq', default=1, type=int) 
    arg = parser.parse_args()
    main(arg)
