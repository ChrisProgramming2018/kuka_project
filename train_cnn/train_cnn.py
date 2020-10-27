import os
import sys
import gym
import time
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
from agent import TQC
import cv2
from PIL import Image
from gym_grasping.envs.robot_sim_env import RobotSimEnv
from replay_buffer import ReplayBuffer, Memory
from helper import FrameStack, time_format
from torch.utils.tensorboard import SummaryWriter




def train_cnn(env, policy, train_policy, args):
    """
    
    
    Args:
       param1(): policy
       param2(): writer
       param3(): episode default 1 number for path to save the video
    """
    size = args.size
    obs_shape = (args.history_length, size, size)
    action_shape = (args.action_dim,)
    memoy = Memory((84,84,3), int(args.buffer_size), args.device)
    replay_buffer = ReplayBuffer(obs_shape, action_shape, int(args.buffer_size), args.image_pad, args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)
    total_timesteps = 0
    done_counter =  deque(maxlen=100)
    scores_window = deque(maxlen=100)
    t0 = time.time()
    pathname = str(args.locexp) + "/" + str(args.env_name) 
    pathname += "_batch_size_" + str(args.batch_size) + "_lr_encoder_" + str(args.lr_encoder)
    tensorboard_name = str(args.locexp) + '/runs/' + pathname
    writer = SummaryWriter(tensorboard_name)
    for i_episode in range(int(args.episodes)):
        obs = env.reset()
        done = False
        episode_reward = 0
        for step in range(args.max_episode_steps):
            action = policy.select_action(np.array(obs))
            new_obs, reward, done, image = env.step(action)
            memoy.add(image)
            episode_reward += reward
            # frame = cv2.imwrite("im{}.png".format(step), np.array(image))
            done_bool = 0 if step + 1 == args.max_episode_steps else float(done)
            total_timesteps += 1
            obs = new_obs
            if step == 49:
                done = True
            if done:
                memoy.create_states(replay_buffer)
                if total_timesteps != 0:
                    if step < 50:
                        done_counter.append(1)
                    else:
                        done_counter.append(0)
                goals = sum(done_counter)
                scores_window.append(episode_reward)
                text = "Total Timesteps: {} Episode Num: {} ".format(total_timesteps, i_episode)
                text += "Episode steps {} ".format(step)
                text += "Goal last 100 ep : {} ".format(goals)
                text += "Reward: {:.2f}  Average Re: {:.2f} Time: {}".format(episode_reward, np.mean(scores_window), time_format(time.time()-t0))
                print(text)
                break
            if total_timesteps > args.start_opt:
                train_policy.train_cnn(replay_buffer, policy, writer)





def main(args):
    """ Starts different tests

    Args:
        param1(args): args

    """
    env= gym.make(args.env_name, renderer='egl')
    env = FrameStack(env, args)
    state = env.reset()
    state_dim = 200
    action_dim = env.action_space.shape[0]
    args.action_dim = action_dim
    max_action = float(1)
    min_action = float(-1)
    args.target_entropy=-np.prod(action_dim)
    
    policy = TQC(state_dim, action_dim, max_action, args) 
    train_policy = TQC(state_dim, action_dim, max_action, args) 
    directory = "pretrained/"
    if args.agent is None:
        filename = "kuka_block_grasping-v0-97133reward_-1.05-agentTCQ" # 93 %
        filename = "kuka_block_grasping-v0-3509reward_-1.22-agentTCQ" # 93 %
    else:
        filename = args.agent

    filename = directory + filename
    print("Load " , filename)
    policy.load(filename)
    train_policy.load(filename)
    
    train_cnn(env, policy, train_policy, args)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="kuka_block_grasping-v0", type=str, help='Name of a environment (set it to any Continous environment you want')
    parser.add_argument('--seed', default=True, type=bool, help='use different seed for each episode')
    parser.add_argument('--epi', default=25, type=int)
    parser.add_argument('--start_opt', default=50, type=int)
    parser.add_argument('--episodes', default=1e6, type=int)    
    parser.add_argument('--max_episode_steps', default=50, type=int)    
    parser.add_argument('--lr-critic', default= 0.0005, type=int)               # Total number of iterations/timesteps
    parser.add_argument('--lr-actor', default= 0.0005, type=int)               # Total number of iterations/timesteps
    parser.add_argument('--lr_alpha', default=3e-4, type=float)
    parser.add_argument('--lr_encoder', default=1e-4, type=float)      # Divide by 5
    parser.add_argument('--save_model', default=True, type=bool)     # Boolean checker whether or not to save the pre-trained model
    parser.add_argument('--batch_size', default= 25, type=int)      # Size of the batch
    parser.add_argument('--discount', default=0.99, type=float)      # Discount factor gamma, used in the calculation of the total discounted reward
    parser.add_argument('--tau', default=0.005, type= float)        # Target network update rate
    parser.add_argument('--size', default=84, type=int)
    parser.add_argument('--num_q_target', default=4, type=int)    # amount of qtarget nets
    parser.add_argument('--tensorboard_freq', default=5000, type=int)    # every nth episode write in to tensorboard
    parser.add_argument('--device', default='cuda', type=str)    # amount of qtarget nets
    parser.add_argument("--n_quantiles", default=25, type=int)
    parser.add_argument("--top_quantiles_to_drop_per_net", default=2, type=int)
    parser.add_argument("--n_nets", default=5, type=int)
    parser.add_argument('--history_length', default=3, type=int)
    parser.add_argument('--image_pad', default=4, type=int)     #
    parser.add_argument('--actor_clip_gradient', default=1., type=float)     # Maximum value of the Gaussian noise added to
    parser.add_argument('--locexp', type=str)     # Maximum value
    parser.add_argument('--debug', default=False, type=bool)
    parser.add_argument('--eval', type=bool, default= True)
    parser.add_argument('--buffer_size', default=3.5e5, type=int)
    parser.add_argument('--agent', default=None, type=str)
    parser.add_argument('--save_buffer', default=False, type=bool)
    arg = parser.parse_args()
    main(arg)
