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



def create_next_obs(state, size, args, state_buffer, policy, debug=False):
    state =  cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = cv2.resize(state,(size, size))
    state = torch.tensor(state, dtype=torch.int8, device=args.device)
    state_buffer.append(state)
    state = torch.stack(list(state_buffer), 0)
    state = state.cpu()
    obs = np.array(state)
    return obs, state_buffer 


def stacked_frames(state, size, args, policy, debug=False):
    if debug:
        img = Image.fromarray(state, 'RGB')
        img.save('my.png')
        img.show()
    state =  cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = cv2.resize(state,(size, size))
    state = torch.tensor(state, dtype=torch.int8, device=args.device)
    zeros = torch.zeros_like(state)
    state_buffer = deque([], maxlen=args.history_length)
    for idx in range(args.history_length - 1):
        state_buffer.append(zeros)
    state_buffer.append(state)
    state = torch.stack(list(state_buffer), 0)
    state = state.cpu()
    obs = np.array(state)
    return obs, state_buffer

def get_state(state, size, args, perception, debug=False):
    if debug:
        img = Image.fromarray(state, 'RGB')
        img.save('my.png')
        img.show()
        img = Image.fromarray(lum_img, 'L')
        img.save('my_gray.png')
    state =  cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = cv2.resize(state,(size, size))
    state = torch.tensor(state, dtype=torch.int8, device=args.device)
    zeros = torch.zeros_like(state)
    state_buffer = deque([], maxlen=args.history_length)
    for idx in range(args.history_length - 1):
        state_buffer.append(zeros)
    state = torch.stack(list(state_buffer), 0)
    obs = np.array(state)
    return obs, state_buffer




def evaluate_policy(policy,args, env, episode=10):
    """
    
    
    Args:
       param1(): policy
       param2(): writer
       param3(): episode default 1 number for path to save the video
    """
    size = args.size
    avg_reward = 0.
    different_seeds = args.seed
    seeds = [x for x in range(episode)]
    goals  = 0
    for s in seeds:
        if different_seeds:
            torch.manual_seed(s)
            np.random.seed(s)
            env.seed(s)
        obs = env.reset()
        done = False
        obs, state_buffer = stacked_frames(obs, size, args, policy)
        for step in range(args.max_episode_steps):
            #frame = cv2.imshow("wi", np.array(obs[0,:,:]))
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            cv2.imshow("wi", cv2.resize(obs[:,:,::-1], (300,300)))
            cv2.waitKey(10)
            if done:
                if step < 49:
                    print("steps", step)
                    goals +=1
                print("goals ", goals)
                break
            obs, state_buffer = create_next_obs(obs, size, args, state_buffer, policy)
            avg_reward += reward * args.reward_scalling


    avg_reward /= len(seeds)
    print ("---------------------------------------")
    print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
    print ("---------------------------------------")
    return avg_reward




def main(args):
    """ Starts different tests

    Args:
        param1(args): args

    """
    size = 84
    env= gym.make(args.env_name, renderer='egl')
    print(env.action_space)
    state = env.reset()
    state_dim = 200
    action_dim = 5
    max_action = float(1)
    min_action = float(-1)
    args.target_entropy=-np.prod(action_dim)
    width = size 
    height = size
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    fps = 30
    video_filename = 'output.avi'
    video= cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
    
    policy = TQC(state_dim, action_dim, max_action, args) 
    directory = "16.09_beta/pytorch_models/"
    if args.agent is None:
        filename = "kuka_block_grasping-v0-802reward_-3.47-agentTCQ"
        filename = "kuka_block_grasping-v0-10533reward_-1.76-agentTCQ"
        filename= "kuka_block_grasping-v0-6898reward_-1.96-agentTCQ"
    else:
        filename = args.agent

    filename = directory + filename
    print("Load " , filename)
    policy.load(filename)
    policy.actor.training = False
    if args.eval:
        evaluate_policy(policy, args,  env, args.epi)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="kuka_block_grasping-v0", type=str, help='Name of a environment (set it to any Continous environment you want')
    parser.add_argument('--seed', default=True, type=bool, help='use different seed for each episode')
    parser.add_argument('--epi', default=10, type=int)
    parser.add_argument('--max_episode_steps', default=50, type=int)    
    parser.add_argument('--eval_freq', default=10000, type=int)  # How often the evaluation step is performed (after how many timesteps)
    parser.add_argument('--repeat', default=1, type=int)    # every nth episode write in to tensorboard
    parser.add_argument('--max_timesteps', default=2e6, type=int)               # Total number of iterations/timesteps
    parser.add_argument('--lr-critic', default= 0.0005, type=int)               # Total number of iterations/timesteps
    parser.add_argument('--lr-actor', default= 0.0005, type=int)               # Total number of iterations/timesteps
    parser.add_argument('--lr_alpha', default=3e-4, type=float)
    parser.add_argument('--lr_decoder', default=1e-4, type=float)      # Divide by 5
    parser.add_argument('--save_model', default=True, type=bool)     # Boolean checker whether or not to save the pre-trained model
    parser.add_argument('--expl_noise', default=0.1, type=float)      # Exploration noise - STD value of exploration Gaussian noise
    parser.add_argument('--batch_size', default= 256, type=int)      # Size of the batch
    parser.add_argument('--discount', default=0.99, type=float)      # Discount factor gamma, used in the calculation of the total discounted reward
    parser.add_argument('--tau', default=0.005, type= float)        # Target network update rate
    parser.add_argument('--policy_freq', default=2, type=int)         # Number of iterations to wait before the policy network (Actor model) is updated
    parser.add_argument('--size', default=84, type=int)
    parser.add_argument('--target_update_freq', default=50, type=int)
    parser.add_argument('--num_q_target', default=4, type=int)    # amount of qtarget nets
    parser.add_argument('--train_every_step', default=True, type=bool)    # amount of qtarget nets
    parser.add_argument('--tensorboard_freq', default=5000, type=int)    # every nth episode write in to tensorboard
    parser.add_argument('--device', default='cuda', type=str)    # amount of qtarget nets
    parser.add_argument('--run', default=1, type=int)    # every nth episode write in to tensorboard
    parser.add_argument('--agent', default=None, type=str)    # load the weights saved after the given number 
    parser.add_argument('--reward_scalling', default=1, type=int)    # amount
    parser.add_argument("--n_quantiles", default=25, type=int)
    parser.add_argument("--top_quantiles_to_drop_per_net", default=2, type=int)
    parser.add_argument("--n_nets", default=5, type=int)
    parser.add_argument('--history_length', default=3, type=int)
    parser.add_argument('--image_pad', default=4, type=int)     #
    parser.add_argument('--actor_clip_gradient', default=1., type=float)     # Maximum value of the Gaussian noise added to
    parser.add_argument('--locexp', type=str)     # Maximum value
    parser.add_argument('--debug', default=False, type=bool)
    parser.add_argument('--no_render', type=bool)
    parser.add_argument('--eval', type=bool, default= True)
    arg = parser.parse_args()
    main(arg)
