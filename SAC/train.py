# Copyright 2020
# Author: Christian Leininger <info2016frei@gmail.com>
 
import os
import sys 
import cv2
import gym
import time
import torch 
import random
import numpy as np
from collections import deque
from datetime import datetime
from replay_buffer import ReplayBuffer
from agent import SAC
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from gym_grasping.envs.robot_sim_env import RobotSimEnv



def create_next_obs(state, size, args, state_buffer, policy, debug=False):
    state =  cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = cv2.resize(state,(size, size))
    #frame = cv2.imshow("wi", state)
    #cv2.waitKey(10)
    state = torch.tensor(state, dtype=torch.int8, device=args.device)
    state_buffer.append(state)
    state = torch.stack(list(state_buffer), 0)
    state = state.cpu()
    obs = np.array(state)
    #print("obs", obs.shape)
    #frame = cv2.imshow("wi", obs[0])
    #cv2.waitKey(10)
    return obs, state_buffer 


def stacked_frames(state, size, args, policy, debug=False):
#def stacked_frames(state, size, args, policy, debug=True):
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

def mkdir(base, name):
    """
    Creates a direction if its not exist
    Args:
       param1(string): base first part of pathname
       param2(string): name second part of pathname
    Return: pathname 
    """
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def evaluate_policy(policy, writer, total_timesteps, args, env, episode=5):
    """
    
    
    Args:
       param1(): policy
       param2(): writer
       param3(): episode default 1 number for path to save the video
    """
    path = args.locexp
    mkdir("", str(path) + "/" + str(total_timesteps) + "/")
    size = args.size
    avg_reward = 0.
    seeds = [x for x in range(episode)]
    goal= 0
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        env.seed(s)
        obs = env.reset()
        done = False
        obs, state_buffer = stacked_frames(obs, size, args, policy)
        for step in range(args.max_episode_steps):
            action = policy.select_action(np.array(obs))
            
            obs, reward, done, _ = env.step(action)
            #cv2.imshow("wi", cv2.resize(obs[:,:,::-1], (300,300)))
            #cv2.waitKey(10)
            frame = cv2.imwrite("{}/{}/wi{}.png".format(path, total_timesteps, step), np.array(obs))
            if done:
                avg_reward += reward * args.reward_scalling
                if step < 49:
                    goal +=1
                break
            obs, state_buffer = create_next_obs(obs, size, args, state_buffer, policy)
            avg_reward += reward * args.reward_scalling

    avg_reward /= len(seeds)
    writer.add_scalar('Evaluation reward', avg_reward, total_timesteps)
    text = "Totaltimesteps: {}  average reward {}  goals: {} ".format(total_timesteps, avg_reward, goal)
    write_into_file(str(args.locexp) + "/eval", text)
    print ("---------------------------------------")
    print (text)
    print ("---------------------------------------")
    return avg_reward




def write_into_file(pathname, text):
    """
    """
    with open(pathname+".txt", "a") as myfile:
        myfile.write(text)
        myfile.write('\n')

def time_format(sec):
    """
    
    Args:
        param1():

    """
    hours = sec // 3600
    rem = sec - hours * 3600
    mins = rem // 60
    secs = rem - mins * 60
    return hours, mins, round(secs,2)



def train_agent(args):
    """

    Args:
    """
    
    # create CNN convert the [1,3,84,84] to [1, 200]
    
    now = datetime.now()    
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pathname = str(args.locexp) + "/" + str(args.env_name) + '_agent_' + str(args.policy)
    pathname += "_batch_size_" + str(args.batch_size) + "_lr_act_" + str(args.lr_actor) 
    pathname += "_lr_critc_" + str(args.lr_critic) + "_lr_decoder_"
    arg_text = str(args)
    write_into_file(pathname, arg_text) 
    tensorboard_name = str(args.locexp) + '/runs/' + pathname 
    writer = SummaryWriter(tensorboard_name)
    size = args.size
    env= gym.make(args.env_name, renderer='egl')
    state = env.reset()
    print("state ", state.shape)
    state_dim = 200
    print("State dim, " , state_dim)
    action_dim = 5 
    print("action_dim ", action_dim)
    max_action = 1
    args.target_entropy=-np.prod(action_dim)
    args.max_episode_steps = 200
    file_name = str(args.locexp) + "/pytorch_models/{}".format(args.env_name)
    obs_shape = (args.history_length, size, size)
    action_shape = (action_dim,)
    print("obs", obs_shape)
    print("act", action_shape)
    policy = TQC(state_dim, action_dim, max_action, args)    
    replay_buffer = ReplayBuffer(obs_shape, action_shape, int(args.buffer_size), args.image_pad, args.device)
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    t0 = time.time()
    scores_window = deque(maxlen=100) 
    episode_reward = 0
    evaluations = []
    tb_update_counter = 0
    # TODO: evaluate 
    evaluations.append(evaluate_policy(policy, writer, total_timesteps, args, env))
    save_model = file_name + '-{}reward_{:.2f}-agent{}'.format(episode_num, evaluations[-1], args.policy) 
    policy.save(save_model)
    done_counter =  deque(maxlen=100)
    while total_timesteps <  args.max_timesteps:
        tb_update_counter += 1
        # If the episode is done
        if done:
            episode_num += 1
            #env.seed(random.randint(0, 100))
            scores_window.append(episode_reward)
            average_mean = np.mean(scores_window)
            if tb_update_counter > args.tensorboard_freq:
                print("Write tensorboard")
                tb_update_counter = 0
                writer.add_scalar('Reward', episode_reward, total_timesteps)
                writer.add_scalar('Reward mean ', average_mean, total_timesteps)
                writer.flush()
            # If we are not at the very beginning, we start the training process of the model
            if total_timesteps != 0:
                if episode_timesteps < 50:
                    done_counter.append(1)
                else:
                    done_counter.append(0)
                goals = sum(done_counter)
                text = "Total Timesteps: {} Episode Num: {} ".format(total_timesteps, episode_num) 
                text += "Episode steps {} ".format(episode_timesteps)
                text += "Goal last 100 ep : {} ".format(goals)
                text += "Reward: {:.2f}  Average Re: {:.2f} Time: {}".format(episode_reward, np.mean(scores_window), time_format(time.time()-t0))
                writer.add_scalar('Goal_freq', goals, total_timesteps)
                
                print(text)
                write_into_file(pathname, text)
                #policy.train(replay_buffer, writer, episode_timesteps)
            # We evaluate the episode and we save the policy
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq 
                evaluations.append(evaluate_policy(policy, writer, total_timesteps, args,  env))
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)
                evaluations.append(evaluate_policy(policy, writer, total_timesteps, args, env))
                save_model = file_name + '-{}reward_{:.2f}-agent{}'.format(episode_num, evaluations[-1], args.policy) 
                policy.save(save_model)
            # When the training step is done, we reset the state of the environment
            state = env.reset()
            obs, state_buffer = stacked_frames(state, size, args, policy)

            # Set the Done to False
            done = False
            # Set rewards and episode timesteps to zero
            episode_reward = 0
            episode_timesteps = 0
            # reset k in memory
            replay_buffer.k = 0
        # Before 10000 timesteps, we play random actions
        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample()
        else: # After 10000 timesteps, we switch to the model
            action = policy.select_action(obs)
        # The agent performs the action in the environment, then reaches the next state and receives the reward
        new_obs, reward, done, _ = env.step(action)
        # print(reward)
        #frame = cv2.imshow("wi", np.array(new_obs))
        #cv2.waitKey(10)
        done = float(done)
        new_obs, state_buffer = create_next_obs(new_obs, size, args, state_buffer, policy)
        
        # We check if the episode is done
        #done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
        done_bool = 0 if episode_timesteps + 1 == args.max_episode_steps else float(done)
        if episode_timesteps + 1 == args.max_episode_steps:
            done = True
        # We increase the total reward
        reward = reward * args.reward_scalling
        episode_reward += reward
        # We store the new transition into the Experience Replay memory (ReplayBuffer)
        if args.debug:
            print("add to buffer next_obs ", obs.shape)
            print("add to bufferobs ", new_obs.shape)
        replay_buffer.add(obs, action, reward, new_obs, done, done_bool)
        # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
        obs = new_obs
        if total_timesteps > args.start_timesteps:
            for i in range(args.repeat_update):
                policy.train(replay_buffer, writer, 1)
        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1


    # We add the last policy evaluation to our list of evaluations and we save our model
    evaluations.append(evaluate_policy(policy, writer, total_timesteps, args, episode_num))
