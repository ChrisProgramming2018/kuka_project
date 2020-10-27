import numpy as np
import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ColorJitter
from torchvision import transforms
import cv2
from collections import deque
import sys


print(transforms.__path__)


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, image_pad, device):
        self.capacity = capacity
        self.device = device
        self.obses = np.empty((capacity, *obs_shape), dtype=np.int8)
        self.obses_aug = np.empty((capacity, *obs_shape), dtype=np.int8)
        self.idx = 0
        self.full = False
        self.k = 0

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, obs_aug):
        self.k +=1
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.obses_aug[self.idx], obs_aug)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0




    def sample(self, batch_size):
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size)    
        obs = self.obses[idxs]
        obs_aug = self.obses_aug[idxs]

        obses = torch.as_tensor(obs, device=self.device).float()
        obses_aug = torch.as_tensor(obs_aug, device=self.device).float() 
        
        return obses, obses_aug



    def save_memory(self, filename):
        """
        Use numpy save function to store the data in a given file
        """


        with open(filename + '/obses.npy', 'wb') as f:
            np.save(f, self.obses)
        
        with open(filename + '/actions.npy', 'wb') as f:
            np.save(f, self.actions)

        with open(filename + '/next_obses.npy', 'wb') as f:
            np.save(f, self.next_obses)
        
        with open(filename + '/rewards.npy', 'wb') as f:
            np.save(f, self.rewards)
        
        with open(filename + '/not_dones.npy', 'wb') as f:
            np.save(f, self.not_dones)
        
        with open(filename + '/not_dones_no_max.npy', 'wb') as f:
            np.save(f, self.not_dones_no_max)
    
    def load_memory(self, filename):
        """
        Use numpy load function to store the data in a given file
        """


        with open(filename + '/obses.npy', 'rb') as f:
            self.obses = np.load(f)
        
        with open(filename + '/actions.npy', 'rb') as f:
            self.actions = np.load(f)

        with open(filename + '/next_obses.npy', 'rb') as f:
            self.next_obses = np.load(f)
        
        with open(filename + '/rewards.npy', 'rb') as f:
            self.rewards = np.load(f)
        
        with open(filename + '/not_dones.npy', 'rb') as f:
            self.not_dones = np.load(f)
        
        with open(filename + '/not_dones_no_max.npy', 'rb') as f:
            self.not_dones_no_max = np.load(f)



class Memory(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, capacity,  device):
        self.capacity = capacity
        self.device = device
        self.obses = np.empty((51, *obs_shape), dtype=np.float32)
        
        self.states = np.empty((capacity, *obs_shape), dtype=np.int8)     # 
        self.states_cj = np.empty((capacity, *obs_shape), dtype=np.int8)  # same state transformed with cj

        self.idx = 0
        self.full = False
        self.k = 0
        self.t = ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0)

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, image):
        self.k +=1
        np.copyto(self.obses[self.k], image)
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0


    def sample(self, batch_size):
        """ get the image from sim and the transformed image

        """
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size)
    
        obses = self.obses[idxs]
        obses = torch.as_tensor(obses, device=self.device).float()
        
        return obses



    def create_states(self, replay_buffer):
        """ Use the images of the last epiosde to create states
            and the pytorch ColorJitter with the same parameters 
        """

        # take 3 images add cj and merge to state
        torch.seed = 0
        brightness = np.random.uniform(0.5, 1.5)
        contrast  =  np.random.uniform(0.5, 1.5)
        for idx in range(self.k - 2):
            image_1 = self.obses[idx] 
            image_2 = self.obses[idx + 1] 
            image_3 = self.obses[idx + 2]
            # create state without cj
            #cv2.imwrite("im1{}.png".format(idx), np.array(image_1))
            #cv2.imwrite("im2{}.png".format(idx), np.array(image_2))
            #cv2.imwrite("im3{}.png".format(idx), np.array(image_3))
            state = self.merge_images(image_1, image_2,image_3)
            img_aug_1 = self.augment_image(image_1, brightness, contrast)
            img_aug_2 = self.augment_image(image_2, brightness, contrast)
            img_aug_3 = self.augment_image(image_3, brightness, contrast)
            #cv2.imwrite("im_aug1{}-{}.png".format(idx,brightness), np.array(img_aug_1))
            #cv2.imwrite("im_aug2{}-{}.png".format(idx, brightness), np.array(img_aug_2))
            #cv2.imwrite("im_aug3{}-{}.png".format(idx, brightness), np.array(img_aug_3))
            state_aug = self.merge_images(img_aug_1, img_aug_2,img_aug_3)
            replay_buffer.add(state, state_aug)
        self.k = 0
        return replay_buffer
            




        # reset counter for episode length    
        self.k = 0





    def merge_images(self, img1, img2, img3):
        """ Create an gray imgae and stack them
        """
        state_buffer = deque([], maxlen=3)
        # print("img1 shape", img1.shape)
        state1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        state2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        state3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        state1 = torch.tensor(state1, dtype=torch.int8, device=self.device)
        state2 = torch.tensor(state2, dtype=torch.int8, device=self.device)
        state3 = torch.tensor(state3, dtype=torch.int8, device=self.device)
        state_buffer.append(state1)
        state_buffer.append(state2)
        state_buffer.append(state3)
        state = torch.stack(list(state_buffer), 0)
        state = state.cpu()
        obs = np.array(state)
        return state


    def augment_image(self, state, brightness, contrast):
        """
        set brightness between  [0.5, 1.5]
        set contrast between [0.5, 1.5]
        """
        im_a = transforms.ToPILImage()(state.astype(np.uint8))
        self.t.brightness = [brightness, brightness]
        aug_img = self.t(im_a)
        array_aug = transforms.ToTensor()(aug_img)
        array_aug = array_aug * 255
        array_aug = np.transpose(array_aug,(1,2,0))
        return np.array(array_aug)

