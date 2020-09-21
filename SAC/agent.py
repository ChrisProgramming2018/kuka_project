import os
import sys
import torch
import copy
from models import QNetwork, Eecoder 
import torch.nn as nn
import torch.nn.functional as F



# Building the whole Training Process into a class

class SAC(object):
    def __init__(self, state_dim, action_dim, actor_input_dim, args):
        input_dim = [args.history_length, args.size, args.size]
        self.policy = GaussianPolicy(state_dim, action_dim, args).to(args.device)
        self.policy_optimizer = torch.optim.Adam(self.actor.parameters(), args.lr_actor)        
        self.critic = QNetwork(state_dim, action_dim, args).to(args.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), args.lr_critic)
        self.target_critic = QNetwork(state_dim, action_dim, args).to(args.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.encoder = Encoder(args).to(args.device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), args.lr_encoder)
        self.target_Encoder = Encoder(args).to(args.device)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.batch_size = args.batch_size
        self.discount = args.discount
        self.tau = args.tau 
        self.device = args.device
        self.write_tensorboard = False
        self.target_entropy = args.target_entropy 
        self.quantiles_total = self.critic.n_quantiles * self.critic.n_nets
        self.log_alpha = torch.zeros((1,), requires_grad=True, device=args.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.lr_alpha)
        self.total_it = 0
        self.step = 0

    
    
    def train(self, replay_buffer, writer, iterations):
        self.step += 1
        if self.step % 1000 == 0:
            self.write_tensorboard = 1 - self.write_tensorboard
        for it in range(iterations):
            # Step 4: We sample a batch of transitions (s, s’, a, r) from the memoy
            sys.stdout = open(os.devnull, "w")
            obs, action, reward, next_obs, not_done, obs_aug, obs_next_aug = replay_buffer.sample(self.batch_size)
            sys.stdout = sys.__stdout__
            # for augment 1
            obs = obs.div_(255)
            next_obs = next_obs.div_(255)
            state = self.encoder.create_vector(obs)
            detach_state = state.detach()
            next_state = self.target_encoder.create_vector(next_obs)
            # for augment 2
            
            obs_aug = obs_aug.div_(255)
            next_obs_aug = obs_next_aug.div_(255)
            state_aug = self.decoder.create_vector(obs_aug)
            detach_state_aug = state_aug.detach()
            next_state_aug = self.target_decoder.create_vector(next_obs_aug)
            
            alpha = torch.exp(self.log_alpha)
            with torch.no_grad(): 
                # Step 5: Get policy action
                
                next_state_action, next_state_log_pi, _ = self.policy.sample(next_state)
                qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = reward_batch + not_done * self.discount * (min_qf_next_target)

                         
            
            #---update critic
            Q1, Q2 = self.critic(state, action)
            qf1_loss = F.mse_loss(qf1, next_q_value)  
            qf2_loss = F.mse_loss(qf2, next_q_value)  
            qf_loss = qf1_loss + qf2_loss 
            self.critic_optimizer.zero_grad()
            self.encoder_optimizer.zero_grad()
            critic_loss.backward()
            self.encoder_optimizer.step()
            self.critic_optimizer.step()
        
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            #---Update policy and alpha
            pi, log_pi, _ = self.policy.sample(detach_state)
            qf1_pi, qf2_pi = self.critic(state_batch, pi) 
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            policy_loss = ((alpha * log_pi) - min_qf_pi).mean() 
            
            # ---alpha 
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha = self.log_alpha.exp()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
                
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        
        torch.save(self.decoder.state_dict(), filename + "_decoder")
        torch.save(self.decoder_optimizer.state_dict(), filename + "_decoder_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor) 
        
        self.decoder.load_state_dict(torch.load(filename + "_decoder"))
        self.decoder_optimizer.load_state_dict(torch.load(filename + "_decoder_optimizer"))
