import numpy as np
import torch as T
from deep_q_network_new import DeepQNetwork
from replay_memory_new import ReplayBuffer
from environment_new import GridWorld, df, stops, n, walls
from numpy import random

step_size_real = 0.001 # each cell in grid = 0.001 lat = 100 m
limit_size = 5  # 500 m distance of stops to centroid is considered in environment definition
step_size = 1
lat_start = 50.4046
lon_start = -104.704
agent_size = 25
alfa=10
beta=0.8

env = GridWorld(step_size, agent_size, stops, df, limit_size, alfa, beta, walls)

class DQNAgent(object):
    def __init__(self, agent_size, gamma, epsilon, lr, input_dims, n_actions, mem_size, 
                 eps_min, batch_size, eps_dec, chkpt_dir):
        self.agent_size = agent_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.mem_size = mem_size
        self.min_mem_size = 1000 #500
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.chkpt_dir = chkpt_dir
        self.losses = []
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0
        self.replace_target_cnt = 50 #1000
        self.loss = 0
        self.q_eval = DeepQNetwork(self.lr, self.n_actions, self.input_dims, 
                                   chkpt_dir=self.chkpt_dir, name='tryal_q_eval_big'+str(agent_size))
        self.q_next = DeepQNetwork(self.lr, self.n_actions, self.input_dims, 
                                   chkpt_dir=self.chkpt_dir, name='tryal_q_next_big'+str(agent_size))
        self.memory = ReplayBuffer(self.mem_size, self.input_dims, self.n_actions)

    def matrix_to_cell(self, n, row, column):
        return (row - 1) * n + column

    def sort_matrix_cells(self, n, row_col_pairs):
        sorted_pairs = sorted(row_col_pairs, key=lambda x: self.matrix_to_cell(n, x[0], x[1]))
        return sorted_pairs


    def choose_action(self, obs, one_hot):  #epsilon greedy algorithm
        lst_action = []
        lst_reward = []
        lst_states = []
        sorted_obs = self.sort_matrix_cells(n, obs)
        for a in range(self.agent_size):
            ID_feature = np.zeros(self.agent_size)
            ID_feature[a]=1
            #new = []
            #s = obs.copy()
            #new.append(s[a])
            #del s[a]
            #for b in s:
                #new.append(b)     
            new = np.concatenate((one_hot, ID_feature))
            input_state_tens = T.tensor(np.array(new).reshape(1, self.input_dims),dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(input_state_tens)         
            action = T.argmax(actions).item()
            reward= env.step(sorted_obs, action, a)[1]
            lst_action.append(action)
            lst_reward.append(reward)
            lst_states.append(new)
        if np.random.random() > self.epsilon:
            ID = np.argmax(np.array(lst_reward))
            chosen_action = lst_action[ID]
        else:
            #print('random')
            chosen_action = np.random.choice(self.action_space)
            ID = random.choice(self.agent_size)
        selected_state = lst_states [ID]
        return chosen_action, selected_state, sorted_obs, ID

    def store_transition(self, obs, chosen_action, reward, obs_):
        self.memory.store_transition(obs, chosen_action, reward, obs_)
#-------------------------------------------------------------------------------
    def sample_memory(self):
        state, action, reward, new_state = self.memory.sample_buffer(self.batch_size)
        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)        
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)
        #dones = T.tensor(done).to(self.q_eval.device)
        return states, actions, rewards, states_ 
    
    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min
    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()
#--------------------------------------------------------------------------------
    def learn(self):  # double dqn
        if self.memory.mem_cntr < self.min_mem_size: # or self.memory.mem_cntr % 10 !=0:
          return
        else:
          self.q_eval.optimizer.zero_grad()
          self.replace_target_network()
          states, actions, rewards, states_ = self.sample_memory()
          indices = np.arange(self.batch_size)
          q_pred = self.q_eval.forward(states)[indices , actions]
          q_next = self.q_next.forward(states_)
          q_eval = self.q_eval.forward(states_)
          if self.learn_step_counter % self.replace_target_cnt == 0:
              print('q_eval',q_eval)      
          max_actions = T.argmax(q_eval, dim=1)
          #q_next[dones.long()] = 1
          q_target = rewards + self.gamma*q_next[indices, max_actions]
          self.loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
          self.loss.backward()
          self.losses.append(self.loss.item())
          self.q_eval.optimizer.step()
          self.learn_step_counter += 1
          self.decrement_epsilon() 
          










