import numpy as np

class ReplayBuffer():
    def __init__(self, mem_size, input_shape, n_actions):
        self.mem_size = mem_size
        self.input_shape = input_shape
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, self.input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, self.input_shape),dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.ID_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        #self.done_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, chosen_action, reward, state_):
        index = self.mem_cntr % self.mem_size
        for i in range(self.input_shape):
            self.state_memory[index][i] = np.array(state).reshape(1,self.input_shape)[0,i]
            self.new_state_memory[index][i] = np.array(state_).reshape(1,self.input_shape,1)[0,i]
        self.action_memory[index] = chosen_action
        self.reward_memory[index] = reward
        #self.done_memory[index] = done
        self.mem_cntr += 1


    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        #dones = self.done_memory[batch]

        return states, actions, rewards, states_