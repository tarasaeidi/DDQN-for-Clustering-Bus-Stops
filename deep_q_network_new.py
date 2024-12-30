import os
import torch as T
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

class DeepQNetwork(nn.Module):    
    def __init__(self, lr, n_actions, input_dims, chkpt_dir, name):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        #os.mkdir(self.checkpoint_file)
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.fc1 = nn.Linear(self.input_dims, 128)
        #self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        #self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 512)
        #self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 128)
        #self.bn4 = nn.BatchNorm1d(128)
        #self.fc5 = nn.Linear(512, 128)
        self.fc6 = nn.Linear(128, self.n_actions)
        # Initialize weights uniformly
        init.uniform_(self.fc1.weight, a=-0.1, b=0.1)
        init.uniform_(self.fc2.weight, a=-0.1, b=0.1)
        init.uniform_(self.fc3.weight, a=-0.1, b=0.1)
        init.uniform_(self.fc4.weight, a=-0.1, b=0.1)
        #init.uniform_(self.fc5.weight, a=-0.1, b=0.1)
        init.uniform_(self.fc6.weight, a=-0.1, b=0.1)
        self.optimizer = optim.RMSprop(self.parameters(), lr=self.lr)   # gradient decent to minimize loss function
        self.loss = nn.MSELoss()
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state.float()))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc4(x))
        x = F.dropout(x, p=0.5, training=self.training)
        #x = F.relu(self.fc5(x))
        x = self.fc6(x)
        actions = F.softmax(x,dim=1)
        #print('actions',actions)
        return actions

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
        #self.eval()
