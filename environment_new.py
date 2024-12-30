import numpy as np
import pandas as pd
import math
import random as lstRandom

m = 177
n = 104
step_size_real = 0.001 # each cell in grid = 0.001 lat = 100 m
lat_start = 50.4046
lon_start = -104.704

stops_df = pd.read_csv(r'C:\Users\saeidit\Documents\PhD\Research\Data\Regina\hier_cluster_unique_active.csv')
stops = []
for index, item in stops_df[['LAT','LON']].iterrows():
    row = n - int((item['LAT']-lat_start)/step_size_real)-1
    col = int((item['LON']-lon_start)/step_size_real)
    stops.append((row,col))
df = pd.DataFrame(stops, columns = ['row' , 'col'])
df['Total Boarding']= stops_df['Total Boarding']
df['Total Alighting']= stops_df['Total Alighting']

walls_df = pd.read_csv(r'C:\Users\saeidit\Documents\PhD\Research\Data\Regina\hier_cluster_unique.csv')
walls = []
for index, item in walls_df[['LAT','LON']].iterrows():
    row = n - int((item['LAT']-lat_start)/step_size_real)-1
    col = int((item['LON']-lon_start)/step_size_real)
    walls.append((row,col))


################################################################################
class GridWorld(object):
    def __init__(self, step_size, agent_size, stops, df, limit_size, alfa, beta, walls): 
        self.step_size = step_size
        self.agent_size = agent_size 
        self.action_space = {0: (0,0), 1: (1,0) , 2:(-1,0) , 3: (0,1) , 4:(0,-1)}
        self.stops = stops
        self.walls = walls
        self.df = df
        self.limit_size = limit_size
        self.alfa = alfa
        self.beta = beta
        self.env_cells =[]
        for r in range(-20, n+10):
            for c in range(-20, m+10):
                if not self.offGridMove(r, c):
                    self.env_cells.append(tuple((r,c)))
        self.size = len(self.env_cells)
        self.cell_df = pd.DataFrame(self.env_cells, columns=['row','col'])
        self.cell_df['index'] = self.cell_df.index
    
    
    def obs_to_one_hot(self, lst):
        one_hot = np.zeros(self.size)
        for row, col in lst:
            selected_rows = self.cell_df.loc[(self.cell_df['row'] == row) & (self.cell_df['col'] == col)]
            i = selected_rows['index'].values[0]
            one_hot[i] = 1
        return one_hot
    
    
    def dist(self, p1 , p2):
        dist = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
        return dist    

    def offGridMove(self, row_, col_):
        offgrid_lst = []
        #if (row_ > (m -1)) or (row_ < 0) or (col_ > (self.n -1)) or (col_ <0):
        for s in self.walls:
            
            if self.dist(tuple((row_, col_)), s) > self.limit_size:
                offgrid_lst.append(0)
            else:
                offgrid_lst.append(1)
        if sum(offgrid_lst) > 0 :
        #and row_<n and row_>-1 and col_<m and col>-1:
            return False
        else:
            return True

    
    def Reward(self, obs):  
        reward_list=[]            
        # Assignment of stops to closest cluster centroid            
        counter=0
        for stop in self.stops:
            dist_list=[]
            for a in range(self.agent_size):
                dist_list.append(self.dist(stop, obs[a]))
                
            clust_ID = np.argmin(dist_list)
            self.df.at[counter, 'cluster'] = clust_ID
            counter +=1
        inertia_lst = []
        for a in self.df['cluster'].unique():
            inertia = 0
            stops_in_cluster = self.df[self.df['cluster']==a]
            stops_in_cluster.reset_index(drop = True, inplace = True)
            real_centroid = tuple((stops_in_cluster['row'].mean() , stops_in_cluster['col'].mean()))
            for st in range(len(stops_in_cluster)):
                stopp = tuple((stops_in_cluster.at[st , 'row'] , stops_in_cluster.at[st , 'col']))
                inertia += (self.dist(stopp, real_centroid))**2
            RMSE = (inertia/len(stops_in_cluster))**0.5
            inertia_lst.append(RMSE)
            size_factor = (2 - (2/(1+(1/(math.e)**(RMSE/self.alfa)))))**self.beta
            B = stops_in_cluster['Total Boarding'].sum()
            A = stops_in_cluster['Total Alighting'].sum()
            sym = np.minimum(B/(A+0.000001) , A/(B+0.000001))
            reward_list.append(sym * size_factor)

        state_value = sum(reward_list) / len(reward_list)
        return state_value , self.df 

    
    def step(self, sorted_obs, action, ID): # should output a vector of agent's state
        obs_ = sorted_obs
        row , col = sorted_obs[ID]
        row_ = row + (self.action_space[action][1] * self.step_size)
        col_ = col + (self.action_space[action][0] * self.step_size) 
        if not self.offGridMove(row_, col_):
            obs_[ID] = (row_ , col_)
        reward , outcome_df = self.Reward(obs_)
        return obs_, reward, outcome_df
    

    def reset(self):
        obs = lstRandom.sample(self.stops , self.agent_size)
        one_hot = self.obs_to_one_hot(obs)
        return obs , one_hot
