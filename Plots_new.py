import numpy as np
from Agent_new import DQNAgent, env, stops, agent_size
from environment_new import stops_df
import matplotlib.pyplot as plt
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
import pandas as pd
import seaborn as sns
import networkx as nx
import pickle
from scipy.spatial import ConvexHull
import ast


with open('data_frames.pickle', 'rb') as file:
    list_clustrs = pickle.load(file)
    

dataframe_test = pd.read_csv(r'C:/Users/saeidit/Documents/PhD/Research/Data/Regina/cluster output/test_gain_score.csv')
dataframe_test2 = pd.read_csv(r'C:/Users/saeidit/Documents/PhD/Research/Data/Regina/cluster output/best_cluster_ever_test.csv')
dataframe_train = pd.read_csv(r'C:/Users/saeidit/Documents/PhD/Research/Data/Regina/cluster output/training_output.csv')


loss_lst = dataframe_train['loss function during all iterations']
reward_lst = dataframe_train['mean rewards in all games']
game_best_reward = dataframe_test['best rewards each game']
initial_reward_list = dataframe_test['initial rewards of games']
labels_best_cluster = dataframe_test2['Best cluster']
n_games = 2000

############################     chart for loss function  #####################

x=list(range(1 , len(loss_lst)))
fig = plt.figure()
fig, ax = plt.subplots()
ax.plot(x, loss_lst)
ax.set_xlabel('Time Steps')
ax.set_ylabel('Loss Function')
ax.legend()


############     chart for average reward of games (moving average) #########
x=list(range(1 , len(reward_lst)))
fig = plt.figure()
fig, ax = plt.subplots()
y=[]
for i in range(1,len(reward_lst)):
    y.append(sum(game_best_reward[max(0,i-50):i+1])/len(game_best_reward[max(0,i-50):i+1]))

ax.plot(x, y)
ax.set_xlabel('Games')
ax.set_ylabel('Average of Last 50 Rewards')
ax.legend()


####################   best cluster    #########################
points = stops_df[['LAT','LON']]
points['cluster'] =ast.literal_eval(labels_best_cluster[0])
centroids = ast.literal_eval(dataframe_test2['centroids'][0])
sns.lmplot(x='LON', y='LAT', data=points, fit_reg=False, legend=False, hue='cluster')
#plt.scatter(stops_df[['LON']],stops_df[['LAT']])
plt.xticks(rotation=45)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(loc='upper center', bbox_to_anchor=(0.4, -0.2), ncol=6)


###############   Association Matrix   ####################
Associate_mat = np.zeros((len(stops),len(stops)))
for i in range(len(stops)):
    for j in range( i+1 ,len(stops), 1):
        count=0
        for r in range(n_games):
            if list_clustrs[r][i]==list_clustrs[r][j]:
                count +=1
        Associate_mat[i,j]=count
        Associate_mat[j,i]=count

##############   VISUALIZATION of Initial reward improvements  #####################
x,y=[],[]
for j in range(100):
    j = j/100
    x.append(j)
    count = 0 
    for i in ast.literal_eval(game_best_reward[0]):
        if i>j:
            count +=1
    y.append(count/n_games)  #shows how many games scored morre than particular value
x_,y_=[],[]
for j in range(100):
    j = j/100
    x_.append(j)
    count = 0 
    for i in ast.literal_eval(initial_reward_list[0]):
        if i>j:
            count +=1
    y_.append(count/n_games)  #shows how many games scored morre than particular value
plt.plot(x_,y_,label='Initial Random Reward')
plt.plot(x,y,label='Achieved Reward')
plt.xlabel('Reward')
plt.ylabel('Probability of achieving higher reward')
plt.legend()
plt.show()


##############    Visualization of heat map   ###################
# create dictionary of nodes, edges, and coordinates
dict_nodes, dict_edges, dict_coordinates = {},{},{}
node_list=[]
# dictionary of nodes based on normalized boarding proportion
Board_proportions = stops_df['Total Boarding'] #/(stops_df['Total Boarding'].sum())
values = Board_proportions.tolist()
for i in range(len(stops)):
    dict_nodes[i] = values[i]
    node_list.append(values[i]*5)
    #node_list.append(0)

list_values, edge_list=[],[]
for i in range(len(stops)):
    item_dict={}
    for j in range(len(stops)):
        if Associate_mat[i,j]>800:
            item_dict[j] = Associate_mat[i,j]/500
            edge_list.append(Associate_mat[i,j]/500)
    list_values.append(item_dict)

for i in range(len(stops)):
    dict_edges[i]=list_values[i]


coordinates = stops_df[['LON','LAT']].to_records(index=False).tolist()
for i in range(len(stops)):
    dict_coordinates[i] = coordinates[i]

#plot
plt.subplots(figsize=(14,14))

#networkx graph time!
G = nx.Graph()

for node in dict_nodes.keys():
    G.add_node(node, size = dict_nodes[node])

for i in dict_edges.keys():
    for j in dict_edges[i].keys():
        G.add_edge(i, j, weight = dict_edges[i][j])

nx.draw(G, pos=dict_coordinates, with_labels=False, font_size = 8,
        font_weight = 'bold', node_size = np.array(node_list), width = edge_list)






