import numpy as np
from Agent_new import DQNAgent, env, agent_size
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
import pandas as pd
import pickle
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    load_checkpoint = False
    # smaller learning rate, smaller epsilon decay rate, and smaller target replace works better generally
    chkpt_dir=r'C:\Users\saeidit\Documents\PhD\Research\OD_Estimation\DQN\New folder\big_run'
    action_space = {0: (0,0), 1: (1,0) , 2:(-1,0) , 3: (0,1) , 4:(0,-1)}
    actions=[0, 1, 2, 3, 4]
    n_actions=len(actions)
    mem_size=1000
    eps_min=0.05
    batch_size=256 #64
    eps_dec=0.001 #0.0001
    n_games = 5000 #2000
    iterations = 200 #250
    gamma = 0.99
    input_dims = agent_size + env.size
    eps = 0.05 #1
    lr=0.0001 #0.0001
    agent = DQNAgent(agent_size, gamma, eps, lr, input_dims, n_actions, mem_size, eps_min, batch_size, eps_dec, chkpt_dir)
    k_value = 0.6 # from k-means at k=25
    list_clustrs, scores, improvements, list_best_rewards, gains, lst_initial_reward =[],[],[],[],[],[]
    best_loss = +np.inf
    
    for i in range(n_games):
        score=0
        iter = 0
        gain=0
        all_gain=0
        print(f'#############GAME{i}##############')
        obs , one_hot = env.reset()  #unsorted
        initial_reward = env.Reward(obs)[0]
        lst_initial_reward.append(initial_reward)
        best_reward = initial_reward
        last_reward = initial_reward
    
        while iter<iterations:
            action, selected_state, sorted_obs, ID = agent.choose_action(obs, one_hot) #selected state is concatenation of one-hot obs and agent ID array
            obs_, reward, new_df = env.step(sorted_obs, action, ID)
            one_hot_ = env.obs_to_one_hot(obs_)
            score += reward
            improve = reward - initial_reward
            gain = reward - last_reward 
            all_gain += gain
            
            if not load_checkpoint: #Train Mode
                selected_future_state = agent.choose_action(obs_, one_hot_)[1] #to find best sort of future state
                agent.store_transition(selected_state, action, gain, selected_future_state)    
            
            else:  #Test mode
                if reward > best_reward:
                    best_reward = reward
                    best_cluster = new_df['cluster'].tolist()
                    
                if reward > k_value:
                    k_value = reward
                    best_cluster_ever = new_df['cluster'].tolist()    
                    centroids = obs_
            obs = obs_
            iter += 1
            last_reward = reward
        
        scores.append(score)
        gains.append(all_gain) #comulutive gain at each iteration
        
        if load_checkpoint: 
            list_clustrs.append(best_cluster)
            list_best_rewards.append(best_reward)
            improvements.append(improve) #comparing last reward of the game with first reward
        if not load_checkpoint: 
            agent.learn()
            if i> 10:
                avg_loss = np.mean(agent.losses[-10:])
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    agent.save_models()
                    
        
    ##########################  save  #########################    
    if load_checkpoint: 
        with open('data_frames.pickle', 'wb') as file:
            pickle.dump(list_clustrs, file)   
            
    #if load_checkpoint: #test
    save_df = pd.DataFrame({'Best cluster':[best_cluster_ever], 'Highest objective value': k_value,'centroids': [centroids]})
    save_df.to_csv('C:/Users/saeidit/Documents/PhD/Research/Data/Regina/cluster output/best_cluster_ever_test_big.csv')    

    if not load_checkpoint: 
        lst_score=[]
        for game in range(n_games):
            lst_score.append(np.mean(scores[max(0,game-500):game+1])/iterations)
        plt.plot(lst_score)
        plt.xlabel('Games')
        plt.ylabel('Ave. achieved objective of the Games')
        
        
        y=[]
        for game in range(n_games):
            y.append(np.mean(gains[max(0,game-500):game+1]))
        #plt.plot(lst_score[200:])
        y=y[200:]
        x=list(range(200 , len(lst_score)))
        fig = plt.figure()
        fig, ax = plt.subplots()
        ax.plot(x,y)
        ax.set_xlabel('Games')
        ax.set_ylabel('Ave. Score of the Last 200 Games')
        ax.set_xlim(200, 1000)

        lst_losses=[]
        for game in range(n_games):
            lst_losses.append(np.mean(agent.losses[max(0,game-500):game+1]))
        plt.plot(lst_losses)  
        plt.xlabel('Games')
        plt.ylabel('Training Loss')
        save_df = pd.DataFrame({'loss function during all iterations':[agent.losses],'cumulative reward of games':[scores],
                                'improvements from initial':[improvements], 'best rewards each game':[list_best_rewards], 'gains':[gains],
                                'initial rewards of games':[lst_initial_reward]})
        save_df.to_csv('C:/Users/saeidit/Documents/PhD/Research/Data/Regina/cluster output/test_gain_score_big.csv')    
