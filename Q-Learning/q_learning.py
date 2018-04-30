# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 05:57:49 2018

@author: rizkyfalih
"""

# Importing the libraries
import pandas as pd
import numpy as np
import random



def build_state(raw):
    nState = len(raw) * len(raw[0])
    list = []
    for i in range (nState):
        list.append(i)
    
    return list

def build_valid_actions(trans):
    va = []
    for i in range(len(trans)):
        current_va = []
        if (trans[i][0] != -1): #up
            current_va.append(0)
        if (trans[i][1] != -1): #right
            current_va.append(1)
        if (trans[i][2] != -1): #down
            current_va.append(2)
        if (trans[i][3] != -1): #left
            current_va.append(3)
        current_va.append(4)
        va.append(current_va)
    return va

def Build_Transition(raw):
    xlen = len(raw)
    ylen = len(raw[0])
    
    d = {'up':[] ,'right':[], 'down':[], 'left':[]}
    d = pd.DataFrame(columns = ['up', 'right', 'down', 'left', 'none'])
    idx = 0 
    
    for i in range(ylen):
        for j in range(xlen):
            trans = [-1,-1,-1,-1, idx]
            if i>0: #up
                trans[0] = idx-xlen
            if i<ylen-1: #down
                trans[2] = idx+xlen
            if j>0: #left
                trans[3] = idx-1
            if j<xlen-1: #right
                trans[1] = idx+1
            d.loc[idx] = trans
            idx+=1
    return d

def Build_R_Matrix(raw):
    xlen = len(raw)
    ylen = len(raw[0])
    
    d = pd.DataFrame(columns = ['up','right','down','left', 'this'])
    idx = 0
    for i in range(ylen):
        for j in range(xlen):
            move = [float('-inf'),float('-inf'),float('-inf'),float('-inf'),raw[i][j]]
            if i>0: #up
                move[0] = raw[i-1][j]
            if i<ylen-1: #down
                move[2] = raw[i+1][j]
            if j>0: #left
                move[3] = raw[i][j-1]
            if j<xlen-1: #right
                move[1] = raw[i][j+1]
            d.loc[idx] = move
            idx+=1
    return d                

def Q_Learning(R, Q, trans, va, states):
    gamma = 0.9
    episodes = 500
    reward_list = []
    for i in range(episodes):
        start_state = random.choice(states)
        goal_state = 9
        current_state = start_state
        current_reward = R[current_state][4]
        while current_state != goal_state:
            action = random.choice(va[current_state])
            next_state = trans[current_state][action]
            future_rewards = []
            for action_nxt in va[next_state]:
                future_rewards.append(Q[next_state][action_nxt])
                
            # Update Q
            qstate = R[current_state][action] + gamma*max(future_rewards)
            Q[current_state][action] = qstate
    
            # Move to the next state and update Reward
            current_state = next_state
            current_reward += R[current_state][4]
            
            # Add total reward to a list if went to the goal state
            if(current_state == goal_state):
                reward_list.append(current_reward)
                
    print('Maximum Reward Learning = ' + str(max(reward_list)))
    return Q

def Path_Q(Q, R, va, trans):
    start_state = 90
    goal_state = 9
    best_reward = []
    path = []
    current_state = start_state
    current_reward = 0
    while current_state != goal_state:    
        max_q = Q[current_state].tolist().index(max(Q[current_state]))
        next_state = trans[current_state][max_q]
        current_state = next_state
        path.append(next_state)
        current_reward += R[current_state][4]
        if(current_state == goal_state):
            best_reward.append(current_reward)
    
    print("State yg dilalui: " + str(path))
    print("Total Reward: " + str(max(best_reward)))

# =========================================
# Main Program
# Import the data
raw = pd.read_csv('DataTugasML3.txt', delimiter = '\t', header = None)
raw = np.array(raw).tolist()

# Build State
states = build_state(raw)

# Build R Matix
R = Build_R_Matrix(raw)
R = np.array(R)

# Build Q Matrice
Q = np.zeros((100,5))

# Build Transisition Matrix
trans = Build_Transition(raw)
trans = np.array(trans)

# Build List of Valid Actions
va = build_valid_actions(trans)
va = np.array(va)

# The Q-Learning
Q = Q_Learning(R, Q, trans, va, states)

# The Q-Path
Path_Q(Q, R, va, trans)