# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 20:10:44 2018

@author: rizkyfalih
"""

import math

# Calculate Prior
def prior_probability(y, y_list):
    p = [0] * len(y_list)
    for i in range(len(y_list)):
        for j in range(len(y)):
            if y_list[i] == y[j]:
                p[i] +=1
    total_p =sum(p)
    for i in range(len(p)):
        p[i]/=total_p
    return p

def likelihood_probability(x, y, x_list, y_list):
    l = []
    for i in range(len(x_list)):
        a = []
        for j in range(len(y_list)):
            temp = 0
            for k in range(len(x)):
                if y[k] == y_list[j]:
                    temp += x[k].count(x_list[i])
            temp +=1
            a.append(temp)
        l.append(a)
    total_l = [0] * len(y_list)
    for i in range(len(l)):
        for j in range(len(l[i])):
            total_l[j] += l[i][j]
    for i in range(len(l)):
        for j in range(len(l[i])):
            l[i][j] /= total_l[j]
    return l

X_train = []
X_train.append(['Chinese', 'Beijing', 'Chinese']) 
X_train.append(['Chinese', 'Chinese', 'Shanghai'])
X_train.append(['Chinese', 'Macao'])
X_train.append(['Tokyo', 'Japan', 'Chinese'])

y_train = ['c','c','c','j']

label_list = list(set(y_train))

prior = prior_probability(y_train, label_list)

bag = []
for i in range(len(X_train)):
    bag += X_train[i]
    
word_list = list(set(bag))
print(word_list)

likelihood = likelihood_probability(X_train, y_train, word_list, label_list)

test = ['Chinese', 'Chinese', 'Chinese', 'Tokyo', 'Japan']

result = []
for i in range(len(label_list)):
    temp=0
    for j in range(len(test)):
        if test[j] in word_list:
            temp += math.log(likelihood[word_list.index(test[j])][i])
    temp += math.log(prior[i])
    result.append(temp)

print(result)
print(label_list[result.index(max(result))])