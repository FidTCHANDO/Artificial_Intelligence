# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 11:25:00 2022

@author: JULIA
"""

#Reinforcement Learning Guide: Solving the Multi-Armed Bandit Problem from Scratch in Python
#EXAMPLE
#Suppose an advertising company is running 10 different ads targeted towards a similar 
#set of population on a webpage. Each column index represents a different ad.
# We have a 1 if the ad was clicked by a user, and 0 if it was not. 
# Random Selection

# IMPORTING THE LIBRARIES
import pandas as pd
import os

# Importing the dataset
os.chdir('C:\\Users\\TCHANDO\\Documents\\Cours Master BIOSTAT\\M2\\Artificial_Intelligence\\Reinforcement')
dataset = pd.read_csv('Ads_Optimisation.csv')

print(dataset.head())
#We have a 1 if the ad was clicked by a user, and 0 if it was not.
# Implementing Random Selection

#First, we will try a random selection technique, where we randomly select any ad 
#and show it to the user. If the user clicks the ad, we get paid and if not, there is no profit.
print('\n\nRandom Selection\n\n')

import random
N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward
#Total reward for the random selection algorithm comes out to be 1214. As this algorithm is not learning anything, 
#it will not smartly select any ad which is giving the maximum return. And hence even if we look at the last 1500 trials, 
#it is not able to find the optimal ad.
print(pd.Series(ads_selected).tail(1500).value_counts(normalize=True))
total_reward

import matplotlib.pyplot as plt
plt.hist(ads_selected)
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected') 
plt.title('Histogram of ads selections RANDOMLY')
plt.show()


#Now, let’s try the Upper Confidence Bound algorithm to do the same
# Implementing UCB
print('\n\nImplementing UCB\n\n')
import math
N = 10000
d = 10
ads_selected = []#is used to append the differents types of ads selected in each round
numbers_of_selections = [0] * d#is used to count the number of time an ad was  selected
sums_of_reward = [0] * d#is used to calculate the cumulative sum of rewards at each round
total_reward = 0

for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_reward[i] / numbers_of_selections[i]
            delta_i = math.sqrt(2 * math.log(n+1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sums_of_reward[ad] += reward
    total_reward += reward

print(pd.Series(ads_selected).head(1500).value_counts(normalize=True))
total_reward
sums_of_reward
#The total_reward for UCB comes out to be 2125. Clearly, this is much better than random selection 
#and indeed a smart exploration technique that can significantly improve our strategy to solve a MABP.

#After just 1500 trials, UCB is already favouring Ad #5 (index 4) which happens to be the optimal ad, 
#and gets the maximum return for the given problem.
import matplotlib.pyplot as plt
plt.hist(ads_selected)
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected') 
plt.title('Histogram of ads selections UCB ')
plt.show()

#from the above vizualization we can see that the fifth ad got the highest click.
#so our model advice us to place the it to the user for getting the highest number of clicks.