#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


Products = np.array(['Smartphones', 'Television', 'Laptops', 'Accessories', 'Batteries'])
Sales = np.array([111051.09, 263457.05, 387728.25, 428643.34, 362563.33])

plt.bar(Products, Sales)
plt.show()


# In[3]:


# Plotting bar chart and setting bar width to 0.5 and aligning it to center
plt.bar(Products, Sales, width= 0.5, align='center', edgecolor='black', 
        color=['purple', 'pink', 'lightgreen', 'royalblue', 'cyan'])

# Adding and formatting title
plt.title("Sales Across Products", fontdict={'fontsize': 15, 'fontweight' : 1, 'color' : 'black'})

# Labeling Axes
plt.xlabel("Products", fontdict={'fontsize': 13, 'fontweight' : 1, 'color' : 'black'})
plt.ylabel("Sales", fontdict={'fontsize': 13, 'fontweight' : 1, 'color' : 'black'})

# Modifying the ticks to show information in lakhs
ticks = np.arange(0, 500000, 100000)
labels = ["{}L".format(i//100000) for i in ticks]
plt.yticks(ticks, labels)

plt.show()


# In[4]:


import pandas as pd

# The dataset has been sourcd from https://www.kaggle.com/aungpyaeap/fish-market

Fish = pd.read_excel('Fish.xlsx')

species = np.array(Fish['Species'])
weight = np.array(Fish['Weight'])
height = np.array(Fish['Height'])
width = np.array(Fish['Width'])

Fish.head(5)


# In[5]:


# Plotting scatter chart

plt.scatter(weight, height, alpha= 0.7, s = 25 )

# Adding and formatting title
plt.title("Weight versus Height of Fish By Species", 
          fontdict={'fontsize': 15, 'fontweight' : 1, 'color' : 'black'})

# Labeling Axes
plt.xlabel("Weight", fontdict={'fontsize': 13, 'fontweight' : 1, 'color' : 'black'})
plt.ylabel("Height", fontdict={'fontsize': 13, 'fontweight' : 1, 'color' : 'black'})

plt.show()


# In[6]:


plt.scatter(height[species == "Bream"], weight[species == "Bream"], 
            cmap= 'blue', alpha= 0.7, s = 150, label="Bream" )

plt.scatter(height[species == "Roach"], weight[species == "Roach"], 
            cmap= 'orange', alpha= 0.7, s = 100, label="Roach" )

plt.scatter(height[species == "Smelt"], weight[species == "Smelt"], 
            cmap= 'green', alpha= 0.7, s = 50, label="Smelt" )

# Adding and formatting title
plt.title("Weight versus Height of Fish By Species", 
          fontdict={'fontsize': 15, 'fontweight' : 1, 'color' : 'black'})

# Labeling Axes
plt.xlabel("Weight", fontdict={'fontsize': 13, 'fontweight' : 1, 'color' : 'black'})
plt.ylabel("Height", fontdict={'fontsize': 13, 'fontweight' : 1, 'color' : 'black'})

# For xy in zip (height[species == "Bream"], weight[species == "Bream"]):
    # plt.annotate(s = "Bream", xy = xy)

plt.legend()

plt.show()


# In[ ]:


# Thank You:) 

