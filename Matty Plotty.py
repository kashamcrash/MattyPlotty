#!/usr/bin/env python
# coding: utf-8

# In[125]:


import numpy as np
import matplotlib.pyplot as plt


# In[126]:


Products = np.array(['Smartphones', 'Television', 'Laptops', 'Accessories', 'Batteries'])
Sales = np.array([111051.09, 263457.05, 387728.25, 428643.34, 362563.33])

plt.bar(Products, Sales)
plt.show()


# In[127]:


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


# In[128]:


import pandas as pd

# The dataset has been sourced from https://www.kaggle.com/aungpyaeap/fish-market

Fish = pd.read_excel('Fish.xlsx')

species = np.array(Fish['Species'])
weight = np.array(Fish['Weight'])
height = np.array(Fish['Height'])
width = np.array(Fish['Width'])

Fish.head(5)


# In[129]:


# Plotting scatter chart

plt.scatter(weight, height, alpha= 0.7, s = 25 )

# Adding and formatting title
plt.title("Weight versus Height of Fish By Species", 
          fontdict={'fontsize': 15, 'fontweight' : 1, 'color' : 'black'})

# Labeling Axes
plt.xlabel("Weight", fontdict={'fontsize': 13, 'fontweight' : 1, 'color' : 'black'})
plt.ylabel("Height", fontdict={'fontsize': 13, 'fontweight' : 1, 'color' : 'black'})

plt.show()


# In[130]:


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


# In[131]:


# Thank You:) 


# Line Graphs 

# In[132]:


# The dataset has been sourced from https://www.kaggle.com/kyanyoga/sample-sales-data

df1 = pd.read_csv('MonthlySales.csv')
df2 = df1.head(12)
df2

# We are using the first 12 months to create the graph

Date = np.array(df2['Month'])
Sales = np.array(df2['Sales'])


# In[133]:


plt.plot(Date, Sales)
plt.show()


# In[134]:


from matplotlib.pyplot import figure

figure(figsize=(10, 5))
plt.plot(Date, Sales, color = 'lightblue')

# Adding and formatting title
plt.title("Monthly Sales in Thousands", fontdict={'fontsize': 15, 'fontweight' : 5, 'color' : 'black'})

# Labeling Axes
plt.xlabel("Month", fontdict={'fontsize': 13, 'fontweight' : 1, 'color' : 'Black'})
plt.ylabel("Sales", fontdict={'fontsize': 13, 'fontweight' : 1, 'color' : 'Black'} )

# Modifying the ticks to show information in lakhs
ticks = np.arange(0, 90000, 5000)
labels = ["{}T".format(i//1000) for i in ticks]
plt.yticks(ticks, labels)
plt.xticks(rotation=90)

for xy in zip(Date, Sales):
    plt.annotate(s = "{}T".format(xy[1]//1000), xy = xy,  textcoords='data')
plt.show()


# In[135]:


# Creating np.arrays for Subplots with Sample Sales Data of Retail Stores

years = np.array(['2016', '2017', '2018', '2019','2020'])
aMart = np.array([138298, 255590, 435317, 394147, 659861])
bMart = np.array([403867, 497683, 637643, 768219, 829822])
cMart = np.array([396108, 420362, 608140, 706632, 988522])
dMart = np.array([124387, 485134, 408102, 248189, 223050])


# In[136]:


fig, sb = plt.subplots()
fig.set_size_inches(10.5, 5.5, forward = True)

aMart_Sales, = sb.plot(years, aMart, linestyle = '-' , linewidth = 2, color = 'lightgreen', alpha = 1)
aMart_Sales.set_label("aMart Sales")

bMart_Sales, = sb.plot(years, bMart, linestyle = '--' , linewidth = 2, color = 'lightblue', alpha = 1)
bMart_Sales.set_label("bMart Sales")
bMart_Sales.set_dashes([2, 2, 2, 2])

cMart_Sales, = sb.plot(years, cMart, linestyle = '-.' , linewidth = 2, color = 'lightpink', alpha = 1)
cMart_Sales.set_label("cMart Sales")
cMart_Sales.set_dashes([2, 2, 5, 2])

dMart_Sales, = sb.plot(years, dMart, linestyle = ':' , linewidth = 2, color = 'yellow', alpha = 1)
dMart_Sales.set_label("dMart Sales")
dMart_Sales.set_dashes([2, 2, 10, 2])

plt.legend()

plt.legend(bbox_to_anchor=(1.31, 0.4))

plt.show()


# In[137]:


fig, sb = plt.subplots(ncols = 2, nrows = 2, sharex = True, sharey = True)
fig.set_size_inches(10, 7, forward = True)

aMart_Sales, = sb[0][0].plot(years, aMart, linestyle = '-' , linewidth = 2, color = 'olive', alpha = 1)
aMart_Sales.set_label("aMart Sales")
cMart_Sales.set_dashes([2, 2, 5, 2])

bMart_Sales = sb[0][1].bar(years, bMart, color = 'lightpink', alpha = 1, edgecolor = 'black', width = 0.5)
bMart_Sales.set_label("bMart Sales")

cMart_Sales = sb[1][0].scatter(years, cMart, alpha = 0.7, s = 25)
cMart_Sales.set_label("cMart Sales")

dMart_Sales, = sb[1][1].plot(years, dMart, linestyle = ':' , linewidth = 2, color = 'yellow', alpha = 1)
dMart_Sales.set_label("dMart Sales")

plt.show()


# In[138]:


# Thank You:)

