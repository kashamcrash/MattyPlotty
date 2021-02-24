#!/usr/bin/env python
# coding: utf-8

# #### COVID-19 in India with Matplotlib

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')

# Covid-19 Dataset has been sourced from GitHub
# https://raw.githubusercontent.com/datasets/covid-19/main/data/countries-aggregated.csv
# https://github.com/datasets/covid-19


# In[2]:


covid = pd.read_csv('https://raw.githubusercontent.com/datasets/covid-19/main/data/countries-aggregated.csv')
#covid.head()
#covid.shape
#covid.dtypes
covid.info()


# In[3]:


# Date is object type, need to conver it to datetime format
import matplotlib.dates as mdates
covid1 = pd.read_csv('https://raw.githubusercontent.com/datasets/covid-19/main/data/countries-aggregated.csv', 
                     parse_dates = ['Date'])
covid1.dtypes
# covid1.head()


# In[4]:


# Sum of total cases -> creating a new column
covid1['Total Confirmed'] = covid1[['Confirmed', 'Recovered', 'Deaths']].sum(axis = 1)
covid1.head()


# In[5]:


# Plotting for wordlwide cases 
worldwide = covid1.groupby(['Date']).sum()
c = worldwide.plot(figsize = (8, 5))
c.set_xlabel('Month', fontdict = {'fontsize' : 14, 'fontweight' : 1})
c.set_ylabel('Covid-19 2020-2021', fontdict = {'fontsize' : 14, 'fontweight' : 1})
c.title.set_text('Covid-19 Worldwide Cases')

plt.legend(bbox_to_anchor=(1.0, 0.4))

plt.show()


# In[6]:


India = covid1[covid1['Country'] =='India'].groupby(['Date']).sum()
India.head()


# In[7]:


I = India.plot(figsize = (8, 5))
I.set_xlabel('Month', fontdict = {'fontsize' : 14, 'fontweight' : 1})
I.set_ylabel('Covid-19 2020-2021', fontdict = {'fontsize' : 14, 'fontweight' : 1})
I.title.set_text('Covid-19 Cases in India')

plt.legend(bbox_to_anchor=(1.0, 0.4))

plt.show()


# In[8]:


fig = figure(figsize = (8,5))
ax = fig.add_subplot(111)

ax.plot(worldwide[['Total Confirmed']], label = 'Worldwide')
ax.plot(India[['Total Confirmed']], label = 'India')
ax.set_xlabel('Month', fontdict = {'fontsize' : 14, 'fontweight' : 1})
ax.set_ylabel('Covid-19 2020-2021', fontdict = {'fontsize' : 14, 'fontweight' : 1})
ax.title.set_text('Worldwide vs India Covid-19 Cases')

plt.legend()

plt.legend(bbox_to_anchor=(1.3, 0.4))

plt.show()


# In[9]:


USA = covid1[covid1['Country'] =='US'].groupby(['Date']).sum()
Brazil = covid1[covid1['Country'] =='Brazil'].groupby(['Date']).sum()


# In[10]:


fig = figure(figsize = (8,5))
ax = fig.add_subplot(111)

# ax.plot(worldwide[['Total Confirmed']], label = 'Worldwide')
ax.plot(India[['Total Confirmed']], label = 'India')
ax.plot(USA[['Total Confirmed']], label = 'USA')
ax.plot(Brazil[['Total Confirmed']], label = 'Brazil')
ax.set_xlabel('Month', fontdict = {'fontsize' : 14, 'fontweight' : 1})
ax.set_ylabel('# Cases 2020-2021', fontdict = {'fontsize' : 14, 'fontweight' : 1})
ax.title.set_text('India vs USA vs Brazil Covid-19 Cases')

plt.legend()

plt.legend(bbox_to_anchor=(1.2, 0.4))

plt.show()


# In[11]:


# Thank You :) 

