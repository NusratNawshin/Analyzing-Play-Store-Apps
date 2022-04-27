#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


df=pd.read_csv("Google-Playstore.csv")


# In[3]:


# Replacing the NaN with 0+ in the Installs Columns
df['Installs'] = df['Installs'].replace(np.nan, '0+')


# In[4]:


# Remove + and , from the string and convert the whole column to int type
df['Installs']=df.Installs.str.replace('[+,","]', '').astype('int64')


# In[5]:


df.dtypes


# In[6]:


# Filter the row where the number of installs is more than 100000
df_new=df[df['Installs']>100000]


# In[7]:


df_new=df_new[df_new['Rating Count']>4500]


# In[8]:


df_new


# In[9]:


# Replace all the NaN with 'No Information' for object type and '0.0' for float
# df['Installs'] = df['Installs'].replace(np.nan, '0+')
# df['Installs'] = df['Installs'].replace(np.nan, '0+')
# df['Installs'] = df['Installs'].replace(np.nan, '0+')
# df['Installs'] = df['Installs'].replace(np.nan, '0+')
df_new['Developer Email'] = df_new['Developer Email'].replace(np.nan, 'No Information')
df_new['Developer Website'] = df_new['Developer Website'].replace(np.nan, 'No Information')
df_new['Minimum Android'] = df_new['Minimum Android'].replace(np.nan, 'No Information')
df_new['Size'] = df_new['Size'].replace(np.nan, 'No Information')
df_new['Currency'] = df_new['Currency'].replace(np.nan, 'No Information')
df_new['Rating Count'] = df_new['Rating Count'].replace(np.nan, 0.0)
df_new['Rating'] = df_new['Rating'].replace(np.nan, 0.0)
# df_new['Released'] = df_new['Released'].replace(np.nan, 'No Information')
df_new["Released"].fillna(df_new["Last Updated"], inplace=True)
df_new['Privacy Policy'] = df_new['Privacy Policy'].replace(np.nan, 'No Information')


df_new.isnull().sum()


# In[10]:


df_new['Released'] = df_new['Released'].astype('datetime64[ns]')
df_new['Last Updated'] = df_new['Last Updated'].astype('datetime64[ns]')


# In[11]:


df_new


# In[12]:


# df_new[(df_new['Released'] > '2019-04-21') & (df_new['Released'] < '2020-04-21')]


# In[13]:


df_new


# In[14]:


df_new.dtypes


# In[15]:


df_new


# In[ ]:





# In[16]:


df_new=df_new.drop(['Scraped Time','Last Updated'],axis=1)


# In[17]:


df_new.reset_index(inplace=True)
df_new.drop("index", axis=1, inplace=True)
df_new


# In[18]:


df_new.to_csv("Play_Store_Dash_new.csv")
# df_new['Rating'].unique()





