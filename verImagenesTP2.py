#!/usr/bin/env python
# coding: utf-8

# ## Visualizar imagenes 

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Dataset 

# In[5]:


train2 = pd.read_csv("sign_mnist_train.csv")
train2.head()


# # Convertimos a array el dataframe y nos quedamos con una foto en formato (28x28)

# In[7]:


train = pd.read_csv("sign_mnist_train.csv").values
train[3][1:].reshape(28, 28)


# # Visualizamos una imagen

# In[3]:


plt.matshow(train[11][1:].reshape(28, 28), cmap = "gray")

