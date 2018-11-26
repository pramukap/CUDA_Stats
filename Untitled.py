import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from clustering import KMeans



data = np.array(pd.read_csv('iris.csv',header=None))
model = KMeans(k=3, iterations=20)

ks = model.fit(data)
print(ks)

labels = model.label(data)
print(labels)

# In[117]:

# In[118]:


#scatter_classes(data,labels,3,0,1)


# In[77]:


def distance(X,Y):
    return np.sum([(x-y)**2 for x,y in zip(X,Y)])


# In[78]:


data[0]


# In[79]:


ks[0]


# In[82]:


print(distance(ks[0],data[0]),
distance(ks[1],data[0]),
distance(ks[2],data[0]))


# In[81]:


def get_arg_min_dist(vector,ks):
    return np.argmin([distance(vector,ksi) for ksi in ks])

get_arg_min_dist(data[0],ks)


# In[83]:


model.label(data)


# In[84]:


ks


# In[85]:


data[0]

