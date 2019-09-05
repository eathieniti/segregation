#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
np.__file__

import pandas as pd

import sys
import numpy as np

from model import SchoolModel

import time
import glob
import os



from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np


from params_copy import *



# In[2]:


from params_copy import *


# In[3]:


total_steps=num_steps+residential_steps;max_steps=total_steps;

x2, x1 = 0,0
average_diff = 10
factor = "f0"
fs="eq"


# In[4]:



def run_one_simulation(paramset):
        print(paramset)
        """
        Run one simulation of the model and gather the statistics
        :param i:
        :param f0:
        :param return_list:
        :return: model_out: pandas dataframe of the model output data
                            - the datacollector and some addtional parameters
        """
        
        [T, cap_max, temp, alpha, sigma, f0] = paramset
        segregation_index=[]
        start_time=time.time()
    


        model = SchoolModel(height=height, width=width, density=density, minority_pc=minority_pc, f0=f0,f1=f0,M0=M0,T=T,
                            M1=M1 , alpha=alpha, temp=temp,cap_max=cap_max,
                           move=move, symmetric_positions=symmetric_positions, residential_steps=residential_steps,
                            schelling=schelling, bounded=bounded, residential_moves_per_step=residential_moves_per_step,
                           school_moves_per_step=school_moves_per_step,radius=radius,fs=fs, variable_f=variable_f, sigma=sigma, sample=sample, 
                           num_neighbourhoods=num_neighbourhoods, schools_per_neighbourhood=schools_per_neighbourhood, displacement=displacement)

        # Stop if it did not change enough the last 70 steps
        while model.running and (model.schedule.steps < total_steps or average_diff>0.05) and model.schedule.steps<max_steps:
            model.step()
            segregation_index.append(model.seg_index)
            x2 = np.mean(segregation_index[-10:] )
            x1 = np.mean(segregation_index[-200:-190] )
            average_diff = (x2-x1)/x2
            print("steps ",model.schedule.steps)


        model_out = model.datacollector.get_model_vars_dataframe()

        elapsed_time = time.time() - start_time
        print(elapsed_time)

        output = [model_out.seg_index.tail(1), model_out.residential_segregation.tail(1)]

        return(output)


# In[5]:



segregation_problem = { 
    'num_vars': 6,
    'names': ['T', 'cap_max', 'temp', 'alpha', 'sigma','f0'],
    'bounds': [[0.6,0.99],
               [1.01,3],
               [0.01,0.5],
              [0.1,0.9],
              [0.1,0.5],
                [0.3,0.8]]}
    


# In[6]:


residential_steps=80;


# In[7]:


param_values = saltelli.sample(segregation_problem, 5)


# In[ ]:


Y = np.zeros([param_values.shape[0]])
Y2 =  np.zeros([param_values.shape[0]]) 
for i, paramset in enumerate(param_values):
    [Y[i], Y2[i]] = run_one_simulation(paramset)
    Y.to_pickle("dataframes/"+ "sensitivity"+time.strftime("%Y-%m-%d-%H_%M"))
    Y2.to_pickle("dataframes/"+ "sensitivity"+time.strftime("%Y-%m-%d-%H_%M"))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




