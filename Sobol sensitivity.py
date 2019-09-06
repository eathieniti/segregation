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
import pickle


from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np
import multiprocessing

from params_copy import *


total_steps=num_steps+residential_steps;max_steps=total_steps;

x2, x1 = 0,0
average_diff = 10
factor = "f0"
fs="eq"


def get_filename_pattern(params):
    fs_print = fs
    if fs_print=="eq":
        fs_print= 0


    if factor =='f0':
        filename_pattern="%s_m=%.2f_M0=%.2f_M1=%.2f_temp_%.2f_h_%d_st_%d_move_%s_sym_%s_res_%d_a_%.2f_den_%.2f_schell_%s_s_mps_%d_r_mps_%d_bounded_%s_r_%d_cp_%.2f_T_%.2f_fs_%.2f_v%s_s%d_n%d_sn%d_d%d"%(
            factor,minority_pc, M0, M1, temp,height, num_steps,
        move,symmetric_positions, residential_steps,alpha, density,schelling,
        school_moves_per_step, residential_moves_per_step, bounded, radius, cap_max, T,fs_print, str(variable_f)[0],sample, num_neighbourhoods, schools_per_neighbourhood, displacement)

    return(filename_pattern)


# add all parameters

def run_one_simulation(paramset, return_list):
        print(paramset)
        """
        Run one simulation of the model and gather the statistics
        Also save the simulation?
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
        model_out.T=T;model_out.cap_max=cap_max;model_out.temp=temp;model_out.alpha=alpha;
        model_out.sigma=sigma;model_out.f0=f0;

        filename_pattern = get_filename_pattern()
        model_out.to_pickle("dataframes/models_" + filename_pattern + time.strftime("%m%d%H%M"))

        elapsed_time = time.time() - start_time
        print(elapsed_time)

        output = [model_out.seg_index.tail(1), model_out.residential_segregation.tail(1)]
        return_list.append(output)
        return(output)



segregation_problem = { 
    'num_vars': 6,
    'names': ['T', 'cap_max', 'temp', 'alpha', 'sigma','f0'],
    'bounds': [[0.6,0.99],
               [1.01,3],
               [0.01,0.5],
              [0.1,0.9],
              [0.1,0.5],
                [0.3,0.8]]}



def run_simulation():
    """
    Run the model for multiple f0 and concatenate the results
    :return:
    """
    print("processes", multiprocessing.cpu_count())

    manager = multiprocessing.Manager()
    return_list = manager.list()
    jobs = []

    # pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()/4)

    start_time = time.time()
    i = 0;

    all_model_agents_df = pd.DataFrame(columns={"AgentID", "local_composition", "type", "id", "iter", "f0", "f1"})

    for f0 in all_f0_f1:
        p = multiprocessing.Process(target=run_one_simulation, args=(i, f0, return_list))
        jobs.append(p)
        p.start()

        # all_models.append(model_out)
        i += 1
    for proc in jobs:
        proc.join()



def run_sensitivity_parallel(param_values):
    print("processes", multiprocessing.cpu_count())

    manager = multiprocessing.Manager()
    return_list = manager.list()
    jobs = []

    for i, paramset in enumerate(param_values):

        paramset_ = [ '%.2f' % elem for elem in paramset ]
        p = multiprocessing.Process(target=run_one_simulation, args=(paramset_, return_list))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    Y = pd.concat(return_list)

    with open("dataframes/"+ "sensitivity"+time.strftime("%Y-%m-%d-%H_%M"),'wb') as f:
        pickle.dump(Y, f)





residential_steps=80;
param_values = saltelli.sample(segregation_problem, 30)
run_sensitivity_parallel(param_values)
