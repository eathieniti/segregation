import numpy as np
np.__file__

import pandas as pd
import sys
import numpy as np

from model import SchoolModel

import glob
import os
import multiprocessing
import time
from params import params
start_time = time.time()
import json
import argparse
import cProfile
import itertools

parser = argparse.ArgumentParser()
parser.add_argument('--params', help='parameters file')
parser.add_argument('--test', help='run minimum steps to test the file')
parser.add_argument('--profile', help='Profile code and write stats in a file')
parser.add_argument('--run_one_f', help='Run one f value only', type=float)

args = parser.parse_args()
test = args.test; profile=args.profile
run_one_f = args.run_one_f

def get_filename_pattern(factor,num_steps, minority_pc, M0, M1, temp,height,
        move,symmetric_positions, residential_steps,alpha, density,schelling,
        school_moves_per_step, residential_moves_per_step, bounded, radius, cap_max, T,fs_print, variable_f,sample,sigma, num_neighbourhoods, schools_per_neighbourhood, displacement,b):

    fs_print = fs
    if fs_print=="eq":
        fs_print= 0


    if factor =='f0':
        filename_pattern="V3fx_%s_m=%.2f_M0=%.2f_M1=%.2f_temp_%.2f_h_%d_st_%d_move_%s_sym_%s_res_%d_a_%.2f_den_%.2f_schell_%s_s_mps_%d_r_mps_%d_bounded_%s_r_%d_cp_%.2f_T_%.2f_fs_%.2f_v%s_s%d_sig%.2f_n%d_sn%d_d%d_b%.2f"%(
            factor,minority_pc, M0, M1, temp,height, num_steps,
        move,symmetric_positions, residential_steps,alpha, density,schelling,
        school_moves_per_step, residential_moves_per_step, bounded, radius, cap_max, T,fs_print, str(variable_f)[0],sample,sigma, num_neighbourhoods, schools_per_neighbourhood, displacement,b)

    return(filename_pattern)


def run_one_simulation(i,f0, return_list,params):

        """
        Run one simulation of the model and gather the statistics
        :param i:
        :param f0:
        :param return_list:
        :return: model_out: pandas dataframe of the model output data
                            - the datacollector and some addtional parameters
        """
        print(f0)
        segregation_index=[]
        start_time=time.time()
        model = SchoolModel(**params)

        # Stop if it did not change enough the last 70 steps

        total_steps = params['residential_steps'] + num_steps
        max_steps=total_steps
        while model.running and (model.schedule.steps < total_steps or average_diff>0.05) and model.schedule.steps<max_steps:
            model.step()
            segregation_index.append(model.seg_index)
            x2 = np.mean(segregation_index[-10:] )
            x1 = np.mean(segregation_index[-200:-190] )
            average_diff = (x2-x1)/x2
            print("steps ",model.schedule.steps)


        model_out = model.datacollector.get_model_vars_dataframe()
        model_out_agents = model.datacollector.get_agent_vars_dataframe()
        model_out_agents = model_out_agents[model_out_agents.type==2]
        model_out_agents = model_out_agents


        length = len(model_out)
        length_agents = len(model_out_agents)

        model_out['iter'] = np.repeat(i, length)
        model_out["f0"] = np.repeat(f0, length)
        model_out["alpha"] = np.repeat(params['alpha'], length)
        model_out["res"]= np.repeat(params['residential_steps'], length)

        model_out_agents['iter'] = np.repeat(i, length_agents)
        model_out_agents['f0'] = np.repeat(f0, length_agents)
        model_out_agents["alpha"] = np.repeat(params['alpha'], length_agents)
        model_out_agents["res"] = np.repeat(params['residential_steps'], length_agents)
    
        elapsed_time = time.time() - start_time
        print(elapsed_time)
        return_list.append(model_out)
        return(model_out)



def run_simulation(params):
    """
    Run the model for multiple f0 and concatenate the results
    :return:
    """
    print("proceses",multiprocessing.cpu_count())
    print(params)
    manager=multiprocessing.Manager()
    return_list = manager.list()
    jobs = []

    #pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()/4)
            
    start_time=time.time()
    i=0;
    
    all_model_agents_df = pd.DataFrame( columns={"AgentID","local_composition", "type", "id", "iter", "f0","f1"})

    for f0 in all_f0_f1:
        p=multiprocessing.Process(target=run_one_simulation, args=(i,f0,return_list,params))
        jobs.append(p)
        p.start()

     
        #all_models.append(model_out)
        i+=1
    for proc in jobs:
        proc.join()
    
    
    all_models_df = pd.concat(return_list)
    print("results",return_list)

    all_models_df.index.name = 'Step'
    all_models_df = all_models_df.reset_index().set_index([factor, 'Step'])
    all_model_agents_df.index = pd.MultiIndex.from_tuples(all_model_agents_df.index, names=['Step', 'Id'])
    all_model_agents_df = all_model_agents_df.reset_index().set_index([factor, 'Step', 'Id'])


    filename_pattern = get_filename_pattern(factor=factor, num_steps=num_steps,**params  )
    all_models_df.to_pickle("dataframes/models_"+ filename_pattern + time.strftime("%m%d%H%M"))

    return(all_models_df)



all_f0_f1 = [0.45,0.55,0.65,0.6,0.7,0.75,0.8,0.85,0.9,0.4,0.5,0.3,0.2]
if run_one_f:
    all_f0_f1 = [run_one_f]
    

# test
# num_steps = 1

x2, x1 = 0,0
average_diff = 10
factor = "f0"
fs="eq"

#for T in [0.85,0.9,0.75,0.8,0.85]:

#for temp in [0.02,0.1,1]:
#for alpha in [0.3,0.4,0.5,0.2]:
#for density in [0.85,0.9,0.95]:
#for radius in [3,6,7,9,11]:


# test
#all_f0_f1 = [0.7]
n_repeats = 1
num_steps = 80
# test
n_repeats=10
if test:
    n_repeats=1
    all_f0_f1 = [0.5]
    num_steps=1


for i in range(0,n_repeats):

    #for temp in [0.5,0.1,0.01]:
    #for alpha in [0.2,0.1,0.3,1]:
    #for density in [0.85,0.9,0.95]:
    #for radius in [3,6,9,12]:
    #for fs in [0.3,0.5]:
    #for T in [0.65,0.75,0.8]:

    #for displacement in [4,8]:
    #for cap_max in [2,1.5,1.01]:
    #for sigma in [0.1,0.3,0.5]:

    params_new = {
    'b': [0, 1, 0.2],
        'alpha': [0, 1, 0.2],
        'temp': [0.01,0.1],
        'radius': [3,6,9,12],
        'T': [0.65,0.75],
         'sigma': [0.1,0.3],
        'residential_steps': [0,80]
    }

    for key in params_new:
        params[key] =  params_new[key]


    keys = list(params)
    for values in itertools.product(*map(params.get, keys)):
        all_models_df = run_simulation(params=dict(zip(keys,values)) )

    #for b in [1,0.4,0.2]:
    # for b in [1]:
    #
    #     for ii in range(0,3):
    #         residential_steps=80;
    #
    #         if test:
    #             residential_steps=1;
    #         total_steps =residential_steps+num_steps
    #         max_steps=total_steps
    #
    #         all_models_df = run_simulation()
    #
    #     for ii in range(0,2):
    #         residential_steps =0;
    #         total_steps =residential_steps+num_steps
    #         max_steps=total_steps
    #
    #         all_models_df = run_simulation()
    #
    #
    #
    #






