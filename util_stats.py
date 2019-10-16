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
from params_test import params_test
from params_compound import params_compound
from parameters_baseline import parameters_baseline
from parameters import parameters
start_time = time.time()
import json
import argparse
import cProfile
import itertools
from copy import copy


def get_filename_pattern(num_steps, minority_pc, M0, M1, temp,height,width,
        move,symmetric_positions, residential_steps,alpha, density,schelling,
        school_moves_per_step, residential_moves_per_step, bounded, radius, cap_max, T,fs, variable_f,sample,sigma, num_neighbourhoods, schools_per_neighbourhood, displacement,b,f0,f1):

    fs_print = fs
    if fs_print=="eq":
        fs_print= 0


    factor="f0"
    filename_pattern="V4_%s_m=%.2f_M0=%.2f_M1=%.2f_temp_%.2f_h_%d_st_%d_move_%s_sym_%s_res_%d_a_%.2f_den_%.2f_schell_%s_s_mps_%d_r_mps_%d_bounded_%s_r_%d_cp_%.2f_T_%.2f_fs_%.2f_v%s_s%d_sig%.2f_n%d_sn%d_d%d_b%.2f"%(
        factor,minority_pc, M0, M1, temp,height, num_steps,
        move,symmetric_positions, residential_steps,alpha, density,schelling,
        school_moves_per_step, residential_moves_per_step, bounded, radius, cap_max, T,fs_print, str(variable_f)[0],sample,sigma, num_neighbourhoods, schools_per_neighbourhood, displacement,b)

    return(filename_pattern)


def run_one_simulation(i,f0, return_list,return_list_agents, params, num_steps):

        """
        Run one simulation of the model and gather the statistics
        :param i:
        :param f0:
        :param return_list:
        :return: model_out: pandas dataframe of the model output data
                            - the datacollector and some addtional parameters
        """
        segregation_index=[]
        start_time=time.time()
        model = SchoolModel(**params)

        average_diff = 10
        # Stop if it did not change enough the last 70 steps

        total_steps = params['residential_steps'] + num_steps
        max_steps=total_steps+30
        while model.running and (model.schedule.steps < total_steps or average_diff>0.05) and model.schedule.steps<max_steps:
            model.step()
            segregation_index.append(model.seg_index)
            x2 = np.mean(segregation_index[-10:] )
            x1 = np.mean(segregation_index[-200:-190] )
            average_diff = (x2-x1)/x2
            print("steps ",model.schedule.steps)


        model_out = model.datacollector.get_model_vars_dataframe()
        model_out_agents = model.datacollector.get_agent_vars_dataframe()


        length = len(model_out)
        length_agents = len(model_out_agents)

        model_out['iter'] = np.repeat(i, length)
        model_out["f0"] = np.repeat(model.f[0], length)
        model_out["res"]= np.repeat(model.residential_steps, length)

        model_out_agents['iter'] = np.repeat(i, length_agents)
        model_out_agents['f0'] = np.repeat(model.f[1], length_agents)
        model_out_agents["res"] = np.repeat(model.residential_steps, length_agents)

        elapsed_time = time.time() - start_time
        print(elapsed_time)
        return_list.append(model_out)
        return_list_agents.append(model_out_agents)
        return([model_out, model_out_agents])




def run_simulation(params, all_f0_f1, num_steps,save_agents=False):
    """
    Run the model for multiple f0 and concatenate the results
    :return:
    """
    print("proceses",multiprocessing.cpu_count())
    manager=multiprocessing.Manager()
    return_list = manager.list()
    return_list_agents = manager.list()
    jobs = []

    #pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()/4)

    start_time=time.time()
    i=0;
    factor = "f0"
    all_model_agents_df = pd.DataFrame( columns={"AgentID","local_composition", "type", "id", "iter", "f0","f1"})

    for f0 in all_f0_f1:
        params['f0'] = f0
        params['f1'] = f0
        print(params)
        p=multiprocessing.Process(target=run_one_simulation, args=(i,f0,return_list,return_list_agents,params, num_steps))
        jobs.append(p)
        p.start()


        #all_models.append(model_out)
        i+=1
    for proc in jobs:
        proc.join()

    print(return_list)
    all_models_df = pd.concat(return_list)
    all_model_agents_df = pd.concat(return_list_agents)

    all_models_df.index.name = 'Step'
    all_models_df = all_models_df.reset_index().set_index([factor, 'Step'])

    all_model_agents_df.index = pd.MultiIndex.from_tuples(all_model_agents_df.index, names=['Step', 'Id'])
    all_model_agents_df = all_model_agents_df.reset_index().set_index([factor, 'Step', 'Id'])


    filename_pattern = get_filename_pattern( num_steps=num_steps,**params  )
    all_models_df.to_pickle("dataframes/models_"+ filename_pattern + time.strftime("%m%d%H%M"))
    print(all_model_agents_df[all_model_agents_df['type']==4])
    if save_agents:
        all_model_agents_df.to_pickle("dataframes/agents_"+ filename_pattern + time.strftime("%m%d%H%M"))


    return(all_models_df)
