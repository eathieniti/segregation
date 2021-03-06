import numpy as np
np.__file__

import pandas as pd

import sys
import numpy as np

from model import SchoolModel

import time
import glob
import os
import multiprocessing 

import time
start_time = time.time()

def get_filename_pattern():
    fs_print = fs
    if fs_print=="eq":
        fs_print= 0

    if factor =="alpha":

        filename_pattern="fasts_minority=%.2f_f0=%.2f_f1=%.2f_M0=%.2f_M1=%.2f_temp_%.2f_height_%d_steps_%d_move_%s_sym_%s_res_%d_schools_%d_den_%.2f_schell_%s_school_mps_%d_res_mps_%d_bounded_%s_radius_%d_cp_%.2f_T_%.2f_fs_%.2f"%(
            factor,minority_pc, f0, f0, M0, M1 ,temp,height, num_steps,
        move,symmetric_positions, residential_steps, num_schools, density,schelling,
        school_moves_per_step, residential_moves_per_step, bounded,radius,cap_max, T,fs_print)

    if factor =='f0':
        filename_pattern="fasts_sub_%s_minority=%.2f_M0=%.2f_M1=%.2f_temp_%.2f_height_%d_steps_%d_move_%s_sym_%s_res_%d_schools_%d_alpha_%.2f_den_%.2f_schell_%s_school_mps_%d_res_mps_%d_bounded_%s_radius_%d_cp_%.2f_T_%.2f_fs_%.2f"%(
            factor,minority_pc, M0, M1, temp,height, num_steps,
        move,symmetric_positions, residential_steps, num_schools, alpha, density,schelling,
        school_moves_per_step, residential_moves_per_step, bounded, radius, cap_max, T,fs_print)

    return(filename_pattern)

def run_simulation():
    
    start_time=time.time()
    i=0;
    
    all_models_df = pd.DataFrame( columns={"agent_count", "seg_index", "happy","total_moves", "iter", "f0", "f1"})
    all_model_agents_df = pd.DataFrame( columns={"AgentID","local_composition", "type", "id", "iter", "f0","f1"})

    for f0 in all_f0_f1:
        start_time=time.time()
        model = SchoolModel(height=height, width=width, density=density, num_schools=num_schools,minority_pc=minority_pc, homophily=3,f0=f0,f1=f0,M0=M0,T=T,
                            M1=M1 , alpha=alpha, temp=temp,cap_max=cap_max,
                           move=move, symmetric_positions=symmetric_positions, residential_steps=residential_steps,
                            schelling=schelling, bounded=bounded, residential_moves_per_step=residential_moves_per_step,
                           school_moves_per_step=school_moves_per_step,radius=radius,fs=fs)

        # Stop if it did not change enough the last 70 steps

        while model.running and (model.schedule.steps < total_steps or average_diff>0.05) and model.schedule.steps<max_steps:
            model.step()
            segregation_index.append(model.seg_index)
            x2 = np.mean(segregation_index[-10:] )
            x1 = np.mean(segregation_index[-200:-190] )
            print(x2,x1)
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
        model_out["f1"] = np.repeat(f1, length)
        model_out["alpha"] = np.repeat(alpha, length)
        model_out["res"]= np.repeat(residential_steps, length)

        model_out_agents['iter'] = np.repeat(i, length_agents)
        model_out_agents['f0'] = np.repeat(f0, length_agents)
        model_out_agents['f1'] = np.repeat(f1, length_agents)
        model_out_agents["alpha"] = np.repeat(alpha, length_agents)
        model_out_agents["res"] = np.repeat(residential_steps, length_agents)


        #all_models.append(model_out)
        all_models_df = all_models_df.append(model_out)
        all_model_agents_df = all_model_agents_df.append(model_out_agents)
        i+=1






        elapsed_time = time.time() - start_time
        print(elapsed_time)

    all_models_df.index.name = 'Step'
    all_models_df = all_models_df.reset_index().set_index([factor, 'Step'])
    all_model_agents_df.index = pd.MultiIndex.from_tuples(all_model_agents_df.index, names=['Step', 'Id'])
    all_model_agents_df = all_model_agents_df.reset_index().set_index([factor, 'Step', 'Id'])


    filename_pattern = get_filename_pattern()
    all_models_df.to_pickle("dataframes/models_"+ filename_pattern + time.strftime("%Y-%m-%d-%H_%M"))

    all_model_agents_df.to_pickle("dataframes/agents_"+ filename_pattern + time.strftime("%Y-%m-%d-%H_%M"))
    




# your code
all_models_df = pd.DataFrame( columns={"agent_count", "seg_index", "happy","total_moves", "iter", "f0", "f1"})
all_model_agents_df = pd.DataFrame( columns={"AgentID","local_composition", "type", "id", "iter", "f0","f1"})

all_models = []
all_model_agents = []
f=0.7
all_f = [0.3,0.3,0.3,0.3,0.4,0.4,0.4,0.4,0.5,0.5,0.5,0.5,0.6,0.6,0.6,0.6]
all_alpha = [0,0.2,0.4,0.6]

alpha_idx = pd.Index([0,0.2,0.4,0.6])
idx = pd.Index(all_f)
f0=0.6
f1=0.6
temp=0.4
minority_pc = 0.5

all_f0_f1 = [0.01,0.01,0.1,0.1,0.2,0.2,0.3,0.3,0.4,0.4,0.5,0.5,0.6,0.6,0.7,0.7,0.8,0.8,0.9,0.9]

all_f0_f1 = [0.3,0.3,0.4,0.4,0.5,0.5,0.6,0.6,0.7,0.7,0.8,0.8,0.9,0.9]
all_f0_f1 = [0.45,0.45,0.55,0.55,0.65,0.65,0.6,0.6,0.7,0.7,0.75,0.75,0.8,0.8,0.9,0.9,0.4,0.4,0.5,0.5,0.3,0.4,0.2]


all_f0_f1 = [0.3,0.4,0.5,0.5,0.6,0.6,0.7,0.7,0.8,0.8,0.9,0.9]
all_f0_f1 = [0.45,0.45,0.55,0.55,0.65,0.65,0.6,0.6,0.7,0.7,0.75,0.75,0.8,0.9,0.4,0.5,0.5,0.3,0.2]


density = 0.9; num_schools = 64; minority_pc =  0.50;
homophily = 3; f0 =  0.70; f1 =  0.70; M0 =  0.8; M1 =  0.8;
alpha =  0.2; temp =  0.1; cap_max =  1.01; move = "boltzmann"; symmetric_positions = True;
schelling=False;bounded=False;radius=6;T=0.75;
school_moves_per_step=500; residential_moves_per_step=500;fs=0.5;
i=0



# if move == 'deterministic':
#     num_steps=280
# else:
#     num_steps=400


num_steps=200

height=100
width=100


segregation_index= []
x2, x1 = 0,0
average_diff = 10

factor = "f0"



fs="eq"

for alpha in [0.3,0.4]:

#for temp in [0.02,0.1,0.5]:
#for T in [0.65,0.7,0.75]:

##for density in [0.85,0.9,0.95]:
#for radius in [3,5,6,7,9,11]:

    processes = []

    num_steps=200
    residential_steps=120;
    total_steps =residential_steps+num_steps
    max_steps=total_steps+20
    for i in range(0,10):
        p = multiprocessing.Process(target=run_simulation)
        processes.append(p)
        p.start()

    residential_steps =0;
    total_steps =residential_steps+num_steps
    max_steps=total_steps+20

    processes = []
    for i in range(0,10):
        p=multiprocessing.Process(target=run_simulation)
        processes.append(p)
        p.start()                   


