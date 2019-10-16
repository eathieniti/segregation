
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

import pickle
from util_stats import *


parser = argparse.ArgumentParser()
parser.add_argument('--paramsf', help='parameters file', default="parameters")
parser.add_argument('--test', help='run minimum steps to test the file')
parser.add_argument('--profile', help='Profile code and write stats in a file')
parser.add_argument('--run_one_f', help='Run one f value only', type=float)
parser.add_argument('--save_agents', help='Save the agents datacollector')
args = parser.parse_args()
test = args.test;
profile = args.profile;
save_agents = args.save_agents;
paramsf = args.paramsf
run_one_f = args.run_one_f

all_f0_f1 = [0.45, 0.55, 0.65, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.4, 0.5, 0.3, 0.2]


def run_simulation(params, num_steps,return_list, save_agents=False):
    """
    Run the model for multiple f0 and concatenate the results
    :return:
    """

    #pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()/4)

    start_time=time.time()
    i=0;
    factor = "f0"
    all_model_agents_df = pd.DataFrame( columns={"AgentID","local_composition", "type", "id", "iter", "f0","f1"})
    return_list_dummy, return_list_agents_dummy = [], []
    [model_out, model_out_agents] = run_one_simulation(i,params['f0'],return_list_dummy,return_list_agents_dummy,params, num_steps )


    #all_models.append(model_out)

    print(model_out.seg_index)
    all_models_df = model_out
    all_model_agents_df = model_out_agents

    all_models_df.index.name = 'Step'
    all_models_df = all_models_df.reset_index().set_index([factor, 'Step'])

    all_model_agents_df.index = pd.MultiIndex.from_tuples(all_model_agents_df.index, names=['Step', 'Id'])
    all_model_agents_df = all_model_agents_df.reset_index().set_index([factor, 'Step', 'Id'])

    filename_pattern = get_filename_pattern( num_steps=num_steps,**params  )
    all_models_df.to_pickle("dataframes/models_"+ filename_pattern + time.strftime("%m%d%H%M"))

    output = all_models_df.tail(1)
    #output = [all_models_df.seg_index.tail(1), all_models_df.residential_segregation.tail(1),all_models_df.mixed_index.tail(1),  all_models_df.res_seg_index.tail(1)]
    return_list.append(output)
    return (output)

   # return(all_models_df)



factor = "f0"

# test
num_steps = 1
# test
n_repeats = 1
run_one_f = 0.7

params_new = {
    'b': [1],
    'alpha': [1],
    # 'alpha':[0,0.4,0.2,0.6,0.8,1],
    # 'b':[0.3,0.2,0.1,0.5,0.6,0.2,0.4],
    # 'alpha':[0.2,0.4],
    # 'radius': [3,6,9],
    'residential_steps': [100, 0],
    # 'temp': [0.1,0.01],
    # 'sigma': [0.3,0.4],
    'T': [0.75, 0.8, 0.85]
}

if run_one_f:
    all_f0_f1 = [run_one_f]
    n_repeats = 1



elif paramsf == "parameters":
    params = copy(parameters)
    params_new = {
        'b': [0.0, 0.15, 0.1],
        'alpha': [0.25],
        'symmetric_positions': [True],
        # 'radius': [3,6,9],
        'residential_steps': [100, 0],
        # 'temp': [0.1,0.01],
        # 'sigma': [0.3,0.4],
        'T': [0.75]
    }

else:
    print("params file not valid")
    sys.exit()

if run_one_f:
    all_f0_f1 = [run_one_f]
    n_repeats = 1


def run_sensitivity_parallel(params, params_new_values, params_new_keys, num_steps):
    print("processes", multiprocessing.cpu_count())

    manager = multiprocessing.Manager()
    return_list = manager.list()
    jobs = []

    i=0
    for i, paramset in enumerate(params_new_values):

        params_new_values_r = [ round(elem,2) % elem for elem in paramset ]
        params_new = dict(zip(params_new_keys,params_new_values_r))

        # Replace parameter set with the new ones
        for key in params_new:
            params[key] = [params_new[key]]

        keys = list(params)
        print(i,params)
        for values in itertools.product(*map(params.get, keys)):
            #p = multiprocessing.Process(target=run_simulation, args=(params=dict(zip(keys, values)),num_steps=num_steps))
            params_to_pass = dict(zip(keys, values))
            p = multiprocessing.Process(target=run_simulation, args=(params_to_pass, num_steps, return_list))

            print(dict(zip(keys, values)))
        jobs.append(p)
        p.start()
        i+=1
    for proc in jobs:
        proc.join()


    for ii in return_list:
        print("new",ii)

    Y = pd.concat(return_list)

    print("return_list", return_list)
    with open("dataframes/"+ "sensitivity"+time.strftime("%Y-%m-%d-%H_%M"),'wb') as f:
        pickle.dump(Y, f)





segregation_problem = {
    'num_vars': 4,
    'names': ['T', 'b', 'alpha','f0'],
    'bounds': [[0.65,0.85],
               [0.01,0.5],
               [0.01,0.5],
              [0.4,0.9]]}

params['residential_steps'] = [80]

#
num_steps = 100

params = copy(parameters)
if test:
    num_steps=1

    params['residential_steps']=[1]
    params['height']=[25]
    params['width'] = [25]
    params['sample']=[10]

new_param_values = saltelli.sample(segregation_problem, 10)
print("simulations ",(np.shape(new_param_values)))
run_sensitivity_parallel(params,new_param_values, segregation_problem['names'], num_steps)

run_name = "100"

with open("dataframes/" + "sensitivity_parameters" + run_name + time.strftime("%Y-%m-%d-%H_%M"), 'wb') as f:
    pickle.dump(segregation_problem, f)




