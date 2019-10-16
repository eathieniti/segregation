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

from util_stats import *


parser = argparse.ArgumentParser()
parser.add_argument('--paramsf', help='parameters file', default="parameters")
parser.add_argument('--test', help='run minimum steps to test the file')
parser.add_argument('--profile', help='Profile code and write stats in a file')
parser.add_argument('--run_one_f', help='Run one f value only', type=float)
parser.add_argument('--save_agents', help='Save the agents datacollector')
args = parser.parse_args()
test = args.test; profile=args.profile; save_agents = args.save_agents; paramsf = args.paramsf
run_one_f = args.run_one_f




all_f0_f1 = [0.45,0.55,0.65,0.6,0.7,0.75,0.8,0.85,0.9,0.4,0.5,0.3,0.2]
    


factor = "f0"


# test
num_steps = 1
# test
n_repeats=1

params_new = {
    'b': [1],
    'alpha':[1],
        #'alpha':[0,0.4,0.2,0.6,0.8,1], 
        #'b':[0.3,0.2,0.1,0.5,0.6,0.2,0.4],
        #'alpha':[0.2,0.4],
        #'radius': [3,6,9],
        'residential_steps': [100,0],
       'T': [0.75,0.8,0.85]
}

if run_one_f:
    all_f0_f1 = [run_one_f]
    n_repeats=1




if paramsf == "parameters_baseline":
    params = copy(parameters_baseline)
    params_new = {
    'b': [1],
    'alpha':[1],
    'residential_steps': [100,0],
    'sigma': [0.3,0.4],
    'T': [0.75,0.8,0.85]
}

elif paramsf == "parameters":
    params = copy(parameters)
    params_new = {
     'b':[0.0,0.15,0.1],
     'alpha':[0.25],
     'symmetric_positions':[True],
     #'radius': [3,6,9],
     'residential_steps': [100,0],
     #'temp': [0.1,0.01],
     #'sigma': [0.3,0.4],
     'T': [0.75]
}


if test:
    n_repeats=1
    all_f0_f1 = [0.7]
    num_steps=1
    params_new={
        'residential_steps': [1],
        'height': [54],
        'width': [54]
    }

else: 
    print("params file not valid")
    sys.exit()

if run_one_f:
    all_f0_f1 = [run_one_f]
    n_repeats=1




for i in range(0,n_repeats):
    for key in params_new:
        params[key] =  params_new[key]


    keys = list(params)
    for values in itertools.product(*map(params.get, keys)):
        all_models_df = run_simulation(params=dict(zip(keys,values)),all_f0_f1=all_f0_f1,num_steps=num_steps )







