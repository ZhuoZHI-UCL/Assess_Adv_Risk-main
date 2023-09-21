"""
Runing the proposed Paret Set Learning (PSL) method on 15 test problems.
"""
import os

import numpy as np
import torch
import pickle
import json
from attack_imagenet_hub import get_problem
import time
import argparse
from lhs import lhs
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from mobo.surrogate_model import GaussianProcess
from mobo.transformation import StandardTransform

from model import ParetoSetModel

# -----------------------------------------------------------------------------
# list of 15 test problems, which are defined in problem.py
#ins_list = ['f1','f2','f3','f4','f5','f6',
#            'vlmop1','vlmop2', 'vlmop3', 'dtlz2',
#            're21', 're23', 're33','re36','re37']
os.environ["CUDA_VISIBLE_DEVICES"]="2"
screen = 'L_AA_Linf' 
"""
S_AA_L2   B_AA_L2     L_AA_L2 
S_AA_Linf B_AA_Linf   L_AA_Linf 
"""
#--------------------------------------------------------------------------
if screen == 'S_AA_L2':
    config_file_name = 'config-jsons/imagenet_AA_l2_eps_0.1_config.json'
    with open(config_file_name) as config_file:
        config = json.load(config_file)
    config["modeln"] = "vit_small_patch16_224"
    epsilon_candidate = np.transpose(np.array([[0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027 ,0.028, 0.029, 0.03]]))
elif screen == 'S_AA_Linf':
    config_file_name = 'config-jsons/imagenet_AA_linf_eps_0.1_config.json'
    with open(config_file_name) as config_file:
        config = json.load(config_file)
    config["modeln"] = "vit_small_patch16_224"
    epsilon_candidate = np.transpose(np.array([[0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006 ,0.00007, 0.00008, 0.00009, 0.0001]]))
#--------------------------------------------------------------------------
elif screen == 'B_AA_L2':
    config_file_name = 'config-jsons/imagenet_AA_l2_eps_0.1_config.json'
    with open(config_file_name) as config_file:
        config = json.load(config_file)
    config["modeln"] = "vit_base_patch16_224"
    epsilon_candidate = np.transpose(np.array([[0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027 ,0.028, 0.029, 0.03]]))
elif screen == 'B_AA_Linf':
    config_file_name = 'config-jsons/imagenet_AA_linf_eps_0.1_config.json'
    with open(config_file_name) as config_file:
        config = json.load(config_file)
    config["modeln"] = "vit_base_patch16_224"
    epsilon_candidate = np.transpose(np.array([[0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006 ,0.00007, 0.00008, 0.00009, 0.0001]]))
    
#--------------------------------------------------------------------------  
elif screen == 'L_AA_L2':
    config_file_name = 'config-jsons/imagenet_AA_l2_eps_0.1_config.json'
    with open(config_file_name) as config_file:
        config = json.load(config_file)
    config["modeln"] = "vit_large_patch16_224"
    epsilon_candidate = np.transpose(np.array([[0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027 ,0.028, 0.029, 0.03]]))
elif screen == 'L_AA_Linf':
    config_file_name = 'config-jsons/imagenet_AA_linf_eps_0.1_config.json'
    with open(config_file_name) as config_file:
        config = json.load(config_file)
    config["modeln"] = "vit_large_patch16_224"
    epsilon_candidate = np.transpose(np.array([[0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006 ,0.00007, 0.00008, 0.00009, 0.0001]]))


problem = get_problem('auto_attack',config)
n_dim = problem.n_dim
n_obj = problem.n_obj
device = 'cuda'
# epsilon_candidate
x_init= np.concatenate((epsilon_candidate,np.random.rand(10,2)),axis=1)
print(x_init)
y_init = problem.evaluate(torch.from_numpy(x_init).to(device))
print(x_init)
print(y_init)
pickle.dump(x_init,open(f'B-Box-eval-square-attack/x_{screen}.pkl','wb'))
pickle.dump(y_init,open(f'B-Box-eval-square-attack/y_{screen}.pkl','wb'))

