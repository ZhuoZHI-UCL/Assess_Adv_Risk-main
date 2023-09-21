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
# 'config-jsons/imagenet_square_l2_eps_0.1_config.json'
# 0.1 0.2	0.3	0.4	0.5	0.6	0.7	0.8	0.9	1

# 'config-jsons/imagenet_square_linf_eps_0.1_config.json'
# 0.01	0.02	0.03	0.04	0.05	0.06	0.07	0.08	0.09	0.1
os.environ["CUDA_VISIBLE_DEVICES"]="0"
screen = 'L_Spuare_Linf' 
print(screen)
"""
VIT小          VIT中          VIT大
S_Spuare_L2   B_Spuare_L2     L_Spuare_L2 
S_Spuare_Linf B_Spuare_Linf   L_Spuare_Linf 
"""
#--------------------------------------------------------------------------
if screen == 'S_Spuare_L2':
    config_file_name = 'config-jsons/imagenet_square_l2_eps_0.1_config.json'
    with open(config_file_name) as config_file:
        config = json.load(config_file)
    config["modeln"] = "vit_small_patch16_224"
    epsilon_candidate = np.transpose(np.array([[1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8, 1.9, 2,2.1,2.2,2.3,2.4,2.5,2.6,2.7 ,2.8, 2.9, 3]]))
    x_init= np.concatenate((epsilon_candidate,np.random.rand(20,2)),axis=1)
elif screen == 'S_Spuare_Linf':
    config_file_name = 'config-jsons/imagenet_square_linf_eps_0.1_config.json'
    with open(config_file_name) as config_file:
        config = json.load(config_file)
    config["modeln"] = "vit_small_patch16_224"
    epsilon_candidate = np.transpose(np.array([[0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008 ,0.009,0.01]]))
    x_init= np.concatenate((epsilon_candidate,np.random.rand(10,2)),axis=1)
#--------------------------------------------------------------------------
elif screen == 'B_Spuare_L2':
    config_file_name = 'config-jsons/imagenet_square_l2_eps_0.1_config.json'
    with open(config_file_name) as config_file:
        config = json.load(config_file)
    config["modeln"] = "vit_base_patch16_224"
    epsilon_candidate = np.transpose(np.array([[1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8, 1.9, 2,2.1,2.2,2.3,2.4,2.5,2.6,2.7 ,2.8, 2.9, 3]]))
    x_init= np.concatenate((epsilon_candidate,np.random.rand(20,2)),axis=1)
elif screen == 'B_Spuare_Linf':
    config_file_name = 'config-jsons/imagenet_square_linf_eps_0.1_config.json'
    with open(config_file_name) as config_file:
        config = json.load(config_file)
    config["modeln"] = "vit_base_patch16_224"
    epsilon_candidate = np.transpose(np.array([[0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008 ,0.009,0.01]]))
    x_init= np.concatenate((epsilon_candidate,np.random.rand(10,2)),axis=1)
    
#--------------------------------------------------------------------------  
elif screen == 'L_Spuare_L2':
    config_file_name = 'config-jsons/imagenet_square_l2_eps_0.1_config.json'
    with open(config_file_name) as config_file:
        config = json.load(config_file)
    config["modeln"] = "vit_large_patch16_224"
    epsilon_candidate = np.transpose(np.array([[1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8, 1.9, 2,2.1,2.2,2.3,2.4,2.5,2.6,2.7 ,2.8, 2.9, 3]]))
    x_init= np.concatenate((epsilon_candidate,np.random.rand(20,2)),axis=1)
elif screen == 'L_Spuare_Linf':
    config_file_name = 'config-jsons/imagenet_square_linf_eps_0.1_config.json'
    with open(config_file_name) as config_file:
        config = json.load(config_file)
    config["modeln"] = "vit_large_patch16_224"
    epsilon_candidate = np.transpose(np.array([[0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008 ,0.009,0.01]]))
    x_init= np.concatenate((epsilon_candidate,np.random.rand(10,2)),axis=1)

print(config)

device = 'cuda'
# -----------------------------------------------------------------------------


problem = get_problem('square_attack',config)
n_dim = problem.n_dim
n_obj = problem.n_obj
    


print(x_init)
y_init = problem.evaluate(torch.from_numpy(x_init).to(device))
print(x_init)
print(y_init)
pickle.dump(x_init,open(f'B-Box-eval-square-attack/x_{screen}.pkl','wb'))
pickle.dump(y_init,open(f'B-Box-eval-square-attack/y_{screen}.pkl','wb'))

