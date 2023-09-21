"""
Runing the proposed Paret Set Learning (PSL) method on 15 test problems.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import torch
import pickle
import json
from attack_imagenet import get_problem
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
ins_list = ['NES_Attack']

config_file_name = 'config-jsons/imagenet_nes_l2_epsilon_1.0_config.json'
with open(config_file_name) as config_file:
    config = json.load(config_file)

print(config)


parser = argparse.ArgumentParser()
parser.add_argument('--coef_lcb', type=float, default=0.01)
parser.add_argument('--coef_guide', type=float, default=0.1)
parser.add_argument('--save_path', type=str, default='result.pkl')
args = parser.parse_args()


# number of independent runs
n_run = 1 #20 
# number of initialized solutions
n_init = 20 
# number of iterations, and batch size per iteration
n_iter = 10
n_sample = 5 

# PSL 
# number of learning steps
n_steps = 100
# number of sampled preferences per step
n_pref_update = 10 
# coefficient of LCB
coef_lcb = args.coef_lcb
# coefficient of guided search, focus the search around p=0.05
coef_guide = args.coef_guide
# number of sampled candidates on the approxiamte Pareto front
n_candidate = 1000 
# number of optional local search
n_local = 1
# device
device = 'cuda'
# -----------------------------------------------------------------------------

hv_list = {}

problem = get_problem('NES_Attack',config)
n_dim = problem.n_dim
n_obj = problem.n_obj
fd, lr = np.meshgrid(np.array([0.2, 0.4, 0.6]), np.array([0.2, 0.4, 0.8]))

fd=np.reshape(fd,(9,1))
lr=np.reshape(lr,(9,1))
#print(fd)
#print(lr)
x_init_hy = np.concatenate((fd,lr),axis=1)
x_init= np.concatenate((0.1*np.ones((9,1)),x_init_hy),axis=1)
print(x_init)
y_init = problem.evaluate(torch.from_numpy(x_init).to(device))
print(x_init)
print(y_init)
pickle.dump(x_init,open('B-Box-eval/hyp_search_eps_0.1.x.pkl','wb'))
pickle.dump(y_init,open('B-Box-eval/hyp_search_eps_0.1.y.pkl','wb'))

