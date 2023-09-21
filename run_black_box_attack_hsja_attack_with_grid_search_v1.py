"""
Runing the proposed Paret Set Learning (PSL) method on 15 test problems.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import numpy as np
import torch
import pickle
import json
from attack_imagenet_hsja import get_problem
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

config_file_name = 'config-jsons/imagenet_hsja_l2_config.json'
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

problem = get_problem('hsja_attack',config)
n_dim = problem.n_dim
n_obj = problem.n_obj

x_init= np.concatenate((np.transpose(np.array([[0.1,0.2,0.3,0.4,0.5]])),np.random.rand(5,2)),axis=1)
print(x_init)
y_init = problem.evaluate(torch.from_numpy(x_init).to(device))
print(x_init)
print(y_init)
pickle.dump(x_init,open('B-Box-eval-hsja-attack/hyp_search_eps.x.pkl','wb'))
pickle.dump(y_init,open('B-Box-eval-hsja-attack/hyp_search_eps.y.pkl','wb'))

