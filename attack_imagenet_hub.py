from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
'''

This file is copied from the following source:
link: https://github.com/ash-aldujaili/blackbox-adv-examples-signhunter/blob/master/src/attacks/blackbox/run.attack.py

@inproceedings{
al-dujaili2020sign,
title={Sign Bits Are All You Need for Black-Box Attacks},
author={Abdullah Al-Dujaili and Una-May O'Reilly},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=SygW0TEFwH}
}

The original license is placed at the end of this file.

basic structure for main:
    1. config args, save_path
    2. set the black-box attack on ImageNet
    3. set the device, model, criterion, training schedule
    4. start the attack process and get labels
    5. save attack result
    
'''

"""
Script for running black-box attacks
"""

import json
import math
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="2"
import time
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

import numpy as np
import pandas as pd
import tensorflow as tf
import torch

from datasets_cs.dataset import Dataset
from utils.compute import tf_nsign, sign
from utils.misc import config_path_join, src_path_join, create_dir, get_dataset_shape
from utils.model_loader import load_torch_models, load_torch_models_imagesub

from attacks.score.nes_attack import NESAttack
from attacks.score.bandit_attack import BanditAttack
from attacks.score.zo_sign_sgd_attack import ZOSignSGDAttack
from attacks.score.sign_attack import SignAttack
from attacks.score.simple_attack import SimpleAttack
from attacks.score.square_attack import SquareAttack
from attacks.score.parsimonious_attack import ParsimoniousAttack
#from attacks.score.dpd_attack import DPDAttack

from attacks.decision.sign_opt_attack import SignOPTAttack
from attacks.decision.hsja_attack import HSJAttack
from attacks.decision.geoda_attack import GeoDAttack
from attacks.decision.opt_attack import OptAttack
from attacks.decision.evo_attack import EvolutionaryAttack
from attacks.decision.sign_flip_attack import SignFlipAttack
from attacks.decision.rays_attack import RaySAttack
from attacks.decision.boundary_attack import BoundaryAttack

from utils.bounds import hb_p_value
from autoattack.autoattack import AutoAttack


# For l_2 NES attack, the range of epsilon is [0,5], fd_eta is [0,5], lr is [0,5]


def get_problem(name, *args, **kwargs):
    name = name.lower()
    
    PROBLEM = {
        'nes_attack': NES_Attack,
        'square_attack' : Square_Attack,
        'hsja_attack' : HSJA_Attack,
        'geoda_attack' : GeoDA_Attack,
        'auto_attack' : Auto_Attack
 }
    if name not in PROBLEM:
        raise Exception("Problem not found.")
    
    return PROBLEM[name](*args, **kwargs)

def cw_loss(logit, label, target=False):
    if target:
        # targeted cw loss: logit_t - max_{i\neq t}logit_i
        _, argsort = logit.sort(dim=1, descending=True)
        target_is_max = argsort[:, 0].eq(label)
        second_max_index = target_is_max.long() * argsort[:, 1] + (~ target_is_max).long() * argsort[:, 0]
        target_logit = logit[torch.arange(logit.shape[0]), label]
        second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
        return target_logit - second_max_logit 
    else:
        # untargeted cw loss: max_{i\neq y}logit_i - logit_y
        _, argsort = logit.sort(dim=1, descending=True)
        gt_is_max = argsort[:, 0].eq(label)
        second_max_index = gt_is_max.long() * argsort[:, 1] + (~gt_is_max).long() * argsort[:, 0]
        gt_logit = logit[torch.arange(logit.shape[0]), label]
        second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
        return second_max_logit - gt_logit

def xent_loss(logit, label, target=False):
    if not target:
        return torch.nn.CrossEntropyLoss(reduction='none')(logit, label)                
    else:
        return -torch.nn.CrossEntropyLoss(reduction='none')(logit, label)                

        # criterion = torch.nn.CrossEntropyLoss(reduce=False)
        # criterion = xent_loss



def get_label(target_type):
    _, logit = model(torch.FloatTensor(x_batch.transpose(0,3,1,2)))
    if target_type == 'random':
        label = torch.randint(low=0, high=logit.shape[1], size=label.shape).long().cuda()
    elif target_type == 'least_likely':
        label = logit.argmin(dim=1) 
    elif target_type == 'most_likely':
        label = torch.argsort(logit, dim=1,descending=True)[:,1]
    elif target_type == 'median':
        label = torch.argsort(logit, dim=1,descending=True)[:,4]
    elif 'label' in target_type:
        label = torch.ones_like(y_batch) * int(target_type[5:])
    return label.detach()


class Auto_Attack():
    def __init__(self, config, n_dim=3):
        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]
        self.config = config
        self.dset = Dataset(config['dset_name'], config)
        self.alpha = config['acc_drop_alpha']

        model_name = self.config['modeln']
        self.batch_size = self.config['attack_config']['batch_size']
        
        if self.config['dset_name'] == 'imagenet_sub':
            self.model = load_torch_models_imagesub(model_name)
        else:
            self.model = load_torch_models(model_name)
        self.criterion = cw_loss
        self.nadir_point = [1, 1]
    
    def evaluate(self,x_norm):
        x = x_norm
        #x = (x_norm-0.0) * 5.0
        #x[:,1] = (x_norm[:,1]-0.0)*1.0+1e-6
        #x[:,2] = (x_norm[:,2]-0.0)*3.0

        print(x)



        if 'gpu' in self.config['device'] and torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        num_samples = len(x)

        objs = torch.zeros((num_samples,self.n_obj)).cuda()
        for i in range(num_samples):
            objs_sample = self.evaluate_single(x[i,:].cpu().numpy())
            objs[i,:] = objs_sample.cuda()
        return objs
        
    def evaluate_single(self, x):
        # x[0]: epsilon
        # x[1]: fd_eta
        # x[2]: lr
        # x[3]: q

        if x[0]<=1e-7:
            objs = torch.tensor([-0.0, x[0]])
            print(objs)
            return objs
        # epsilon, p, max_queries, sub_dim, tol, alpha, mu, search_space, grad_estimator_batch_size, lb, ub, batch_size, sigma
        self.attacker = AutoAttack(self.model, norm='L'+self.config['attack_config']['p'], eps=x[0], version='standard')
        

        with torch.no_grad():

            epsilon=x[0]

            target = self.config['target']

            # Iterate over the samples batch-by-batch
            num_eval_examples = self.config['num_eval_examples']
            eval_batch_size = self.config['attack_config']['batch_size']
            num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

            print('Iterating over {} batches'.format(num_batches))
            start_time = time.time()

            adv_acc = 0
            clean_acc = 0

            for ibatch in range(num_batches):
                bstart = ibatch * eval_batch_size
                bend = min(bstart + eval_batch_size, num_eval_examples)
                print('batch size: {}: ({}, {})'.format(bend - bstart, bstart, bend))

                x_batch, y_batch = self.dset.get_eval_data(bstart, bend)
                y_batch = torch.LongTensor(y_batch).cuda()
                x_ori = torch.FloatTensor(x_batch.copy().transpose(0,3,1,2)).cuda()

                x_adv = self.attacker.run_standard_evaluation(x_ori, y_batch, bs=len(x_batch))

                adv_acc += self.attacker.clean_accuracy(x_adv,y_batch,bs=len(x_batch))*len(x_batch)
                clean_acc += self.attacker.clean_accuracy(x_ori,y_batch,bs=len(x_batch))*len(x_batch)


        self.attacker.total_failures = adv_acc
        self.attacker.total_successes = clean_acc - adv_acc


        print(self.attacker.total_failures)
        print(self.attacker.total_successes)
        clean_acc = (self.attacker.total_failures+self.attacker.total_successes)/num_eval_examples

        adv_acc = self.attacker.total_failures/num_eval_examples

        acc_drop_relative = (clean_acc-adv_acc)/clean_acc

        #print(acc_drop_relative)
        #print(num_eval_examples)

        # use the correct prediction sample number as the n value

        p_value = hb_p_value(acc_drop_relative, self.attacker.total_failures+self.attacker.total_successes, self.alpha)
        
        objs = torch.tensor([p_value, epsilon, acc_drop_relative])

        print(objs)

        return objs



class GeoDA_Attack():
    def __init__(self, config, n_dim=3):
        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]
        self.config = config
        self.dset = Dataset(config['dset_name'], config)
        self.alpha = config['acc_drop_alpha']

        model_name = self.config['modeln']
        self.batch_size = self.config['attack_config']['batch_size']
        
        if self.config['dset_name'] == 'imagenet_sub':
            self.model = load_torch_models_imagesub(model_name)
        else:
            self.model = load_torch_models(model_name)
        self.criterion = cw_loss
        self.nadir_point = [1, 1]
    
    def evaluate(self,x_norm):
        x = x_norm
        #x = (x_norm-0.0) * 5.0
        #x[:,1] = (x_norm[:,1]-0.0)*1.0+1e-6
        #x[:,2] = (x_norm[:,2]-0.0)*3.0

        print(x)



        if 'gpu' in self.config['device'] and torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        num_samples = len(x)

        objs = torch.zeros((num_samples,self.n_obj)).cuda()
        for i in range(num_samples):
            objs_sample = self.evaluate_single(x[i,:].cpu().numpy())
            objs[i,:] = objs_sample.cuda()
        return objs
        
    def evaluate_single(self, x):
        # x[0]: epsilon
        # x[1]: fd_eta
        # x[2]: lr
        # x[3]: q

        if x[0]<=1e-7:
            objs = torch.tensor([-0.0, x[0]])
            print(objs)
            return objs
        # epsilon, p, max_queries, sub_dim, tol, alpha, mu, search_space, grad_estimator_batch_size, lb, ub, batch_size, sigma
        self.attacker = GeoDAttack(max_queries=self.config['attack_config']['max_queries'], sub_dim=self.config['attack_config']['sub_dim'],
                                  epsilon=x[0], p=self.config['attack_config']['p'], tol=self.config['attack_config']['tol'],alpha=self.config['attack_config']['alpha'],
                                    mu=self.config['attack_config']['mu'], search_space=self.config['attack_config']['search_space'], grad_estimator_batch_size = self.config['attack_config']['grad_estimator_batch_size'],
                                  lb=self.dset.min_value, ub=self.dset.max_value, batch_size=self.config['attack_config']['batch_size'], sigma=self.config['attack_config']['sigma'])
        with torch.no_grad():

            epsilon=x[0]

            target = self.config['target']

            # Iterate over the samples batch-by-batch
            num_eval_examples = self.config['num_eval_examples']
            eval_batch_size = self.config['attack_config']['batch_size']
            num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

            print('Iterating over {} batches'.format(num_batches))
            start_time = time.time()

            for ibatch in range(num_batches):
                bstart = ibatch * eval_batch_size
                bend = min(bstart + eval_batch_size, num_eval_examples)
                print('batch size: {}: ({}, {})'.format(bend - bstart, bstart, bend))

                x_batch, y_batch = self.dset.get_eval_data(bstart, bend)
                y_batch = torch.LongTensor(y_batch).cuda()
                x_ori = torch.FloatTensor(x_batch.copy().transpose(0,3,1,2)).cuda()

                if target:
                    y_batch = get_label(self.config["target_type"])


                if self.config['attack_name'] in ["SignOPTAttack","HSJAttack","GeoDAttack","OptAttack","EvolutionaryAttack",
                                            "SignFlipAttack","RaySAttack","BoundaryAttack"]:
                    logs_dict = self.attacker.run(x_batch, y_batch, self.model, target, self.dset)

                else:
                    def loss_fct(xs, es = False):
                        if type(xs) is torch.Tensor:
                            x_eval = xs.permute(0,3,1,2)
                        else:
                            x_eval = torch.FloatTensor(xs.transpose(0,3,1,2)).cuda()
                        x_eval = torch.clamp(x_eval - x_ori, -epsilon, epsilon) + x_ori
                        x_eval = torch.clamp(x_eval, 0, 1)
                        #---------------------#
                        # sigma = config["sigma"]
                        #---------------------#
                        # x_eval = x_eval + sigma * torch.randn_like(x_eval)
                        y_logit = self.model(x_eval)
                        loss = self.criterion(y_logit, y_batch, target)
                        #print(loss)
                        if es:
                            y_logit = y_logit.detach()
                            correct = torch.argmax(y_logit, dim=1) == y_batch
                            if target:
                                return correct, loss.detach()
                            else:
                                return ~correct, loss.detach()
                        else: 
                            return loss.detach()

                    def early_stop_crit_fct(xs):
                        if type(xs) is torch.Tensor:
                            x_eval = xs.permute(0,3,1,2).cuda()
                        else:
                            x_eval = torch.FloatTensor(xs.transpose(0,3,1,2)).cuda()
                        x_eval = torch.clamp(x_eval - x_ori, -epsilon, epsilon) + x_ori
                        x_eval = torch.clamp(x_eval, 0, 1)
                        #---------------------#
                        # sigma = config["sigma"]
                        #---------------------#
                        # x_eval = x_eval + sigma * torch.randn_like(x_eval)
                        y_logit = self.model(x_eval)
                        y_logit = y_logit.detach()
                        correct = torch.argmax(y_logit, dim=1) == y_batch
                        if target:
                            return correct
                        else:
                            # return a mask, 1 is incorrect, 0 is correct
                            return ~correct
                    logs_dict = self.attacker.run(x_batch, loss_fct, early_stop_crit_fct)
        
        print("Batches done after {} s".format(time.time() - start_time))



        print(self.attacker.total_failures)
        print(self.attacker.total_successes)
        clean_acc = (self.attacker.total_failures+self.attacker.total_successes)/num_eval_examples

        adv_acc = self.attacker.total_failures/num_eval_examples

        acc_drop_relative = (clean_acc-adv_acc)/clean_acc

        #print(acc_drop_relative)
        #print(num_eval_examples)

        # use the correct prediction sample number as the n value

        p_value = hb_p_value(acc_drop_relative.cpu().numpy(), self.attacker.total_failures.cpu().numpy()+self.attacker.total_successes.cpu().numpy(), self.alpha)
        
        objs = torch.tensor([p_value, epsilon, acc_drop_relative.cpu().numpy()])

        print(objs)

        return objs





class HSJA_Attack():
    def __init__(self, config, n_dim=3):
        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]
        self.config = config
        self.dset = Dataset(config['dset_name'], config)
        self.alpha = config['acc_drop_alpha']

        model_name = self.config['modeln']
        self.batch_size = self.config['attack_config']['batch_size']
        
        if self.config['dset_name'] == 'imagenet_sub':
            self.model = load_torch_models_imagesub(model_name)
        else:
            self.model = load_torch_models(model_name)
        self.criterion = cw_loss
        self.nadir_point = [1, 1]
    
    def evaluate(self,x_norm):
        x = x_norm
        #x = (x_norm-0.0) * 5.0
        #x[:,1] = (x_norm[:,1]-0.0)*1.0+1e-6
        #x[:,2] = (x_norm[:,2]-0.0)*3.0

        print(x)



        if 'gpu' in self.config['device'] and torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        num_samples = len(x)

        objs = torch.zeros((num_samples,self.n_obj)).cuda()
        for i in range(num_samples):
            objs_sample = self.evaluate_single(x[i,:].cpu().numpy())
            objs[i,:] = objs_sample.cuda()
        return objs
        
    def evaluate_single(self, x):
        # x[0]: epsilon
        # x[1]: fd_eta
        # x[2]: lr
        # x[3]: q

        if x[0]<=1e-7:
            objs = torch.tensor([-0.0, x[0]])
            print(objs)
            return objs

        self.attacker = HSJAttack(max_queries=self.config['attack_config']['max_queries'], max_num_evals=self.config['attack_config']['max_num_evals'],
                                  epsilon=x[0], p=self.config['attack_config']['p'], gamma=self.config['attack_config']['gamma'],stepsize_search=self.config['attack_config']['stepsize_search'],
                                    EOT=self.config['attack_config']['EOT'], init_num_evals=self.config['attack_config']['init_num_evals'], sigma = self.config['attack_config']['sigma'],
                                  lb=self.dset.min_value, ub=self.dset.max_value, batch_size=self.config['attack_config']['batch_size'])
        with torch.no_grad():

            epsilon=x[0]

            target = self.config['target']

            # Iterate over the samples batch-by-batch
            num_eval_examples = self.config['num_eval_examples']
            eval_batch_size = self.config['attack_config']['batch_size']
            num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

            print('Iterating over {} batches'.format(num_batches))
            start_time = time.time()

            for ibatch in range(num_batches):
                bstart = ibatch * eval_batch_size
                bend = min(bstart + eval_batch_size, num_eval_examples)
                print('batch size: {}: ({}, {})'.format(bend - bstart, bstart, bend))

                x_batch, y_batch = self.dset.get_eval_data(bstart, bend)
                y_batch = torch.LongTensor(y_batch).cuda()
                x_ori = torch.FloatTensor(x_batch.copy().transpose(0,3,1,2)).cuda()

                if target:
                    y_batch = get_label(self.config["target_type"])


                if self.config['attack_name'] in ["SignOPTAttack","HSJAttack","GeoDAttack","OptAttack","EvolutionaryAttack",
                                            "SignFlipAttack","RaySAttack","BoundaryAttack"]:
                    logs_dict = self.attacker.run(x_batch, y_batch, self.model, target, self.dset)

                else:
                    def loss_fct(xs, es = False):
                        if type(xs) is torch.Tensor:
                            x_eval = xs.permute(0,3,1,2)
                        else:
                            x_eval = torch.FloatTensor(xs.transpose(0,3,1,2)).cuda()
                        x_eval = torch.clamp(x_eval - x_ori, -epsilon, epsilon) + x_ori
                        x_eval = torch.clamp(x_eval, 0, 1)
                        #---------------------#
                        # sigma = config["sigma"]
                        #---------------------#
                        # x_eval = x_eval + sigma * torch.randn_like(x_eval)
                        y_logit = self.model(x_eval)
                        loss = self.criterion(y_logit, y_batch, target)
                        #print(loss)
                        if es:
                            y_logit = y_logit.detach()
                            correct = torch.argmax(y_logit, dim=1) == y_batch
                            if target:
                                return correct, loss.detach()
                            else:
                                return ~correct, loss.detach()
                        else: 
                            return loss.detach()

                    def early_stop_crit_fct(xs):
                        if type(xs) is torch.Tensor:
                            x_eval = xs.permute(0,3,1,2).cuda()
                        else:
                            x_eval = torch.FloatTensor(xs.transpose(0,3,1,2)).cuda()
                        x_eval = torch.clamp(x_eval - x_ori, -epsilon, epsilon) + x_ori
                        x_eval = torch.clamp(x_eval, 0, 1)
                        #---------------------#
                        # sigma = config["sigma"]
                        #---------------------#
                        # x_eval = x_eval + sigma * torch.randn_like(x_eval)
                        y_logit = self.model(x_eval)
                        y_logit = y_logit.detach()
                        correct = torch.argmax(y_logit, dim=1) == y_batch
                        if target:
                            return correct
                        else:
                            # return a mask, 1 is incorrect, 0 is correct
                            return ~correct
                    logs_dict = self.attacker.run(x_batch, loss_fct, early_stop_crit_fct)
        
        print("Batches done after {} s".format(time.time() - start_time))



        print(self.attacker.total_failures)
        print(self.attacker.total_successes)
        clean_acc = (self.attacker.total_failures+self.attacker.total_successes)/num_eval_examples

        adv_acc = self.attacker.total_failures/num_eval_examples

        acc_drop_relative = (clean_acc-adv_acc)/clean_acc

        #print(acc_drop_relative)
        #print(num_eval_examples)

        # use the correct prediction sample number as the n value

        p_value = hb_p_value(acc_drop_relative.cpu().numpy(), self.attacker.total_failures.cpu().numpy()+self.attacker.total_successes.cpu().numpy(), self.alpha)
        
        objs = torch.tensor([p_value, epsilon, acc_drop_relative.cpu().numpy()])

        print(objs)

        return objs





class NES_Attack():
    def __init__(self, config, n_dim=3):
        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]
        self.config = config
        self.dset = Dataset(config['dset_name'], config)
        self.alpha = config['acc_drop_alpha']

        model_name = self.config['modeln']
        self.batch_size = self.config['attack_config']['batch_size']
        
        if self.config['dset_name'] == 'imagenet_sub':
            self.model = load_torch_models_imagesub(model_name)
        else:
            self.model = load_torch_models(model_name)
        self.criterion = cw_loss
        self.nadir_point = [1, 1]
    
    def evaluate(self,x_norm):
        x = x_norm
        #x = (x_norm-0.0) * 5.0
        #x[:,1] = (x_norm[:,1]-0.0)*1.0+1e-6
        #x[:,2] = (x_norm[:,2]-0.0)*3.0

        print(x)



        if 'gpu' in self.config['device'] and torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        num_samples = len(x)

        objs = torch.zeros((num_samples,self.n_obj)).cuda()
        for i in range(num_samples):
            objs_sample = self.evaluate_single(x[i,:].cpu().numpy())
            objs[i,:] = objs_sample.cuda()
        return objs
        
    def evaluate_single(self, x):
        # x[0]: epsilon
        # x[1]: fd_eta
        # x[2]: lr
        # x[3]: q

        if x[0]<=1e-7:
            objs = torch.tensor([-0.0, x[0]])
            print(objs)
            return objs

        self.attacker = NESAttack(max_loss_queries=self.config['attack_config']['max_loss_queries'], 
                                  epsilon=x[0], p=self.config['attack_config']['p'], fd_eta=x[1], lr=x[2], q=15, 
                                  lb=self.dset.min_value, ub=self.dset.max_value, batch_size=self.config['attack_config']['batch_size'], name='NES')
        with torch.no_grad():

            epsilon=x[0]

            target = self.config['target']

            # Iterate over the samples batch-by-batch
            num_eval_examples = self.config['num_eval_examples']
            eval_batch_size = self.config['attack_config']['batch_size']
            num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

            print('Iterating over {} batches'.format(num_batches))
            start_time = time.time()

            for ibatch in range(num_batches):
                bstart = ibatch * eval_batch_size
                bend = min(bstart + eval_batch_size, num_eval_examples)
                print('batch size: {}: ({}, {})'.format(bend - bstart, bstart, bend))

                x_batch, y_batch = self.dset.get_eval_data(bstart, bend)
                y_batch = torch.LongTensor(y_batch).cuda()
                x_ori = torch.FloatTensor(x_batch.copy().transpose(0,3,1,2)).cuda()

                if target:
                    y_batch = get_label(self.config["target_type"])


                if self.config['attack_name'] in ["SignOPTAttack","HSJAttack","GeoDAttack","OptAttack","EvolutionaryAttack",
                                            "SignFlipAttack","RaySAttack","BoundaryAttack"]:
                    logs_dict = self.attacker.run(x_batch, y_batch, self.model, target, self.dset)

                else:
                    def loss_fct(xs, es = False):
                        if type(xs) is torch.Tensor:
                            x_eval = xs.permute(0,3,1,2)
                        else:
                            x_eval = torch.FloatTensor(xs.transpose(0,3,1,2)).cuda()
                        x_eval = torch.clamp(x_eval - x_ori, -epsilon, epsilon) + x_ori
                        x_eval = torch.clamp(x_eval, 0, 1)
                        #---------------------#
                        # sigma = config["sigma"]
                        #---------------------#
                        # x_eval = x_eval + sigma * torch.randn_like(x_eval)
                        y_logit = self.model(x_eval)
                        loss = self.criterion(y_logit, y_batch, target)
                        #print(loss)
                        if es:
                            y_logit = y_logit.detach()
                            correct = torch.argmax(y_logit, dim=1) == y_batch
                            if target:
                                return correct, loss.detach()
                            else:
                                return ~correct, loss.detach()
                        else: 
                            return loss.detach()

                    def early_stop_crit_fct(xs):
                        if type(xs) is torch.Tensor:
                            x_eval = xs.permute(0,3,1,2).cuda()
                        else:
                            x_eval = torch.FloatTensor(xs.transpose(0,3,1,2)).cuda()
                        x_eval = torch.clamp(x_eval - x_ori, -epsilon, epsilon) + x_ori
                        x_eval = torch.clamp(x_eval, 0, 1)
                        #---------------------#
                        # sigma = config["sigma"]
                        #---------------------#
                        # x_eval = x_eval + sigma * torch.randn_like(x_eval)
                        y_logit = self.model(x_eval)
                        y_logit = y_logit.detach()
                        correct = torch.argmax(y_logit, dim=1) == y_batch
                        if target:
                            return correct
                        else:
                            # return a mask, 1 is incorrect, 0 is correct
                            return ~correct
                    logs_dict = self.attacker.run(x_batch, loss_fct, early_stop_crit_fct)
        
        print("Batches done after {} s".format(time.time() - start_time))



        
        clean_acc = (self.attacker.total_failures+self.attacker.total_successes)/num_eval_examples

        adv_acc = self.attacker.total_failures/num_eval_examples

        acc_drop_relative = (clean_acc-adv_acc)/clean_acc

        #print(acc_drop_relative)
        #print(num_eval_examples)

        # use the correct prediction sample number as the n value

        p_value = hb_p_value(acc_drop_relative.cpu().numpy(), self.attacker.total_failures.cpu().numpy()+self.attacker.total_successes.cpu().numpy(), self.alpha)
        
        objs = torch.tensor([p_value, epsilon, acc_drop_relative.cpu().numpy()])

        print(objs)

        return objs


class Square_Attack():
    def __init__(self, config, n_dim=3):
        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]
        self.config = config
        self.dset = Dataset(config['dset_name'], config)
        self.alpha = config['acc_drop_alpha']

        model_name = self.config['modeln']
        self.batch_size = self.config['attack_config']['batch_size']
        
        if self.config['dset_name'] == 'imagenet_sub':
            self.model = load_torch_models_imagesub(model_name)
        else:
            self.model = load_torch_models(model_name)
        self.criterion = cw_loss
        self.nadir_point = [1, 1]
    
    def evaluate(self,x_norm):
        x = x_norm
        #x = (x_norm-0.0) * 5.0
        #x[:,1] = (x_norm[:,1]-0.0)*1.0+1e-6
        #x[:,2] = (x_norm[:,2]-0.0)*3.0

        print(x)



        if 'gpu' in self.config['device'] and torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        num_samples = len(x)

        objs = torch.zeros((num_samples,self.n_obj)).cuda()
        for i in range(num_samples):
            objs_sample = self.evaluate_single(x[i,:].cpu().numpy())
            objs[i,:] = objs_sample.cuda()
        return objs
        
    def evaluate_single(self, x):
        # x[0]: epsilon
        # x[1]: fd_eta
        # x[2]: lr
        # x[3]: q

        if x[0]<=1e-7:
            objs = torch.tensor([-0.0, x[0]])
            print(objs)
            return objs

        self.attacker = SquareAttack(max_loss_queries=self.config['attack_config']['max_loss_queries'], 
                                  epsilon=x[0], p=self.config['attack_config']['p'], p_init=self.config['attack_config']['p_init'], 
                                  lb=self.dset.min_value, ub=self.dset.max_value, batch_size=self.config['attack_config']['batch_size'], name='SquareAttack')
        with torch.no_grad():

            epsilon=x[0]

            target = self.config['target']

            # Iterate over the samples batch-by-batch
            num_eval_examples = self.config['num_eval_examples']
            eval_batch_size = self.config['attack_config']['batch_size']
            num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

            print('Iterating over {} batches'.format(num_batches))
            start_time = time.time()

            for ibatch in range(num_batches):
                bstart = ibatch * eval_batch_size
                bend = min(bstart + eval_batch_size, num_eval_examples)
                print('batch size: {}: ({}, {})'.format(bend - bstart, bstart, bend))

                x_batch, y_batch = self.dset.get_eval_data(bstart, bend)
                y_batch = torch.LongTensor(y_batch).cuda()
                x_ori = torch.FloatTensor(x_batch.copy().transpose(0,3,1,2)).cuda()

                if target:
                    y_batch = get_label(self.config["target_type"])


                if self.config['attack_name'] in ["SignOPTAttack","HSJAttack","GeoDAttack","OptAttack","EvolutionaryAttack",
                                            "SignFlipAttack","RaySAttack","BoundaryAttack"]:
                    logs_dict = self.attacker.run(x_batch, y_batch, self.model, target, self.dset)

                else:
                    def loss_fct(xs, es = False):
                        if type(xs) is torch.Tensor:
                            x_eval = xs.permute(0,3,1,2)
                        else:
                            x_eval = torch.FloatTensor(xs.transpose(0,3,1,2)).cuda()
                        x_eval = torch.clamp(x_eval - x_ori, -epsilon, epsilon) + x_ori
                        x_eval = torch.clamp(x_eval, 0, 1)
                        #---------------------#
                        # sigma = config["sigma"]
                        #---------------------#
                        # x_eval = x_eval + sigma * torch.randn_like(x_eval)
                        y_logit = self.model(x_eval)
                        loss = self.criterion(y_logit, y_batch, target)
                        #print(loss)
                        if es:
                            y_logit = y_logit.detach()
                            correct = torch.argmax(y_logit, dim=1) == y_batch
                            if target:
                                return correct, loss.detach()
                            else:
                                return ~correct, loss.detach()
                        else: 
                            return loss.detach()

                    def early_stop_crit_fct(xs):
                        if type(xs) is torch.Tensor:
                            x_eval = xs.permute(0,3,1,2).cuda()
                        else:
                            x_eval = torch.FloatTensor(xs.transpose(0,3,1,2)).cuda()
                        x_eval = torch.clamp(x_eval - x_ori, -epsilon, epsilon) + x_ori
                        x_eval = torch.clamp(x_eval, 0, 1)
                        #---------------------#
                        # sigma = config["sigma"]
                        #---------------------#
                        # x_eval = x_eval + sigma * torch.randn_like(x_eval)
                        y_logit = self.model(x_eval)
                        y_logit = y_logit.detach()
                        correct = torch.argmax(y_logit, dim=1) == y_batch
                        if target:
                            return correct
                        else:
                            # return a mask, 1 is incorrect, 0 is correct
                            return ~correct
                    logs_dict = self.attacker.run(x_batch, loss_fct, early_stop_crit_fct)
        
        print("Batches done after {} s".format(time.time() - start_time))



        
        clean_acc = (self.attacker.total_failures+self.attacker.total_successes)/num_eval_examples

        adv_acc = self.attacker.total_failures/num_eval_examples

        acc_drop_relative = (clean_acc-adv_acc)/clean_acc

        #print(acc_drop_relative)
        #print(num_eval_examples)

        # use the correct prediction sample number as the n value

        p_value = hb_p_value(acc_drop_relative.cpu().numpy(), self.attacker.total_failures.cpu().numpy()+self.attacker.total_successes.cpu().numpy(), self.alpha)
        
        objs = torch.tensor([p_value, epsilon, acc_drop_relative.cpu().numpy()])

        print(objs)

        return objs
