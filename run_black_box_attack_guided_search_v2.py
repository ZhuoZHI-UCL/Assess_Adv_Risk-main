"""
Runing the proposed Paret Set Learning (PSL) method on 15 test problems.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
import torch
import pickle
import json
from attack_imagenet import get_problem
import time

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
coef_lcb = 0.1
# coefficient of guided search, focus the search around p=0.05
coef_guide = 0.5
# number of sampled candidates on the approxiamte Pareto front
n_candidate = 1000 
# number of optional local search
n_local = 1
# device
device = 'cuda'
# -----------------------------------------------------------------------------

hv_list = {}



for test_ins in ins_list:
    print(test_ins)
    
    # get problem info
    hv_all_value = np.zeros([n_run, n_iter])
    problem = get_problem(test_ins,config)
    n_dim = problem.n_dim
    n_obj = problem.n_obj

    ref_point = problem.nadir_point
    ref_point = [2.1*x for x in ref_point]
    load_init = True

    # pre-train the PS model to be uniform distribution


    
    # repeatedly run the algorithm n_run times
    for run_iter in range(n_run):
        
        # TODO: better I/O between torch and np
        # currently, the Pareto Set Model is on torch, and the Gaussian Process Model is on np 
        
        # initialize n_init solutions 
        print(run_iter)
        if load_init and run_iter==0:
            x_init = pickle.load(open('B-Box-eval/'+config_file_name.split('/')[1]+'.x.pkl','rb'))
            y_init = pickle.load(open('B-Box-eval/'+config_file_name.split('/')[1]+'.y.pkl','rb'))
        else:
            x_init = lhs(n_dim, n_init)
            y_init = problem.evaluate(torch.from_numpy(x_init).to(device))
            print('initial x')
            print(x_init)
            print('initial y')
            print(y_init)
            pickle.dump(x_init,open('B-Box-eval/'+config_file_name.split('/')[1]+'.x.pkl','wb'))
            pickle.dump(y_init,open('B-Box-eval/'+config_file_name.split('/')[1]+'.y.pkl','wb'))

        X = x_init
        Y = y_init.cpu().numpy()
        # if run_iter==0:
        #     X = x_init
        #     Y = y_init.cpu().numpy()
        # else:
        #     X = np.concatenate((x_init,X_origin),axis=0)
        #     Y = np.concatenate((y_init.cpu().numpy(),Y_origin),axis=0)

        z = torch.zeros(n_obj).to(device)
        
        # n_iter batch selections 
        for i_iter in range(n_iter):

            #if i_iter>0:
            #    X = np.concatenate((X,X_origin),axis=0)
            #    Y = np.concatenate((Y,Y_origin),axis=0)
            
            # intitialize the model and optimizer 
            psmodel = ParetoSetModel(n_dim, n_obj)
            psmodel.to(device)
                
            # optimizer
            optimizer = torch.optim.Adam(psmodel.parameters(), lr=1e-3)


            # pre-train the PS model to be uniform distribution
            alpha = np.ones(n_obj)
            #alpha[0]=10.0
            for i_pt in range(100):
                
                pref = np.random.dirichlet(alpha,n_pref_update)
                pref_vec  = torch.tensor(pref).to(device).float() + 0.0001
                x = psmodel(pref_vec)
                loss = torch.sum(x * torch.log(x))/n_pref_update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #print('output')
                #print(x)
            #time.sleep(10)
                



          
            # solution normalization
            transformation = StandardTransform([0,1])
            transformation.fit(X, Y)
            X_norm, Y_norm = transformation.do(X, Y) 

            #print(X)
            #print(Y)

            #print(X_norm)
            #pickle.dump(Y_norm, open('Y_norm.pkl','wb'))
            #exit()
            
            # train GP surrogate model 
            surrogate_model = GaussianProcess(n_dim, n_obj, nu = 5)
            surrogate_model.fit(X_norm,Y_norm)

            pred_Y = surrogate_model.evaluate(X_norm)['F']
            #print('pred_Y')
            #print(pred_Y)
            
            z =  torch.min(torch.cat((z.reshape(1,n_obj),torch.from_numpy(Y_norm).to(device) - 0.1)), axis = 0).values.data
            #print(z)
            #exit()
            # z is the optimal point for each axis/objective

            # nondominated X, Y
            nds = NonDominatedSorting()
            idx_nds = nds.do(Y_norm)
            
            X_nds = X_norm[idx_nds[0]]
            Y_nds = Y_norm[idx_nds[0]]
            
            # t_step Pareto Set Learning with Gaussian Process
            for t_step in range(n_steps):
                psmodel.train()
                
                # sample n_pref_update preferences
                alpha = np.ones(n_obj)
                #alpha[0]=10.0
                pref = np.random.dirichlet(alpha,n_pref_update)
                pref_vec  = torch.tensor(pref).to(device).float() + 0.0001
                
                # get the current coressponding solutions
                x = psmodel(pref_vec)
                x_np = x.detach().cpu().numpy()
                #print('PS model sampling')
                #print(x_np)
                # obtain the value/grad of mean/std for each obj
                mean = torch.from_numpy(surrogate_model.evaluate(x_np)['F']).to(device)
                #print('mean')
                #print(mean)
                #time.sleep(3)
                mean_grad = torch.from_numpy(surrogate_model.evaluate(x_np, calc_gradient=True)['dF']).to(device)
                
                std = torch.from_numpy(surrogate_model.evaluate(x_np, std=True)['S']).to(device)
                std_grad = torch.from_numpy(surrogate_model.evaluate(x_np, std=True, calc_gradient=True)['dS']).to(device)

                #print('std')
                #print(std)
                #time.sleep(3)
                # print(std_grad.size())
                
                # calculate the value/grad of tch decomposition with LCB

                # when the p-value is far from 0.05, add penalty to the search
                # x_np_origin, obj_origin = transformation.undo(x_np, mean.detach().cpu().numpy())
                #print(obj_origin)
                # obj_origin[:,1] = 0.0
                # obj_origin[:,0] = np.abs(obj_origin[:,0]+0.05)
                # _, obj_norm = transformation.do(x_np_origin,obj_origin)
                # obj_norm[:,1] = 0.0
                # obj_norm_t = torch.from_numpy(obj_norm).to(device)
                

                
                value = mean - coef_lcb * std 
                value_grad = mean_grad - coef_lcb * std_grad 
               
                tch_idx = torch.argmax((1 / pref_vec) * (value - z), axis = 1)
                #print('size of tch_idx')
                #print(tch_idx.size())
                tch_idx_mat = [torch.arange(len(tch_idx)),tch_idx]
                #print('tch_idx_mat')
                #print(tch_idx_mat)
                #print(1/pref_vec)
                #print((1 / pref_vec)[tch_idx_mat])
                #print((1 / pref_vec)[tch_idx_mat].view(n_pref_update,1))
                tch_grad = (1 / pref_vec)[tch_idx_mat].view(n_pref_update,1) *  value_grad[tch_idx_mat] + 0.01 * torch.sum(value_grad, axis = 1) 
                #print('size of tch_grad')
                #print(tch_grad.size())

                tch_grad = tch_grad / torch.norm(tch_grad, dim = 1)[:, None]
                #print(tch_grad.size())
                # gradient-based pareto set model update 
                optimizer.zero_grad()
                psmodel(pref_vec).backward(tch_grad)
                optimizer.step()  
                
            # solutions selection on the learned Pareto set
            psmodel.eval()
            
            # sample n_candidate preferences
            alpha = np.ones(n_obj)
            #alpha[0]=5.0
            pref = np.random.dirichlet(alpha,n_candidate)
            #print(pref)
            pref  = torch.tensor(pref).to(device).float() + 0.0001
    
            # generate correponding solutions, get the predicted mean/std
            X_candidate = psmodel(pref).to(torch.float64)
            X_candidate_np = X_candidate.detach().cpu().numpy()
            Y_candidate_mean = surrogate_model.evaluate(X_candidate_np)['F']
            
            Y_candidata_std = surrogate_model.evaluate(X_candidate_np, std=True)['S']

            # compute preference
            obj_origin = transformation.undo(None, Y_candidate_mean)
            obj_origin[:,1] = 0.0
            obj_origin[:,0] = np.abs(obj_origin[:,0]+0.05)
            # obj_origin_t = torch.from_numpy(obj_origin).to(device)
            # obj_origin_grad = np.zeros(obj_origin.shape)
            # obj_origin_grad[:,0] = np.sign(obj_origin[:,0]+0.05)
            # obj_origin_grad_t = torch.from_numpy(obj_origin_grad).to(device)

            print('three component')
            print(Y_candidate_mean)
            print(coef_lcb * Y_candidata_std)
            print(coef_guide * obj_origin)
            #time.sleep(5)

            Y_candidate = Y_candidate_mean - coef_lcb * Y_candidata_std + coef_guide * obj_origin
            
            # optional TCH-based local Exploitation 
            if n_local > 0:
                X_candidate_tch = X_candidate_np
                z_candidate = z.cpu().numpy()
                pref_np = pref.cpu().numpy()
                for j in range(n_local):
                    candidate_mean =  surrogate_model.evaluate(X_candidate_tch)['F']
                    candidate_mean_grad =  surrogate_model.evaluate(X_candidate_tch, calc_gradient=True)['dF']
                    
                    candidate_std = surrogate_model.evaluate(X_candidate_tch, std=True)['S']
                    candidate_std_grad = surrogate_model.evaluate(X_candidate_tch, std=True, calc_gradient=True)['dS']
                    
                    # add penalty to p far from 0.05
                    
                    
                    candidate_value = candidate_mean - coef_lcb * candidate_std
                    candidate_grad = candidate_mean_grad - coef_lcb * candidate_std_grad
                    
                    candidate_tch_idx = np.argmax((1 / pref_np) * (candidate_value - z_candidate), axis = 1)
                    candidate_tch_idx_mat = [np.arange(len(candidate_tch_idx)),list(candidate_tch_idx)]
                    
                    candidate_tch_grad = (1 / pref_np)[np.arange(len(candidate_tch_idx)),list(candidate_tch_idx)].reshape(n_candidate,1) * candidate_grad[np.arange(len(candidate_tch_idx)),list(candidate_tch_idx)] 
                    candidate_tch_grad +=  0.01 * np.sum(candidate_grad, axis = 1) 
                    
                    X_candidate_tch = X_candidate_tch - 0.01 * candidate_tch_grad
                    X_candidate_tch[X_candidate_tch <= 0]  = 0
                    X_candidate_tch[X_candidate_tch >= 1]  = 1  
                    
                X_candidate_np = np.vstack([X_candidate_np, X_candidate_tch])
                
                Y_candidate_mean = surrogate_model.evaluate(X_candidate_np)['F']
                Y_candidata_std = surrogate_model.evaluate(X_candidate_np, std=True)['S']

                obj_origin = transformation.undo(None, Y_candidate_mean)
                obj_origin[:,1] = 0.0
                obj_origin[:,0] = np.abs(obj_origin[:,0]+0.05)

                Y_candidate = Y_candidate_mean - coef_lcb * Y_candidata_std + coef_guide * obj_origin
            
            # greedy batch selection 
            best_subset_list = []
            Y_p = Y_nds 
            for b in range(n_sample):
                hv = HV(ref_point=np.max(np.vstack([Y_p,Y_candidate]), axis = 0))
                best_hv_value = 0
                best_subset = None
                
                for k in range(len(Y_candidate)):
                    Y_subset = Y_candidate[k]
                    Y_comb = np.vstack([Y_p,Y_subset])
                    hv_value_subset = hv(Y_comb)
                    if hv_value_subset > best_hv_value:
                        best_hv_value = hv_value_subset
                        best_subset = [k]
                        
                Y_p = np.vstack([Y_p,Y_candidate[best_subset]])
                best_subset_list.append(best_subset)  
                
            best_subset_list = np.array(best_subset_list).T[0]
            
            # evaluate the selected n_sample solutions
            X_candidate = torch.tensor(X_candidate_np).to(device)
            X_new = X_candidate[best_subset_list]
            X_new_origin = transformation.undo(X_new.detach().cpu().numpy(), None)
            Y_new = problem.evaluate(torch.from_numpy(X_new_origin).to(device))
            
            print(X_norm.shape)
            
            # update the set of evaluated solutions (X,Y)
            X_norm = np.vstack([X_norm, X_new.detach().cpu().numpy()])
            Y_norm = np.vstack([Y_norm, transformation.do(None,Y_new.detach().cpu().numpy())])

            print(X_norm.shape)
            # check the current HV for evaluated solutions
            hv = HV(ref_point=np.array(ref_point))
            hv_value = hv(Y)
            hv_all_value[run_iter, i_iter] = hv_value

            X_origin, Y_origin = transformation.undo(X_norm, Y_norm)

            X = X_origin
            Y = Y_origin

            

            # save
            nds = NonDominatedSorting()
            idx_nds = nds.do(Y_origin)
            
            X_sol = X_origin[idx_nds[0]]
            Y_sol = Y_origin[idx_nds[0]]
            solution={}
            solution['X']=X_sol
            solution['Y']=Y_sol
            with open('result/hv_psl_mobo_0612_guided_search_solution_v2.pickle', 'wb') as output_file:
                pickle.dump(solution, output_file)
            
            
            print("hv", "{:.2e}".format(np.mean(hv_value)))
            print("***")
        
        # store the final performance
        hv_list[test_ins] = hv_all_value
        hv_list[test_ins+'_X_sol'] = X_origin
        hv_list[test_ins+'_Y_sol'] = Y_origin

        
        print("************************************************************")


        with open('result/hv_psl_mobo_0612_guided_search_v2.pickle', 'wb') as output_file:
            pickle.dump([hv_list], output_file)

