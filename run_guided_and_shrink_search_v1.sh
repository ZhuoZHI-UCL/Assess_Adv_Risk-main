#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
coef_lcb=0.01
coef_guide=0.01
python run_black_box_attack_guided_and_shrink_search_v1.py --coef_lcb $coef_lcb --coef_guide $coef_guide --save_path 'result_coef_guide_'$coef_guide'_coef_lcb_'$coef_lcb'.pkl'


coef_lcb=0.01
coef_guide=0.1
python run_black_box_attack_guided_and_shrink_search_v1.py --coef_lcb $coef_lcb --coef_guide $coef_guide --save_path 'result_coef_guide_'$coef_guide'_coef_lcb_'$coef_lcb'.pkl'

coef_lcb=0.01
coef_guide=1.0
python run_black_box_attack_guided_and_shrink_search_v1.py --coef_lcb $coef_lcb --coef_guide $coef_guide --save_path 'result_coef_guide_'$coef_guide'_coef_lcb_'$coef_lcb'.pkl'

coef_lcb=0.01
coef_guide=10.0
python run_black_box_attack_guided_and_shrink_search_v1.py --coef_lcb $coef_lcb --coef_guide $coef_guide --save_path 'result_coef_guide_'$coef_guide'_coef_lcb_'$coef_lcb'.pkl'

