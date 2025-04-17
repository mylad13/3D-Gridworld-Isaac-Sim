#!/bin/sh
env="IsaacSim"
scenario="IsaacSim-SearchAndRescue-v0"
num_agents=3
num_obstacles=30
algo="amat"
exp="Render_AMAT"
seed_max=1


echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python launch_multislam.py\
      --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
      --num_agents ${num_agents} --n_rollout_threads 1 \
      --max_steps 500 --agent_view_size 7 \
      --model_dir "/home/farjadnm/CATMiP/hetmarl/scripts/results/GridWorld/MiniGrid-SearchAndRescue-v0/amat/MAT_for_target_finding/wandb/run-20250415_151529-tu2yed55/files" \
      --grid_size 30 --wandb_name "mylad" \
      --user_name "mylad"  \
      --use_wandb  --use_action_masking --action_size 3 \
      --n_head 1 --n_embd 192 --n_block 1 --recurrent_hidden_size 192 \
      --n_eval_rollout_threads 1 \
      --n_agent_types 2 --agent_types "explorer" "explorer" "rescuer" --agent_types_list 0 1 1 --detect_traces \
      --use_full_comm --asynch \
      --use_render \
      #removed  --debug
done

# NOTES:
# n_rollout_threads should be the same as n_eval_rollout_threads for eval to work properly for now.