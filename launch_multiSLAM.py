import subprocess
import time
import os
import signal
import argparse
import random
import cv2
import channelUtils.channel_processing as cproc
from isaacsimUtils.ros_utils import run_ros_command, send_nav_goal
from catmipUtils.utils import get_nav_goal, plot_macro_obs, get_available_actions, get_action_and_observation_spaces, adjacent_cells, try_group_activation
from catmipUtils.config import get_config
import threading
import numpy as np
import torch
import setproctitle
import wandb
import socket
from pathlib import Path

"""
This is the main file for the multi-robot SLAM simulation for CATMiP.
It launches the Isaac Sim simulation, multi-robot SLAM, and navigation2.
It also closes the loop between Isaac Sim and CATMiP by:
- Extracting the channels from the robots and sending them to CATMiP
- Recieving navigation goals from CATMiP and sending them to the robots
"""

# Set paths
HOME = os.path.expanduser("~")
ISAAC_SIM_PATH = os.path.join(HOME, "isaacsim")

if __name__ == "__main__":
    parser = get_config()
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--num_agents", type=int, default=3, help="number of robots to use")
    parser.add_argument("--agent_types", nargs="+", type=str, default=["explorer", "explorer", "rescuer"], help="types of robots to use")
    parser.add_argument('--scenario_name', type=str, default='simple_spread', help="Which scenario to run on")
    parser.add_argument('--grid_size', type=int, default=30, help="map size")
    parser.add_argument('--agent_view_size', type=int, default=7, help="depth the agent can view")
    parser.add_argument('--max_steps', type=int, default=100, help="maximum steps in each episode")
    parser.add_argument('--docker', action='store_true', help="if running in docker")
    parser.add_argument('--target_pos', type=int, nargs=2, default=[21, 9], help="target position for the robots")


    # eval by time step
    parser.add_argument('--use_time', default=False, action='store_true')
    parser.add_argument('--max_timestep', default=240, type=float)

    args = parser.parse_args()
    if args.debug: print("In Debug mode.")

    assert args.use_eval or args.use_render, ("Either use_eval or use_render should be True.")
    
    # cuda
    if args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(args.n_training_threads)
        if args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")\
          / args.env_name / args.scenario_name / args.algorithm_name / args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    
    if args.use_wandb:
        run = wandb.init(config=args,
                         project=args.env_name,
                         entity=args.wandb_name,
                         notes=socket.gethostname(),
                         name=str(args.algorithm_name) + "_" +
                         str(args.experiment_name) +
                         "_seed" + str(args.seed),
                         group=args.scenario_name,
                         dir=str(run_dir),
                         job_type="evaluation",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(args.algorithm_name) + "-" + \
        str(args.env_name) + "-" + str(args.experiment_name) + "@" + str(args.user_name))

    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    config = {
        "all_args": args,
        # "envs": envs,
        # "eval_envs": eval_envs,
        # "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    from runner.shared.isaacsim_runner import IsaacSimRunner as Runner

    runner = Runner(config)

    if args.use_render:
        runner.render()
    elif args.use_eval:
        runner.eval()

    ##########################################################################  

    

    

        