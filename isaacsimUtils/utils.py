import os
import subprocess
import time

def launch_isaac_sim(ISAAC_SIM_PATH):
    """Launch Isaac Sim with ROS 2 bridge."""

    log_file = "logs/isaacsim_log.txt"
    with open(log_file, "w") as f:
        # Call load_isaacsim_stage python file
        return subprocess.Popen(
            [
              os.path.join(ISAAC_SIM_PATH, "python.sh"), 
              "isaacsimUtils/load_isaacsim_stage.py", 
              "--scene", 
              "omniverse://localhost/Projects/zero-to-slam/scene_turtle.usd"
            ],
            stdout=f, 
            stderr=f, 
            env=os.environ.copy(),
            preexec_fn=os.setsid,  # Creates a new process group
          )