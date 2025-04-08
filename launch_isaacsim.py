import subprocess
import time
import os


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

def launch_isaac_sim():
    """Launch Isaac Sim with ROS 2 bridge."""

    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, "isaacsim_log.txt")
    with open(log_file, "w") as f:
        # Call load_isaacsim_stage python file
        return subprocess.Popen(
            [
              os.path.join(ISAAC_SIM_PATH, "python.sh"), 
              "isaacsimUtils/load_isaacsim_stage.py", 
              "--scene", 
              "/home/farjadnm/isaac_multi_slam/Simple_Environments/scene_turtle.usd"
            ],
            stdout=f, 
            stderr=f, 
            env=os.environ.copy(),
            preexec_fn=os.setsid,  # Creates a new process group
          )

if __name__ == "__main__":
# may need to run the following commmands in the terminal that runs this script:
# export FASTRTPS_DEFAULT_PROFILES_FILE=/home/farjadnm/IsaacSim-ros_workspaces/humble_ws/fastdds.xml
# export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/farjadnm/isaacsim/exts/isaacsim.ros2.bridge/humble/lib

  print("Launching Isaac Sim with ROS 2 bridge...")
  isaac_sim = launch_isaac_sim()