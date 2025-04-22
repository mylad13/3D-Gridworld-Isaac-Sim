import numpy as np
import torch
import os
import time
import subprocess
import signal
import cv2
import threading
from isaacsimUtils.utils import launch_isaac_sim
from isaacsimUtils.ros_utils import run_ros_command, send_nav_goal
from catmipUtils.utils import get_nav_goal, plot_macro_obs, get_available_actions, get_action_and_observation_spaces, adjacent_cells, try_group_activation
import channelUtils.channel_processing as cproc

STANDBY_WAIT_INITIAL = 2
STANDBY_WAIT_RETRY = 3

class IsaacSimRunner(object):
    def __init__(self, config):
        self.args = config['all_args']
        self.device = config['device']
        self.run_dir = config['run_dir']
        
        self.ros_processes = []
        self.init_variables()
        self.init_robots()

    def setup_environment(self):
        # Load the Isaac Sim environment
        print("Launching Isaac Sim with ROS 2 bridge...")
        HOME = os.path.expanduser("~")
        ISAAC_SIM_PATH = os.path.join(HOME, "isaacsim")
        self.isaac_sim = launch_isaac_sim(ISAAC_SIM_PATH)
        time.sleep(15)  # Wait for Isaac Sim to stabilize

    def init_variables(self):
        self.num_agents = self.args.num_agents # Number of agents
        self.agent_types = self.args.agent_types # List of agent types (rescuer, explorer, etc.)

        self.height = self.args.grid_size # Height of the grid
        self.width = self.args.grid_size # Width of the grid

        self.target_pos = self.args.target_pos # Target position
        print("Target pos is ", self.target_pos)
        self.target_reached = False # Flag to check if target is reached by a rescuer robot

    def init_robots(self):
        # Robot namespaces
        self.robot_ids = []
        self.initial_poses = {} # Initial poses (x,y) wrt the corner of the map
        self.n_explorer = 0
        self.n_rescuer = 0
        # self.static_transform_processes = []
        for i in range(int(self.num_agents)):
            if self.agent_types[i] == "rescuer":
                self.robot_ids.append(f"rescuer{self.n_rescuer+1}")
                self.initial_poses[f"rescuer{self.n_rescuer+1}"] = [3+i, 5, 0]
                self.n_rescuer += 1
            elif self.agent_types[i] == "explorer":
                self.robot_ids.append(f"explorer{self.n_explorer+1}")
                self.initial_poses[f"explorer{self.n_explorer+1}"] = [3+i, 5, 0]
                self.n_explorer += 1

            #TODO: Figure out how to do multi-robot SLAM with robots not starting at the same position by publishing static transforms between their odometry frames
            # odom_static_transform_command = f"ros2 run tf2_ros static_transform_publisher --x 0 --y {-2*i} --z 0 --yaw 0 --pitch 0 --roll 0 --frame-id /odom --child-frame-id /{robot_ids[i]}/odom"
            # odom_static_transform_process = run_ros_command(odom_static_transform_command, f"logs/{robot_ids[i]}", "static_transform_log.txt")
            # self.static_transform_processes.append(odom_static_transform_process)

        print("Robot IDs:", self.robot_ids)
        print("Initial poses:", self.initial_poses)
    
    def start_ros_nodes(self):
        # Start ROS2 nodes (SLAM, Nav2, etc.) using subprocess or launch files
        print("Launching SLAM components ...")
        for id in self.robot_ids:
            # High-resolution SLAM
            slam_command = f"ros2 launch slam_toolbox online_async_multirobot_launch.py namespace:={id} use_sim_time:=True"
            slam_process = run_ros_command(slam_command, f"logs/{id}", "slam_log.txt")
            self.ros_processes.append(slam_process)
            
            # Low-resolution SLAM
            lowres_slam_command = f"ros2 launch slam_toolbox online_async_multirobot_launch.py namespace:={id}_lowres use_sim_time:=True resolution:=1"
            lowres_slam_process = run_ros_command(lowres_slam_command, f"logs/{id}", "lowres_slam_log.txt")
            self.ros_processes.append(lowres_slam_process)

        time.sleep(10) # Wait for SLAM nodes to stabilize

        print("Launching Nav components ...")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up two levels from runner/shared to the root where maps should be
        self.base_dir = os.path.dirname(os.path.dirname(current_dir)) 
        map_dir = os.path.join(self.base_dir, "maps") 
        
        nav_command_args = f"map_dir:={map_dir}"
        if not self.args.debug: # Whether to launch RViz or not
            nav_command_args += " use_rviz:=false"
                
        nav_command = f"ros2 launch turtle_navigation multiple_robot_turtle_navigation.launch.py {nav_command_args}"
        nav_process = run_ros_command(nav_command, "logs", "nav_log.txt")
        self.ros_processes.append(nav_process)

        print("Launching Pose Subscribers ...")
        for id in self.robot_ids:
            pose_subscriber_command = f"python3 channelUtils/pose_subscriber.py --namespace {id}" # Ensure path is correct relative to execution
            pose_subscriber_process = run_ros_command(pose_subscriber_command, f"logs/{id}", "pose_subscriber.txt")
            self.ros_processes.append(pose_subscriber_process)

        print("Launching Map Subscribers ...")
        for id in self.robot_ids:
            map_subscriber_command = f"python3 channelUtils/map_subscriber.py --namespace {id}_lowres" # Ensure path is correct relative to execution
            map_subscriber_process = run_ros_command(map_subscriber_command, f"logs/{id}", "map_subscriber.txt")
            self.ros_processes.append(map_subscriber_process)

        time.sleep(10) # Wait for Nav and Subscribers to stabilize
        print("ROS nodes launched.")

    def setup_catmip(self):
        from catmipUtils.algorithms.transformer_policy import TransformerPolicy
        action_space, observation_space = get_action_and_observation_spaces(self.args.algorithm_name,
                                                                            num_agents=self.num_agents,
                                                                            n_agent_types=self.args.n_agent_types)
        self.policy = TransformerPolicy(self.args,
                                observation_space[0],
                                observation_space[0],
                                action_space[0],
                                self.num_agents,
                                self.args.n_agent_types,
                                device=self.device)
        if self.args.model_dir is not None:
            self.policy.restore(self.args.model_dir)

    
    def render(self, episodes=1):
        try:
            # If running Isaac Sim on a workstation and the rest of the code inside a docker container, skip this step
            if self.args.docker:
                print("Running in Docker, assuming Isaac Sim is already set up.")
            else:   
                self.setup_environment()
            
            self.start_ros_nodes()
            self.setup_catmip() 

            for ep in range(episodes):
                print(f"--- Starting Episode {ep+1} ---")
                self.run_episode() # You'll need to implement this
                print(f"--- Finished Episode {ep+1} ---")

        except Exception as e:
            print(f"An error occurred during render: {e}")
 
        finally:
            # Cleanup is called regardless of whether an exception occurred
            self.cleanup()
        
    
    def run_episode(self):
        ### Assuming Full Communication between all robots

        # initial goal is just the robot's current position
        for robot_id in self.robot_ids:
            robot_pose = cproc.getPose(robot_id, self.initial_poses[robot_id])
            # Save image of the goal
            print(f"Robot {robot_id} pose: {robot_pose}")
            goal_map = cproc.to_single_pose_map(robot_pose[0], robot_pose[1], self.height)

            dir_path = os.path.join(self.base_dir, "maps", robot_id)
            # Ensure directory exists
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            cv2.imwrite(f"{dir_path}/goal_map.png", goal_map)

        robot_states = {robot_id: "active" for robot_id in self.robot_ids}
        shared_nav_goals = {robot_id: None for robot_id in self.robot_ids}
        rnn_states = {robot_id: np.zeros((self.args.recurrent_hidden_size,), dtype=np.float32) for robot_id in self.robot_ids}
        masks = {robot_id: 1 for robot_id in self.robot_ids} # masks become 0 when the robot is done
        masks_array = np.ones((1, self.num_agents, 1), dtype=np.float32) # masks become 0 when the robot is done
        rnn_states_array = np.zeros((1, self.num_agents, self.args.recurrent_hidden_size), dtype=np.float32)
        available_actions = np.ones((1, self.num_agents , 49), dtype=np.int32)
        self.macro_step_counter = 0
        
        self.mac_obs = None
        self.sorted = None

        state_lock = threading.Lock()   

        def robot_loop(robot_id):
            while self.macro_step_counter < self.args.episode_length and not self.target_reached:
                
                active_robots = []
                with state_lock:
                    if robot_states[robot_id] == "active":
                        if shared_nav_goals[robot_id] is not None: # This means another robot has computed a goal for this robot
                            x_rel, y_rel = shared_nav_goals[robot_id]
                            print(f"{robot_id} has a shared goal: {shared_nav_goals[robot_id]}")
                            shared_nav_goals[robot_id] = None
                        else: # This robot is the one computing the goal
                            active_robots.append(robot_id) # put robot_id first in the list
                            for r in self.robot_ids:
                                if r != robot_id and robot_states[r] == "active":
                                    active_robots.append(r)
                            ## Get Macro-Observations for all active robots
                            macro_observations, sorted_robots = cproc.get_macro_observations(self.robot_ids, active_robots, self.initial_poses, map_size=(self.width, self.height), target_pos=self.target_pos)
                            self.mac_obs = macro_observations.copy()
                            self.sorted = sorted_robots.copy()
                            ## Get navigation goal from CATMiP, active agents do it together
                            rnn_states_array[0,:] = 0
                            for i, r in enumerate(active_robots):
                                available_actions[0][i] = get_available_actions(macro_observations, i, action_size=3, total_actions=49)
                                rnn_states_array[0,i] = rnn_states[r]
                            print(f"Active robots are {active_robots}.")
                            new_nav_goals, new_rnn_states = get_nav_goal(self.policy, macro_observations, masks_array, rnn_states_array, available_actions, action_size=3)
                            self.macro_step_counter += 1
                            print(f"Macro step counter: {self.macro_step_counter}")
                            if len(active_robots) > 1:
                                for i, id in enumerate(active_robots):
                                    rnn_states[id] = new_rnn_states[0,i]
                                    shared_nav_goals[id] = new_nav_goals[0][i][0]
                                    print(f"New navigation goal for {id}: {shared_nav_goals[id]}")
                                x_rel, y_rel = new_nav_goals[0][0][0]
                                shared_nav_goals[robot_id] = None
                            else: # Only one robot is active
                                print(f"New navigation goal for {robot_id}: {new_nav_goals[0][0][0]}")
                                x_rel, y_rel = new_nav_goals[0][0][0]
                                    
                if robot_states[robot_id] == "active":
                    robot_pose = cproc.getPose(robot_id, self.initial_poses[robot_id])
                    if "rescuer" in robot_id and (robot_pose == self.target_pos or robot_pose in adjacent_cells(self.target_pos[0], self.target_pos[1], self.width, self.height)):
                        print(f"{robot_id} reached the target at ({robot_pose[0]}, {robot_pose[1]})!")
                        with state_lock:
                            self.target_reached = True
                            robot_states[robot_id] = "inactive"
                            masks[robot_id] = 0
                        continue
                    print(f"{robot_id} pose: {robot_pose}")
                    
                    if self.args.algorithm_name == "amat":
                        x_goal = robot_pose[0] + x_rel
                        y_goal = robot_pose[1] + y_rel
                    else:
                        raise NotImplementedError("Algorithm not implemented")
                    
                    goal_map = cproc.to_single_pose_map(int(x_goal), int(y_goal))
                    cv2.imwrite(f"maps/{robot_id}/goal_map.png", goal_map)
                    
                    event = threading.Event()
                    
                    send_nav_goal(
                        robot_id,
                        x_goal - self.initial_poses[robot_id][0],
                        y_goal - self.initial_poses[robot_id][1],
                        event=event,
                    )
                    with state_lock:
                        robot_states[robot_id] = "inactive"

                    print(f"{robot_id} sent to ({x_goal}, {y_goal}) and is now inactive.")

                    event.wait()
                
                # Standby mode
                with state_lock:
                    robot_states[robot_id] = "on-standby"
                    print(f"{robot_id} finished navigation and is on standby.")
                
                
                if robot_states[robot_id] == "on-standby":
                    # Wait 5s, try group activation
                    time.sleep(STANDBY_WAIT_INITIAL)
                    if try_group_activation(robot_id, self.robot_ids, robot_states): # If group activation is successful, continue
                        continue
                
                if robot_states[robot_id] == "on-standby":
                    # Wait another 5s if still on standby, try again
                    print(f"{robot_id} waiting another 3s for partners...")
                    time.sleep(STANDBY_WAIT_RETRY)

                if robot_states[robot_id] == "on-standby":
                    if not try_group_activation(robot_id, self.robot_ids, robot_states):
                        with state_lock:
                            # If no partners activated, activate alone
                            robot_states[robot_id] = "active"
                            print(f"{robot_id} activating alone.")
    
        # Start a thread for each robot
        robot_threads = []
        for robot_id in self.robot_ids:
            t = threading.Thread(target=robot_loop, args=(robot_id,))
            t.start()
            robot_threads.append(t)
        
        while not self.target_reached and self.macro_step_counter < self.args.episode_length: # Simplified condition
            # Find the index of rescuer1 in sorted
            if self.sorted is not None and "rescuer1" in self.sorted:
                rescuer1_index = self.sorted.index("rescuer1")
                plot_macro_obs(self.mac_obs, rescuer1_index)

            time.sleep(30)
        
        print("Episode condition met (target reached or max steps).")
        # Wait for all robot threads to finish cleanly
        print("Waiting for robot threads to complete...")
        for t in robot_threads:
            t.join(timeout=30) # Add a timeout to prevent indefinite blocking
            if t.is_alive():
                # Log or handle cases where threads don't stop as expected
                print(f"Warning: Thread {t.name} did not terminate within timeout.")

        print("All robot threads finished.")

    def cleanup(self):
        # Graceful shutdown of simulation and ROS2 nodes
        print("Cleaning up ROS processes...")
        for proc in reversed(self.ros_processes): # Terminate in reverse order
            if proc and proc.poll() is None: # Check if process exists and is running
                try:
                    # Send SIGTERM to the entire process group
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait(timeout=5) # Wait a bit for graceful shutdown
                except ProcessLookupError:
                    print(f"Process {proc.pid} not found.")
                except subprocess.TimeoutExpired:
                    print(f"Process {proc.pid} did not terminate gracefully, sending SIGKILL.")
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL) # Force kill if needed
                except Exception as e:
                    print(f"Error terminating process {proc.pid}: {e}")
        
        self.ros_processes = [] # Clear the list

        # Shutdown ROS2 daemon
        try:
            print("Stopping ROS2 daemon...")
            subprocess.run("ros2 daemon stop", shell=True, check=True, timeout=10, executable="/bin/bash")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
             print(f"Failed to stop ROS2 daemon: {e}")

        # Terminate Isaac Sim if it was launched by this script
        if hasattr(self, 'isaac_sim') and self.isaac_sim and self.isaac_sim.poll() is None:
             print("Terminating Isaac Sim...")
             try:
                 os.killpg(os.getpgid(self.isaac_sim.pid), signal.SIGTERM)
                 self.isaac_sim.wait(timeout=10)
             except Exception as e:
                 print(f"Could not terminate Isaac Sim gracefully: {e}")
                 os.killpg(os.getpgid(self.isaac_sim.pid), signal.SIGKILL)

        print("Cleanup complete.")
    
    

    def run(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
