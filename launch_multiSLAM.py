import subprocess
import time
import os
import signal
import argparse
import random
import cv2
import channelUtils.channel_processing as cproc
from isaacsimUtils.ros_utils import run_ros_command, send_nav_goal, get_nav_goal
import threading

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi SLAM with optional debug mode.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--n_agents", type=str, default="3", help="number of robots to use")
    parser.add_argument("--agent_types", nargs="+", type=str, default=["explorer", "explorer", "rescuer"], help="types of robots to use")

    args = parser.parse_args()
    if args.debug: print("In Debug mode.")

    # Robot namespaces
    robot_ids = []
    initial_poses = {} # Initial poses (x,y) wrt the corner of the map
    n_explorer = 0
    n_rescuer = 0
    static_transform_processes = []
    for i in range(int(args.n_agents)):
        if args.agent_types[i] == "rescuer":
            robot_ids.append(f"rescuer{n_rescuer+1}")
            initial_poses[f"rescuer{n_rescuer+1}"] = [3+i, 5, 0]
            n_rescuer += 1
        elif args.agent_types[i] == "explorer":
            robot_ids.append(f"explorer{n_explorer+1}")
            initial_poses[f"explorer{n_explorer+1}"] = [3+i, 5, 0]
            n_explorer += 1

        odom_static_transform_command = f"ros2 run tf2_ros static_transform_publisher --x 0 --y {-2*i} --z 0 --yaw 0 --pitch 0 --roll 0 --frame-id /odom --child-frame-id /{robot_ids[i]}/odom"
        odom_static_transform_process = run_ros_command(odom_static_transform_command, f"logs/{robot_ids[i]}", "static_transform_log.txt")
        static_transform_processes.append(odom_static_transform_process)

    print("Robot IDs:", robot_ids)
    print("Initial poses:", initial_poses)
    ###--- Generate Target ---###

    # Choose a random target location for the robot
    ground_truth_occupancy_map = cv2.imread("groundTruth/occupancy_map30x30.png", cv2.IMREAD_GRAYSCALE)
    height, width = ground_truth_occupancy_map.shape

    while True:
        x = random.randint(0,width-1)
        y = random.randint(0,height-1)

        if ground_truth_occupancy_map[y, x] == 0 and x >= 10 and y >= 10:
            print(f"Chosen target: ({x}, {y})")
            target_pos = (x, y)
            target_map = cproc.to_single_pose_map(x, y)
            cv2.imwrite(f"channels/global/target_map.png", target_map)
            break
    
    ###--- Launch Processes ---###    

    try:
        # Source ROS2
        subprocess.run("source /opt/ros/humble/setup.bash", shell=True, executable="/bin/bash")
        subprocess.run("source ~/ros2_ws/install/setup.bash", shell=True, executable="/bin/bash")

        # Assuming Isaac Sim is already running the correct scene

        print("Launching SLAM components ...")
        slam_processes = []
        for id in robot_ids:
            # pose = initial_poses[id]
            # pose_str = f"[{','.join(str(v) for v in pose)}]"
            # slam_command = f"ros2 launch slam_toolbox online_async_multirobot_launch.py namespace:={id} map_start_pose:='{pose_str}' use_sim_time:=True"
            slam_command = f"ros2 launch slam_toolbox online_async_multirobot_launch.py namespace:={id} use_sim_time:=True"
            slam_process = run_ros_command(slam_command, f"logs/{id}", "slam_log.txt")
            slam_processes.append(slam_process)
        time.sleep(5) # Wait for SLAM to stabilize

        print("Launching Nav components ...")
        # Get the directory of the folder holding the current script
        package_dir = os.path.dirname(os.path.abspath(__file__))
        map_dir = os.path.join(package_dir, "maps")
        if args.debug: 
            nav_command = f"ros2 launch turtle_navigation multiple_robot_turtle_navigation.launch.py map_dir:={map_dir}"
        else:
            nav_command = f"ros2 launch turtle_navigation multiple_robot_turtle_navigation.launch.py map_dir:={map_dir} use_rviz:=false"
        nav_process = run_ros_command(nav_command, "logs", "nav_log.txt")

        print("Launching Pose Subscribers ...")
        pose_subscribers = []
        for id in robot_ids:
            pose_subscriber_command = f"python3 channelUtils/pose_subscriber.py --namespace {id}"
            pose_subscriber_process = run_ros_command(pose_subscriber_command, f"logs/{id}", "pose_subscriber.txt")
            pose_subscribers.append(pose_subscriber_process)

        time.sleep(5) # Wait for Nav and Pose Subscribers to stabilize

        print("Simulation is fully running! Ready to send navigation goals. CTRL C to close.")
        print()

        ### Assuming Full Communication between all robots
        # Get Macro-Observations for all robots

        # initial goal is just the robot's current position
        for robot_id in robot_ids:
            robot_pose = cproc.getPose(robot_id, initial_poses[robot_id])
            # Save image of the goal
            print(f"Robot {robot_id} pose: {robot_pose}")
            goal_map = cproc.to_single_pose_map(robot_pose[0], robot_pose[1], height)
            dir_path = f"channels/{robot_id}"
            # Ensure directory exists
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            cv2.imwrite(f"{dir_path}/goal_map.png", goal_map)
        macro_observations = cproc.get_macro_observations(robot_ids, robot_ids, initial_poses, map_size=(width, height), target_pos=target_pos)
        
        
        robot_states = {robot_id: "active" for robot_id in robot_ids}
        shared_nav_goals = {robot_id: None for robot_id in robot_ids} 
        state_lock = threading.Lock()
        
        
        def robot_loop(robot_id):
            while True:
                print(f"{robot_id} is in {robot_states[robot_id]} state.")

                active_robots = []
                with state_lock:
                    if robot_states[robot_id] == "active":
                        if shared_nav_goals[robot_id] is not None: # This means another robot has computed a goal for this robot
                            x_rel, y_rel = shared_nav_goals[robot_id]
                            print(f"{robot_id} has a shared goal: {shared_nav_goals[robot_id]}")
                            shared_nav_goals[robot_id] = None
                        else: # This robot is the one computing the goal
                            active_robots.append(robot_id) # put robot_id first in the list
                            for r in robot_ids:
                                if r != robot_id and robot_states[r] == "active":
                                    active_robots.append(r)
                            ## Get Macro-Observations for all active robots
                            # macro_observation = cproc.get_macro_observations(robot_ids, active_robots, initial_poses, map_size=(width, height), target_pos=target_pos)
                            ## Currently just getting a random goal, #TODO: Get navigation goal from CATMiP, active agents do it together
                            print(f"Active robots are {active_robots}.")
                            new_nav_goals = get_nav_goal(n_robots = len(active_robots))
                            if len(active_robots) > 1:
                                for i, id in enumerate(active_robots):
                                    shared_nav_goals[id] = new_nav_goals[i]
                                    print(f"New navigation goal for {id}: {shared_nav_goals[id]}")
                                x_rel, y_rel = new_nav_goals[0]
                                shared_nav_goals[robot_id] = None
                            else: # Only one robot is active
                                x_rel, y_rel = new_nav_goals[0]
                                  
                if robot_states[robot_id] == "active":
                    robot_pose = cproc.getPose(robot_id, initial_poses[robot_id])
                    x_goal = robot_pose[0] + x_rel
                    y_goal = robot_pose[1] + y_rel
                    
                    goal_map = cproc.to_single_pose_map(int(x_goal), int(y_goal))
                    cv2.imwrite(f"channels/{robot_id}/goal_map.png", goal_map)
                    
                    event = threading.Event()
                    
                    send_nav_goal(
                        robot_id,
                        x_goal - initial_poses[robot_id][0],
                        y_goal - initial_poses[robot_id][1],
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
                
                def try_group_activation():
                    with state_lock:
                        others_on_standby = any(
                            r != robot_id and state == "on-standby"
                            for r, state in robot_states.items()
                        )
                        if others_on_standby:
                            for r in robot_ids:
                                if robot_states[r] == "on-standby":
                                    robot_states[r] = "active"
                                    print(f"{r} activating with others.")
                            return True
                        return False
                
                if robot_states[robot_id] == "on-standby":
                    # Wait 5s, try group activation
                    time.sleep(5)
                    if try_group_activation(): # If group activation is successful, continue
                        continue
                
                if robot_states[robot_id] == "on-standby":
                    # Wait another 5s if still on standby, try again
                    print(f"{robot_id} waiting another 5s for partners...")
                    time.sleep(5)

                if robot_states[robot_id] == "on-standby":
                    if not try_group_activation():
                        with state_lock:
                            # If no partners activated, activate alone
                            robot_states[robot_id] = "active"
                            print(f"{robot_id} activating alone.")

        # Start a thread for each robot
        robot_threads = []
        for robot_id in robot_ids:
            t = threading.Thread(target=robot_loop, args=(robot_id,))
            t.start()
            robot_threads.append(t)
 

        while True:
            time.sleep(1)
    
    except Exception as e:
        print(e)
    finally:
        print("Shutting down processes...")
        processes = [nav_process] + static_transform_processes + slam_processes + pose_subscribers
        # processes = [isaac_sim, nav_process] + slam_processes + pose_subscribers
        for proc in processes:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception as ex:
                print(f"Could not kill process group for pid {proc.pid}: {ex}")
        # os.killpg(os.getpgid(isaac_sim.pid), signal.SIGTERM) # Make sure Isaac Sim is killed

        # Shutdown ROS2
        subprocess.run("ros2 daemon stop", shell=True, executable="/bin/bash")
        print("Processes terminated.")

        