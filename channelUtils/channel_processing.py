import subprocess
import os
import argparse
import sys
import cv2
import numpy as np
import yaml
import subprocess
from typing import Optional
import matplotlib.pyplot as plt
import time
from catmipUtils.utils import adjacent_cells

def resize_image(img: np.ndarray, size: int) -> np.ndarray:
    """
    Resize an image to a square of size x size.
    """
    unknown_val = 205 # Grey

    # Isolate the explored area
    black_pixels = np.argwhere(img == 0)
    if black_pixels.size == 0:
        # No black pixels found, use the whole image
        cropped = img
    else:
        # Isolate the explored area
        # Find the black pixel closest to the top left corner
        top_left = black_pixels.min(axis=0)
        # Find the black pixel closest to the bottom right corner
        bottom_right = black_pixels.max(axis=0)
        cropped = img[top_left[0]:bottom_right[0] + 1, top_left[1]:bottom_right[1] + 1]

    h, w = cropped.shape
    if h > size or w > size:
        print(f"Warning: Cropped map size ({h}, {w}) is larger than desired size ({size}). Cropping...")
        cropped = cropped[:size, :size]
        h, w = cropped.shape

    # Create empty grey canvas and place cropped image in top-left
    padded = np.full((size, size), unknown_val, dtype=np.uint8)
    padded[:h, :w] = cropped

    return padded
    
def to_occupancy_map(img: np.ndarray, size: int = 30) -> np.ndarray:
    """
    Converts a slam map image to an occupancy map.
    Occupied cells are black (0) and safe (free/unknown) are white (255).
    """
    
    # Resize the image
    cropped = resize_image(img, size)

    # Save the cropped image
    occupancy_binary = np.where(cropped == 0, 255, 0).astype(np.uint8) # Occupied cells are white (255) and safe (free/unknown) are black (0)

    return occupancy_binary

def to_exploration_map(img: np.ndarray, size: int = 30) -> np.ndarray:
    """
    Converts a map image to an exploration map.
    Explored areas (free or occupied) are white (255) and unexplored (unknown) are black (0).
    """
    
    # Resize the image
    cropped = resize_image(img, size)

    # Save the cropped image
    exploration_binary = np.where(cropped == 205, 0, 255).astype(np.uint8) # Explored areas (free or occupied) are white (255) and unexplored (unknown) are black (0)

    return exploration_binary

def to_single_pose_map(x: int, y: int, size: Optional[int] = 30, traces = False, enlarge = False) -> None:
    """
    Converts a map image to an single pose map.
    0 for every pixel except the given position, which is 255.
    """
    
    map = np.zeros((size, size), dtype=np.uint8) 
    map[y, x] = 255
    if traces:
        for cells in adjacent_cells(y, x, size, size, surround=True):
            map[cells[0], cells[1]] = 64
    elif enlarge:
        for cells in adjacent_cells(y, x, size, size, surround=True):
            map[cells[0], cells[1]] = 255

    return map

def to_multi_pose_map(coordinates: list[tuple[int, int]], size, enlarge = False) -> None:
    """
    Converts a map image to a agent pose maps.
    0 for every pixel except agent positions, which is 255.
    """
    map = np.zeros((size, size), dtype=np.uint8)
    for x, y in coordinates:
        # x += 4 # Offset based on starting position in Isaac Sim
        # y += 4
        map[y, x] = 255
        if enlarge:
            for cells in adjacent_cells(y, x, size, size, surround=True):
                map[cells[0], cells[1]] = 255
    return map

def getPose(namespace: str = "robot1", initial_pose=[0,0,0]) -> tuple[int, int]:
    """
    Reads the last robot position from a file and returns the x, y coordinates wrt the origin of the map.
    """
    
    try:
        # Load the latest position of the robot
        with open(f"logs/{namespace}/pose.txt", "r") as f:
            lines = f.readlines()
            if not lines:
                raise FileNotFoundError(f"No position data found for {namespace}")
            # Get the latest position
            y, x = lines[-1].strip().split(",") # The coordinate system is flipped in nav2 map vs the image
            x, y = int(x), int(y)
    except FileNotFoundError:
        print(f"No position data found for {namespace}")
        x, y = 0, 0
    
    # Offset based on starting position in Isaac Sim
    x = x + initial_pose[0]
    y = y + initial_pose[1]

    return x, y

def getMap(namespace: str = "robot1", size: int = 30) -> np.ndarray:
    """
    Extracts the channels from the robot's map and saves them as images.
    """
    # # wait for the map to be saved
    # while not os.path.exists(f"maps/{namespace}_lowres/map.pgm"):
    #     print(f"Waiting for map to be saved to maps/{namespace}_lowres/map.pgm...")
    #     time.sleep(1)

    # Load YAML file to get map metadata
    yaml_path = f"maps/{namespace}_lowres/map.yaml"
    # if not os.path.exists(yaml_path):
    #     print(f"Map YAML file not found at {yaml_path}.")
    # else:
    #     with open(yaml_path, 'r') as f:
    #         map_info = yaml.safe_load(f)

    # Construct the full path to the pgm file
    pgm_path = f"maps/{namespace}_lowres/map.pgm"

    # Load the map image
    img = cv2.imread(pgm_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Failed to load image from {pgm_path}.")
    else:
        # Rotate the image +270 degrees to match orientation of numpy array
        img = np.rot90(img, k=3)
        img = cv2.flip(img, 1)

    return img

def isolateLocalMap(x: int, y: int, img: np.ndarray, padding_value=0) -> np.ndarray:
    """
    Isolates a local map of 7x7 centered around the robot's position.
    """

    # Find corners of the local map within bounds
    x1 = max(0, x - 3)
    x2 = min(img.shape[1], x + 4)
    y1 = max(0, y - 3)
    y2 = min(img.shape[0], y + 4)

    local_map = img[y1:y2, x1:x2]

    # Pad the local map if it is smaller than 7x7 ensuring the robot is centered
    if local_map.shape[0] < 7:
        top_pad = (7 - local_map.shape[0]) // 2
        bottom_pad = 7 - local_map.shape[0] - top_pad
        local_map = np.pad(local_map, ((top_pad, bottom_pad), (0, 0)), mode='constant', constant_values=padding_value)
    if local_map.shape[1] < 7:
        left_pad = (7 - local_map.shape[1]) // 2
        right_pad = 7 - local_map.shape[1] - left_pad
        local_map = np.pad(local_map, ((0, 0), (left_pad, right_pad)), mode='constant', constant_values=padding_value)

    return local_map

def get_macro_observations(all_robots: list[str], active_robots: list[str], initial_poses, map_size, target_pos):
    """
    Get macro-observations for all robots.
    robot_ids: List of robot namespaces
    map_size: (width, height) of the map
    """
    macro_obs = {}
    macro_obs['agent_class_identifier'] = np.zeros((1, len(all_robots), 2), dtype=int) # [1, 0] for rescuer, [0, 1] for explorer
    macro_obs['global_agent_map'] = np.zeros((1, len(all_robots), 7, *map_size), dtype=np.float32) # 7 channels, full map size
    macro_obs['local_agent_map'] = np.zeros((1, len(all_robots), 6, 7, 7), dtype=np.float32) # 6 channels, local map size
    macro_obs['agent_position'] = np.zeros((1, len(all_robots), 2), dtype=int) # [x, y] coordinates

    robot_positions = {}
    rescuers = []
    explorers = []
    for robot_id in all_robots:
        # Identify the robot type
        if "rescuer" in robot_id:
            rescuers.append(robot_id)
        elif "explorer" in robot_id:
            explorers.append(robot_id)
        else:
            raise ValueError(f"Unknown robot type: {robot_id}")
        
        # Get the robot's pose
        x, y = getPose(robot_id, initial_poses[robot_id])
        robot_positions[robot_id] = (x, y)

    # Copy the list of active robots to a new list called sorted_robots, and add the rest of the robots from all_robots list to sorted_robots
    sorted_robots = active_robots.copy()
    for robot_id in all_robots:
        if robot_id not in sorted_robots:
            sorted_robots.append(robot_id)




    for i, robot_id in enumerate(sorted_robots):
        if robot_id in rescuers:
            macro_obs["agent_class_identifier"][0, i, 0] = 1
        elif robot_id in explorers:
            macro_obs["agent_class_identifier"][0, i, 1] = 1
        else:
            raise ValueError(f"Unknown robot type: {robot_id}")
        
        # Load the map image
        img = getMap(robot_id, map_size[0])

        # Convert the map image to the different global channels:
        # 0. Exploration map, 1. Occupancy map, 2. Target map, 3. Ego-pose map 4. Rescuers map, 5. Explorers map, 6. Goal map
        if img is not None:
            exploration_map = to_exploration_map(img, map_size[0])
            occupancy_map = to_occupancy_map(img, map_size[0])
        else:
            exploration_map = np.zeros((map_size[0], map_size[1]), dtype=np.uint8)
            occupancy_map = np.zeros((map_size[0], map_size[1]), dtype=np.uint8)
        
        if exploration_map[target_pos[1], target_pos[0]] == 255:
            target_map = to_single_pose_map(target_pos[0], target_pos[1], map_size[0], enlarge=True)
            pre_local_target_map = to_single_pose_map(target_pos[0], target_pos[1], map_size[0], traces=True)
            print(f"Target found in {robot_id}'s map.")
        else:
            target_map = np.zeros((map_size[0], map_size[1]), dtype=np.uint8)
            pre_local_target_map = np.zeros((map_size[0], map_size[1]), dtype=np.uint8)
        
        ego_pose_map = to_single_pose_map(robot_positions[robot_id][0], robot_positions[robot_id][1], map_size[0], enlarge=True)
        
        rescuers_map = np.zeros((map_size[0], map_size[1]), dtype=np.uint8)
        rescuer_positions = [robot_positions[rescuer] for rescuer in rescuers if rescuer != robot_id]
        rescuers_map = to_multi_pose_map(rescuer_positions, map_size[0], enlarge=True)
        pre_local_rescuers_map = to_multi_pose_map(rescuer_positions, map_size[0])

        
        explorers_map = np.zeros((map_size[0], map_size[1]), dtype=np.uint8)
        explorer_positions = [robot_positions[explorer] for explorer in explorers if explorer != robot_id]
        explorers_map = to_multi_pose_map(explorer_positions, map_size[0], enlarge=True)
        pre_local_explorers_map = to_multi_pose_map(explorer_positions, map_size[0])

        goal_map = cv2.imread(f"maps/{robot_id}/goal_map.png", cv2.IMREAD_GRAYSCALE)
        
        macro_obs['global_agent_map'][0, i, 0] = exploration_map
        macro_obs['global_agent_map'][0, i, 1] = occupancy_map
        macro_obs['global_agent_map'][0, i, 2] = target_map
        macro_obs['global_agent_map'][0, i, 3] = ego_pose_map
        macro_obs['global_agent_map'][0, i, 4] = rescuers_map
        macro_obs['global_agent_map'][0, i, 5] = explorers_map
        macro_obs['global_agent_map'][0, i, 6] = goal_map
        
        # Obtain local maps from their global counterparts:
        # 0. Exploration map, 1. Occupancy map, 2. Target map, 3. Rescuers map, 4. Explorers map, 5. Goal map

        local_occupancy_map = isolateLocalMap(robot_positions[robot_id][0], robot_positions[robot_id][1], occupancy_map)
        local_exploration_map = isolateLocalMap(robot_positions[robot_id][0], robot_positions[robot_id][1], exploration_map)
        local_target_map = isolateLocalMap(robot_positions[robot_id][0], robot_positions[robot_id][1], pre_local_target_map)
        local_rescuers_map = isolateLocalMap(robot_positions[robot_id][0], robot_positions[robot_id][1], pre_local_rescuers_map)
        local_explorers_map = isolateLocalMap(robot_positions[robot_id][0], robot_positions[robot_id][1], pre_local_explorers_map)
        local_goal_map = isolateLocalMap(robot_positions[robot_id][0], robot_positions[robot_id][1], goal_map)

        macro_obs['local_agent_map'][0, i, 0] = local_exploration_map
        macro_obs['local_agent_map'][0, i, 1] = local_occupancy_map
        macro_obs['local_agent_map'][0, i, 2] = local_target_map
        macro_obs['local_agent_map'][0, i, 3] = local_rescuers_map
        macro_obs['local_agent_map'][0, i, 4] = local_explorers_map
        macro_obs['local_agent_map'][0, i, 5] = local_goal_map

        macro_obs['agent_position'][0, i] = robot_positions[robot_id]

    # Normalize the maps to [0, 1]
    macro_obs['global_agent_map'] = macro_obs['global_agent_map'] / 255.0
    macro_obs['local_agent_map'] = macro_obs['local_agent_map'] / 255.0
    return macro_obs, sorted_robots

if __name__ == "__main__":
    
    print("This is a utility file for processing channels in the multi-robot SLAM simulation.")