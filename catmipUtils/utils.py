import random
import numpy as np
import matplotlib.pyplot as plt

def get_nav_goal(n_robots = 1):
  """Generate a n_robots number of random ego-relative navigation goal (x, y) with values between -3 and 3."""
  nav_goals = []
  for i in range(n_robots):
    x = random.randint(-3, 3)
    y = random.randint(-3, 3)
    nav_goals.append((x, y))
  return nav_goals

def plot_macro_obs(macro_obs, agent_num):
    """Plot all the channels of the macro observation space."""
    global_exploration = macro_obs['global_agent_map'][0, agent_num, 0]
    global_occupancy = macro_obs['global_agent_map'][0, agent_num, 1]
    global_target = macro_obs['global_agent_map'][0, agent_num, 2]
    global_ego_pose = macro_obs['global_agent_map'][0, agent_num, 3]
    global_rescuers = macro_obs['global_agent_map'][0, agent_num, 4]
    global_explorers = macro_obs['global_agent_map'][0, agent_num, 5]
    global_goal = macro_obs['global_agent_map'][0, agent_num, 6]

    local_exploration = macro_obs['local_agent_map'][0, agent_num, 0]
    local_occupancy = macro_obs['local_agent_map'][0, agent_num, 1]
    local_target = macro_obs['local_agent_map'][0, agent_num, 2]
    local_rescuers = macro_obs['local_agent_map'][0, agent_num, 3]
    local_explorers = macro_obs['local_agent_map'][0, agent_num, 4]
    local_goal = macro_obs['local_agent_map'][0, agent_num, 5]

    plt.subplot(3, 3, 1)
    plt.imshow(global_exploration, cmap='gray')
    plt.title('Global Exploration')
    plt.subplot(3, 3, 2)
    plt.imshow(global_occupancy, cmap='gray')
    plt.title('Global Occupancy')
    plt.subplot(3, 3, 3)
    plt.imshow(global_target, cmap='gray')
    plt.title('Global Target')
    plt.subplot(3, 3, 4)
    plt.imshow(global_ego_pose, cmap='gray')
    plt.title('Global Ego Pose')
    plt.subplot(3, 3, 5)
    plt.imshow(global_rescuers, cmap='gray')
    plt.title('Global Rescuers')
    plt.subplot(3, 3, 6)
    plt.imshow(global_explorers, cmap='gray')
    plt.title('Global Explorers')
    plt.subplot(3, 3, 7)
    plt.imshow(global_goal, cmap='gray')
    plt.title('Global Goal')
    plt.show()