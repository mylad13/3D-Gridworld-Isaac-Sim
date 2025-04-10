import random
import numpy as np
import matplotlib.pyplot as plt

def get_nav_goal(n_robots = 1):
  """Generate a n_robots number of random ego-relative navigation goal (x, y) with values between -3 and 3."""
  nav_goals = []
  for i in range(n_robots):
    x = random.randint(-1, 3)
    y = random.randint(-1, 3)
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
    
    fig, axs = plt.subplots(2, 7, figsize=(16, 8))
    fig.suptitle('Macro Observation Space', fontsize=16)

    # Global maps
    axs[0, 0].imshow(global_exploration, cmap='gray')
    axs[0, 0].set_title('Global Exploration')
    axs[0, 1].imshow(global_occupancy, cmap='gray')
    axs[0, 1].set_title('Global Occupancy')
    axs[0, 2].imshow(global_target, cmap='gray')
    axs[0, 2].set_title('Global Target')
    axs[0, 3].imshow(global_ego_pose, cmap='gray')
    axs[0, 3].set_title('Global Ego Pose')
    axs[0, 4].imshow(global_rescuers, cmap='gray')
    axs[0, 4].set_title('Global Rescuers')
    axs[0, 5].imshow(global_explorers, cmap='gray')
    axs[0, 5].set_title('Global Explorers')
    axs[0, 6].imshow(global_goal, cmap='gray')
    axs[0, 6].set_title('Global Goal')

    # Local maps
    axs[1, 0].imshow(local_exploration, cmap='gray')
    axs[1, 0].set_title('Local Exploration')
    axs[1, 1].imshow(local_occupancy, cmap='gray')
    axs[1, 1].set_title('Local Occupancy')
    axs[1, 2].imshow(local_target, cmap='gray')
    axs[1, 2].set_title('Local Target')
    axs[1, 3].imshow(local_rescuers, cmap='gray')
    axs[1, 3].set_title('Local Rescuers')
    axs[1, 4].imshow(local_explorers, cmap='gray')
    axs[1, 4].set_title('Local Explorers')
    axs[1, 5].imshow(local_goal, cmap='gray')
    axs[1, 5].set_title('Local Goal')

    # Hide unused subplots
    for i in range(6, 7):
        axs[1, i].axis('off')

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
