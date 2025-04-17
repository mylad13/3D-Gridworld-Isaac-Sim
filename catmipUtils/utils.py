import random
import numpy as np
import matplotlib.pyplot as plt
import gym

def _t2n(x): #tensor to numpy array
    return x.detach().cpu().numpy()

def get_nav_goal(policy, macro_observations, masks, rnn_states, available_actions, action_size = 3):
    """Get navigation goals from a policy."""
    for key in macro_observations.keys():
        # merge the first two dimensions (num_threads and num_agents)
        macro_observations[key] = np.concatenate(macro_observations[key], axis=0)
    action, rnn_states = policy.act(
                    macro_observations, # not using centralized observations
                    macro_observations,
                    np.concatenate(masks),
                    rnn_states,
                    available_actions = available_actions,
                    deterministic=True
                )
    rnn_states = np.array(_t2n(rnn_states))

    goal = np.array(np.split(_t2n(action), 1)).astype(np.int32)
    row = goal//(2*action_size+1) - action_size
    col = goal%(2*action_size+1) - action_size
    nav_goals = np.stack((row, col), axis=-1)

    return nav_goals, rnn_states


def plot_macro_obs(macro_obs, agent_num):
    """Plot all the channels of the macro observation space."""

    if macro_obs == None:
        print("No macro observation available.")
        return
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

    agent_indication = macro_obs['agent_class_identifier'][0, agent_num].reshape(1, -1)
    
    fig, axs = plt.subplots(2, 7, figsize=(16, 8))
    fig.suptitle('Macro Observations', fontsize=16)

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
    axs[1, 6].imshow(agent_indication, cmap='gray')
    axs[1, 6].set_title('Agent Class Identifier')

    # Hide unused subplots
    for i in range(6, 7):
        axs[1, i].axis('off')

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def adjacent_cells(x, y, map_width, map_height, surround=False):
        """
        Get the list of cells adjacent to a given cell
        """
        adj = []
        if surround:
            steps = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        else:
            steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in steps:
            if x+dx >= 0 and x+dx < map_width and y+dy >= 0 and y+dy < map_height:
                adj.append((x+dx, y+dy))
        

        return adj


def get_available_actions(macro_obs, agent_num, action_size=3, total_actions=49):
    """Get the available actions for a specific agent based on the macro observation space."""
    # Extract the local observation for the specified agent
    local_exploration = macro_obs['local_agent_map'][0, agent_num, 0]
    local_occupied = macro_obs['local_agent_map'][0, agent_num, 1]

    global_exploration = macro_obs['global_agent_map'][0, agent_num, 0]
    global_occupied = macro_obs['global_agent_map'][0, agent_num, 1]
    global_ego_pose = macro_obs['global_agent_map'][0, agent_num, 3]

    agent_pose = macro_obs['agent_position'][0, agent_num]
    map_size = global_exploration.shape[0]

    available_actions = np.ones(total_actions, dtype=np.int32)
    # Unexplored cells, occupied cells, and cells outside the rnage of the map are unavailable
    for x in range(-action_size, action_size+1):
        for y in range(-action_size, action_size+1):
            coord = np.array([x, y]) + agent_pose
            if coord[0] < 1 or coord[0] >= map_size - 1 or coord[1] < 1 or coord[1] >= map_size - 1:
                available_actions[(x + action_size)*(2*action_size+1) + (y + action_size)] = 0
            # elif global_exploration[coord[0], coord[1]] == 0 or global_occupied[coord[0], coord[1]] == 1:
            elif global_occupied[coord[0], coord[1]] == 1:
                available_actions[(x + action_size)*(2*action_size+1) + (y + action_size)] = 0
            else:
                for adjacent_cell in adjacent_cells(coord[0], coord[1], map_size, map_size):
                    if global_occupied[adjacent_cell[0], adjacent_cell[1]] == 0:
                        break
                else:
                    available_actions[(x + action_size)*(2*action_size+1) + (y + action_size)] = 0
    
    return available_actions

def get_action_and_observation_spaces(algorithm_name, num_agents = 3, n_agent_types = 2, action_size_length = 7, grid_size = 30):
    global_observation_space = {}
    if algorithm_name == 'mat' or algorithm_name == 'amat':
        global_observation_space['agent_class_identifier'] = gym.spaces.Box(low=0, high=1, shape=(n_agent_types,), dtype='uint8')
        global_observation_space['global_agent_map'] = gym.spaces.Box(low=0, high=1, shape=(7 ,grid_size, grid_size), dtype='float')
        global_observation_space['local_agent_map'] = gym.spaces.Box(low=0, high=1, shape=(6, action_size_length, action_size_length), dtype='float')
    elif algorithm_name == 'mancp':
        global_observation_space['agent_class_identifier'] = gym.spaces.Box(low=0, high=1, shape=(n_agent_types,), dtype='uint8')
        global_observation_space['global_agent_map'] = gym.spaces.Box(low=0, high=1, shape=(7 ,grid_size, grid_size), dtype='float')
        global_observation_space['local_agent_map'] = gym.spaces.Box(low=0, high=1, shape=(6, action_size_length, action_size_length), dtype='float')
        global_observation_space['timespan'] = gym.spaces.Box(low=0, high=30, shape=(1,), dtype='uint8')
    elif algorithm_name[:2] == "ft":
        pass
    else:
        raise NotImplementedError
    
    observation_space = []
    action_space = []
    for _ in range(num_agents):
        observation_space.append(gym.spaces.Dict(global_observation_space))
        action_space.append(gym.spaces.Discrete(action_size_length * action_size_length))

    return action_space, observation_space

def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    elif obs_space.__class__.__name__ == 'Dict':
        obs_shape = obs_space.spaces
    else:
        raise NotImplementedError
    return obs_shape

def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == 'Discrete':
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1  
    return act_shape

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr