import numpy as np
import gym
import pygame
import matplotlib.pyplot as plt

# Environment
env = gym.make('FrozenLake-v1', is_slippery=True)

# Q-table
q_table = np.zeros([env.observation_space.n, env.action_space.n])
# [[0. 0. 0. 0.]    0
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#       ...					...
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]   15 (grid is 4x4, we flatten it and add 4 possible actions to each place)

# Actions
# 0 - left
# 1 - down
# 2 - right
# 3 - up

# Observation represented as (from 0 to 15)
# Agent current position -> current_row * n_rows + current_col

# Hyperparameters
alpha = 0.1   # Learning rate

gamma = 0.99  # Discount factor

epsilon = 0.1 # Exploration rate - what is the probability to select random action for exploration

episodes = 1000  # For how many episodes we want to train it?

TILE_SIZE = 100
GRID_COLOR = (0, 0, 255)
HOLE_COLOR = (0, 0, 0)
FROZEN_COLOR = (173, 216, 230)
START_COLOR = (0, 255, 0)
GOAL_COLOR = (255, 215, 0)
AGENT_COLOR = (255, 0, 0)

pygame.init()
rows, cols = env.desc.shape
screen_width = cols * TILE_SIZE
screen_height = rows * TILE_SIZE
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("FrozenLake Visualization")

def draw_grid(state):
    for row in range(rows):
        for col in range(cols):
            tile = env.desc[row, col].decode("utf-8")
            x, y = col * TILE_SIZE, row * TILE_SIZE

            if tile == "S":
                color = START_COLOR
            elif tile == "H":
                color = HOLE_COLOR
            elif tile == "G":
                color = GOAL_COLOR
            else:
                color = FROZEN_COLOR

            pygame.draw.rect(screen, color, (x, y, TILE_SIZE, TILE_SIZE))
            pygame.draw.rect(screen, GRID_COLOR, (x, y, TILE_SIZE, TILE_SIZE), 2)

    # Highlight the agent's position
    agent_row, agent_col = divmod(state, cols)
    agent_x, agent_y = agent_col * TILE_SIZE, agent_row * TILE_SIZE
    pygame.draw.rect(screen, AGENT_COLOR, (agent_x, agent_y, TILE_SIZE, TILE_SIZE))

def visualize_q_values(q_table, episode):
    grid_size = 4
    fig, ax = plt.subplots(figsize=(8, 8))

    for state in range(env.observation_space.n):
        row, col = divmod(state, grid_size)

        q_values = q_table[state]

        y_offset = -0.5

        positions = {
            0: (col - 0.15, grid_size - row - 0.5 + y_offset),  # Left
            1: (col, grid_size - row - 0.65 + y_offset),        # Down
            2: (col + 0.15, grid_size - row - 0.5 + y_offset),  # Right
            3: (col, grid_size - row - 0.35 + y_offset),        # Up
        }

        for action, (x, y) in positions.items():
            color = plt.cm.viridis(q_values[action] / max(1e-3, q_table.max()))
            rect = plt.Rectangle((x - 0.05, y - 0.05), 0.1, 0.1, color=color)
            ax.add_patch(rect)

            ax.text(x, y, f"{q_values[action]:.2f}", ha="center", va="center", fontsize=6, color="white")

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.set_xticklabels(range(grid_size))
    ax.set_yticklabels(range(grid_size))
    ax.set_title(f"Q-values for Each Action (Episode {episode})")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")

    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=q_table.min(), vmax=q_table.max()))
    plt.colorbar(sm, ax=ax, label="Q-value")

    plt.savefig(f"slippery_q_values_{episode}.png")
    plt.close()


def get_max_action(state):
    action_values = q_table[state]
    max_value_indices = [0]
    max_value = action_values[0]

    for i in range(1, len(action_values)):
        if action_values[i] > max_value:
            max_value = action_values[i]
            max_value_indices = [i]
        elif action_values[i] == max_value:
            max_value_indices.append(i)

    return np.random.choice(max_value_indices)
        

# Training Loop
for episode in range(1, episodes + 1):
    state = env.reset()[0] # (int, {'prob': int})
    done = False

    while not done:

        # Choose action (ε-greedy)
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = get_max_action(state)  # Exploit

        # Take action
        next_state, reward, done, _, _ = env.step(action)
        # print('state -', state, 'action -', action, 'next_state -', next_state)

        # Q-value update
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        q_table[state, action] = old_value + alpha * (reward + gamma * next_max - old_value)

        state = next_state

        # if done:
        #     print('FELL INTO THE HOLE!')

        # Visualization during training
        # screen.fill((0, 0, 0))
        # draw_grid(state)
        # pygame.display.flip()
        # pygame.time.delay(10)

    print('Episode:', episode)
    if episode % 100 == 0:
        # max_q_action_values_grid = np.max(q_table.reshape(4, 4, 4), axis=2)
        # print(max_q_action_values_grid)
        visualize_q_values(q_table, episode)


np.save('q_table.npy', q_table)

q_table = np.load('q_table.npy')

# Testing the Policy
state = env.reset()[0]
done = False
print(q_table)
while not done:
    action = get_max_action(state)
    state, reward, done, _, _ = env.step(action)

    screen.fill((0, 0, 0))
    draw_grid(state)
    pygame.display.flip()
    pygame.time.delay(300)

pygame.quit()
