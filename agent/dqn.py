import numpy as np
import random
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple
import collections
import cv2
import os

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DQN_Agent:
    def __init__(self, environment_name=None, lr=5e-4, render=False, env=None):
        # Initialize the DQN Agent.
        if env is None:
            self.env = gym.make(environment_name)
        else:
            self.env = env

        torch.cuda.empty_cache() 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.policy_net = QNetwork(self.env, self.lr)
        self.target_net = QNetwork(self.env, self.lr)
        self.target_net.net.load_state_dict(self.policy_net.net.state_dict())  # Copy the weight of the policy network
        self.rm = ReplayMemory(self.env) 
        self.burn_in_memory()
        self.batch_size = 8
        self.gamma = 0.99
        self.c = 0
    
    def save_model_weights(self, filename):
        torch.save(self.policy_net.net.state_dict(), filename)
    
    def burn_in_memory(self):
        print("=======================================================================")
        print("Creating and populating the replay memory with a burn_in number of episodes... The process should take around 1 hour with 10000")
        # Initialize replay memory with a burn-in number of episodes/transitions.
        cnt = 0
        terminated = False
        truncated = False
        state, _ = self.env.reset()
        state = self.process_state(state)

        # Iterate until we store "burn_in" buffer
        while cnt < self.rm.burn_in:
            # Reset environment if terminated or truncated
            if terminated or truncated:
                state, _ = self.env.reset()
                state = self.process_state(state)
            
            # Randomly select an action (left or right) and take a step
            action = self.env.action_space.sample()
            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
            if terminated:
                next_state = None
            else:
                next_state = self.process_state(next_state)
                
            # Store new experience into memory
            transition = Transition(state, torch.tensor([[self.env.action_space.sample()]], dtype=torch.long, device=self.device), next_state, reward)
            self.rm.memory.append(transition)
            state = next_state
            cnt += 1

        print("Process complete! Initializing training")
        print("=======================================================================")

    def epsilon_greedy_policy(self, q_values, epsilon=0.05):
        # Implement an epsilon-greedy policy. 
        p = random.random()
        if p > epsilon:
            with torch.no_grad():
                return self.greedy_policy(q_values)
        else:
            return torch.tensor([[self.env.action_space.sample()]], dtype=torch.long, device=self.device)

    def greedy_policy(self, q_values):
        # Implement a greedy policy for test time.
        return torch.argmax(q_values)

    def process_state(self, state):
        # Resize the RGB image to 224x224
        rgb_data = cv2.resize(state['rgb_data'], (224, 224))
        rgb_data = np.transpose(rgb_data, (2, 0, 1))
        rgb_data = torch.from_numpy(rgb_data).float().to(self.device)
        position = torch.FloatTensor(state['position']).to(self.device)
        situation = torch.FloatTensor([state['situation']]).to(self.device)
        target_position = torch.FloatTensor(state['target_position']).to(self.device)

        rest = torch.cat((position, situation, target_position)).to(self.device)

        return (rgb_data, rest)
        
    def train(self):
        # Train the Q-network using Deep Q-learning.
        state, _ = self.env.reset()
        rgb_state, rest_state = self.process_state(state)
        terminated = False
        truncated = False
        # Loop until reaching the termination state
        while not (terminated or truncated):
            with torch.no_grad():
                q_values = self.policy_net.net((rgb_state, rest_state))

            # Decide the next action with epsilon greedy strategy
            action = self.epsilon_greedy_policy(q_values).reshape(1, 1)
            
            # Take action and observe reward and next state
            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            reward = torch.tensor([reward], device=self.device)
            if terminated:
                next_state = None
            else:
                rgb_next_state, rest_next_state = self.process_state(next_state)

            # Store the new experience
            transition = Transition((rgb_state, rest_state), action, (rgb_next_state, rest_next_state), reward)
            self.rm.memory.append(transition)

            # Move to the next state
            rgb_state, rest_state = rgb_next_state, rest_next_state

            # Sample minibatch with size N from memory
            transitions = self.rm.sample_batch(self.batch_size)
            batch = Transition(*zip(*transitions))
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool, device=self.device) # Shape: torch.Size([8])
            
            # non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            non_final_next_states = tuple(s for s in batch.next_state if s is not None)
            non_final_rgb_next_states = torch.stack([s[0] for s in non_final_next_states]).to(self.device)
            non_final_rest_next_states = torch.stack([s[1] for s in non_final_next_states]).to(self.device)

            batch_rgb_states = torch.stack([s[0] for s in batch.state]).to(self.device)
            batch_rest_states = torch.stack([s[1] for s in batch.state]).to(self.device)

            action_batch = torch.cat(batch.action).to(self.device) # Shape: torch.Size([8, 1])
            reward_batch = torch.cat(batch.reward).to(self.device)

            # Get current and next state values
            state_q_values = self.policy_net.net((batch_rgb_states, batch_rest_states)).squeeze(0) # Shape: torch.Size([8, 4]
            state_action_values = state_q_values.gather(1, action_batch).float() # extract values corresponding to the actions Q(S_t, A_t)
            next_state_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float32) # Shape: torch.Size([8])
        
            with torch.no_grad():
                max_q_value = self.target_net.net((non_final_rgb_next_states, non_final_rest_next_states)).squeeze(0).max(1)[0] # extract max value
                # no next_state_value update if an episode is terminated (next_satate = None)
                # only update the non-termination state values (Ref: https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)
                next_state_values[non_final_mask] = max_q_value

                
            # Update the model
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch
            criterion = torch.nn.MSELoss()
            loss = criterion(state_action_values.float(), expected_state_action_values.unsqueeze(1).float())
            self.policy_net.optimizer.zero_grad()
            loss.backward()
            self.policy_net.optimizer.step()

            # Update the target Q-network in each 50 steps
            self.c += 1
            if self.c % 50 == 0:
                self.target_net.net.load_state_dict(self.policy_net.net.state_dict())
        
        print("Episode ended!")
    
    def test(self, model_file=None):
        # Evaluates the performance of the agent over 20 episodes.

        max_t = 1000
        state, _ = self.env.reset()
        rewards = []

        for t in range(max_t):
            state = self.process_state(state)
            with torch.no_grad():
                q_values = self.policy_net.net(state)
            action = self.greedy_policy(q_values)
            state, reward, terminated, truncated, _ = self.env.step(action.item())
            rewards.append(reward)
            if terminated or truncated:
                break

        return np.sum(rewards)

class DQNNetwork(nn.Module):
    def __init__(self, output_n):
        super(DQNNetwork, self).__init__()

        # Define the neural network architecture
        # RGB: (224, 224) -> 512
        self.model1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # Global average pooling to get a fixed-size feature vector
        )

        # Rest: 7
        self.model3 = nn.Sequential(
            nn.Linear(7, 256),
        )

        self.final_model = nn.Sequential(
            nn.Linear(512 + 256, 512),  # Combine image and rest features, output 512-dimensional vector
            nn.ReLU(),
            nn.Linear(512, output_n)
        )

        # Initialization using Xavier uniform
        for layer in [self.model1, self.model3, self.final_model]:
            for sub_layer in layer:
                if isinstance(sub_layer, nn.Linear) or isinstance(sub_layer, nn.Conv2d):
                    nn.init.xavier_uniform_(sub_layer.weight)
                    nn.init.constant_(sub_layer.bias, 0.0)
    
    def forward(self, state):
        rgb_input = state[0]
        rest_input = state[1]

        image_features = self.model1(rgb_input)
        image_features = torch.squeeze(image_features).unsqueeze(0)  # Remove dummy dimensions
        rest_output = self.model3(rest_input).unsqueeze(0)

        combined_features = torch.cat((image_features, rest_output), dim=-1)
        q_values = self.final_model(combined_features)
        return q_values # shape: torch.Size([1, 8, 4])

class QNetwork:
    def __init__(self, env, lr, logdir=None):
        # Define Q-network with specified architecture
        self.net = DQNNetwork(env.action_space.n).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.env = env
        self.lr = lr 
        self.logdir = logdir
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

    def load_model(self, model_file):
        # Load pre-trained model from a file
        return self.net.load_state_dict(torch.load(model_file))

    def load_model_weights(self, weight_file):
        # Load pre-trained model weights from a file
        return self.net.load_state_dict(torch.load(weight_file))
    
class ReplayMemory:
    def __init__(self, env, memory_size=50000, burn_in=10000):
        # Initializes the replay memory, which stores transitions recorded from the agent taking actions in the environment.
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.memory = collections.deque([], maxlen=memory_size)
        self.env = env

    def sample_batch(self, batch_size=32):
        # Returns a batch of randomly sampled transitions to be used for training the model.
        return random.sample(self.memory, batch_size)

    def append(self, transition):
        # Appends a transition to the replay memory.
        self.memory.append(transition)