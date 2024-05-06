import numpy as np
import random
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple
import collections
from env.env_aux.point_net import PointNetfeat


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DQN_Agent:
    def __init__(self, environment_name=None, lr=5e-4, render=False, env=None):
        # Initialize the DQN Agent.
        if env is None:
            self.env = gym.make(environment_name)
        else:
            self.env = env
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Feature extractors
        #   Lidar PointNet
        self.lidar_pointfeat = PointNetfeat(global_feat=True)
        self.lidar_pointfeat.eval().to(self.device)
        #   RGB ResNet50
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True, trust_repo=True)
        self.resnet50.eval().to(self.device)
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])
            
        self.lr = lr
        self.policy_net = QNetwork(self.env, self.lr)
        self.target_net = QNetwork(self.env, self.lr)
        self.target_net.net.load_state_dict(self.policy_net.net.state_dict())  # Copy the weight of the policy network
        self.rm = ReplayMemory(self.env)
        self.burn_in_memory()
        self.batch_size = 16
        self.gamma = 0.99
        self.c = 0

    def burn_in_memory(self):
        # Initialize replay memory with a burn-in number of episodes/transitions.
        cnt = 0
        terminated = False
        truncated = False
        state, _ = self.env.reset()
        # state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        state = self.process_state(state)

        # Iterate until we store "burn_in" buffer
        while cnt < self.rm.burn_in:
            # Reset environment if terminated or truncated
            if terminated or truncated:
                state, _ = self.env.reset()
                # state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                state = self.process_state(state)
            
            # Randomly select an action (left or right) and take a step
            action = self.env.action_space.sample()
            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            reward = torch.tensor([reward])
            if terminated:
                next_state = None
            else:
                # next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                next_state = self.process_state(next_state)
                
            # Store new experience into memory
            transition = Transition(state, torch.tensor([[self.env.action_space.sample()]], dtype=torch.long), next_state, reward)
            self.rm.memory.append(transition)
            state = next_state
            cnt += 1

    def epsilon_greedy_policy(self, q_values, epsilon=0.05):
        # Implement an epsilon-greedy policy. 
        p = random.random()
        if p > epsilon:
            with torch.no_grad():
                return self.greedy_policy(q_values)
        else:
            return torch.tensor([[self.env.action_space.sample()]], dtype=torch.long)

    def greedy_policy(self, q_values):
        # Implement a greedy policy for test time.
        return torch.argmax(q_values)

    def process_state(self, state):  
        rgb_data = torch.from_numpy(state['rgb_data']).float().to(self.device)
        lidar_data = torch.from_numpy(state['lidar_data']).float().to(self.device)
        position = torch.FloatTensor(state['position']).to(self.device)
        situation = torch.FloatTensor([state['situation']]).to(self.device)
        target_position = torch.FloatTensor(state['target_position']).to(self.device)

        lidar_data = lidar_data.unsqueeze(0)
        lidar_data, _, _ = self.lidar_pointfeat(lidar_data)
        lidar_data = lidar_data.squeeze(0)
    
        # RGB (2048,)
        # Assuming rgb_data is a numpy array with shape (360, 640, 3)
        rgb_data = rgb_data.transpose(0, 2)  # Transpose to (3, 360, 640)
        rgb_features = self.resnet50(rgb_data.unsqueeze(0)).squeeze()  # Reshape to (1, 2048) and squeeze to (2048,)
        # create random tensor with shape (2048,)
        # rgb_features = torch.rand(2048).to(self.device)

        rest = torch.cat((position, situation, target_position)).to(self.device)

        return (rgb_features, lidar_data, rest)
        
    def train(self):
        # Train the Q-network using Deep Q-learning.
        state, _ = self.env.reset()
        # state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        state = self.process_state(state)
        terminated = False
        truncated = False

        # Loop until reaching the termination state
        while not (terminated or truncated):
            with torch.no_grad():
                q_values = self.policy_net.net(state)

            # Decide the next action with epsilon greedy strategy
            action = self.epsilon_greedy_policy(q_values).reshape(1, 1)
            
            # Take action and observe reward and next state
            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            reward = torch.tensor([reward])
            if terminated:
                next_state = None
            else:
                # next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                next_state = self.process_state(next_state)

            # Store the new experience
            transition = Transition(state, action, next_state, reward)
            self.rm.memory.append(transition)

            # Move to the next state
            state = next_state

            # Sample minibatch with size N from memory
            transitions = self.rm.sample_batch(self.batch_size)
            batch = Transition(*zip(*transitions))
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # Get current and next state values
            state_action_values = self.policy_net.net(state_batch).gather(1, action_batch) # extract values corresponding to the actions Q(S_t, A_t)
            next_state_values = torch.zeros(self.batch_size)
            
            with torch.no_grad():
                # no next_state_value update if an episode is terminated (next_satate = None)
                # only update the non-termination state values (Ref: https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)
                next_state_values[non_final_mask] = self.target_net.net(non_final_next_states).max(1)[0] # extract max value
                
            # Update the model
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch
            criterion = torch.nn.MSELoss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
            self.policy_net.optimizer.zero_grad()
            loss.backward()
            self.policy_net.optimizer.step()

            # Update the target Q-network in each 50 steps
            self.c += 1
            if self.c % 50 == 0:
                self.target_net.net.load_state_dict(self.policy_net.net.state_dict())
    
    def test(self, model_file=None):
        # Evaluates the performance of the agent over 20 episodes.

        max_t = 1000
        state, _ = self.env.reset()
        rewards = []

        for t in range(max_t):
            # state = torch.from_numpy(state).float().unsqueeze(0)
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
    def __init__(self, action_space):
        super(DQNNetwork, self).__init__()

        # Define the neural network architecture
        # RGB: 2048
        self.model1 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )
        
        # Lidar: 1024
        self.model2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Rest: 7
        self.model3 = nn.Sequential(
            nn.Linear(7, 256),
        )
        
        # Join the three models: 512 + 256 + 256 = 1024
        self.final_model = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_space.n)
        )

        # Initialization using Xavier uniform (a popular technique for initializing weights in NNs)
        for layer in [self.model1, self.model2, self.model3, self.final_model]:
            for sub_layer in layer:
                if isinstance(sub_layer, nn.Linear):
                    nn.init.xavier_uniform_(sub_layer.weight)
                    nn.init.constant_(sub_layer.bias, 0.0)

    def forward(self, inputs):
        # Forward pass
        rgb_value = self.model1(inputs[0])
        lidar_value = self.model2(inputs[1])
        rest_value = self.model3(inputs[2])
        return self.final_model(torch.cat([rgb_value, lidar_value, rest_value], dim=1))

    
class QNetwork:
    def __init__(self, env, lr, logdir=None):
        # Define Q-network with specified architecture
        self.net = DQNNetwork(env.action_space)
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