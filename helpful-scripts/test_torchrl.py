from collections import defaultdict
import multiprocessing

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

from env.environment import CarlaEnv

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_fork = multiprocessing.get_start_method() == "fork"
num_cells = 256 # Number of cells in the hidden layer
lr = 3e-4 # Learning rate
max_grad_norm = 1.0 # Maximum gradient norm
frames_per_batch = 1000
total_frames = 10000
sub_batch_size = 64
num_epochs = 10
clip_epsilon = (0.2)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4 

# Environment Setup
# You can also create a gym env variable and then wrap it in the torchrl's GymEnv
base_env = GymEnv('carla-rl-gym-v0', device=device)
# base_env = GymEnv("InvertedDoublePendulum-v4", device=device)

env = TransformedEnv(
    base_env,
    Compose(
        # Normalize observations
        ObservationNorm(in_keys=["observation"]),
        DoubleToFloat(),
        StepCounter(),
    ),
)

env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

check_env_specs(env)