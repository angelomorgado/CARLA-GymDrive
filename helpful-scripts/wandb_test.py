import wandb
import random

# Initialize wandb
wandb.init(project='simple_wandb_test')

# Generate random data and log it to wandb
for i in range(100):
    random_number = random.random()  # Generate random number
    wandb.log({"random_number": random_number, "iteration": i})  # Log random number and iteration

# Finish logging
wandb.finish()