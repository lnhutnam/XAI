# Dataset
BATCH_SIZE = 64  # In Julia, we don't multiply by GPU count automatically; adjust as needed

# Architecture
num_features = 128*128
num_classes = 2

# Hyperparameters
random_seed = 1
learning_rate = 0.001
num_epochs = 3