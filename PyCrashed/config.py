import os

n_train = os.getenv("N_TRAIN", 2000)
n_val = os.getenv("N_VAL", 200)
n_test = os.getenv("N_TEST", 5000)
batch_size = os.getenv("BATCH_SIZE", 8)
