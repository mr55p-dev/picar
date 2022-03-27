import tensorflow as tf
import numpy as np
from PyCrashed.pipeline import Dataset

# x = np.zeros((5, 5))
# y = x + 1
# z = x + 2

# arr = np.stack([x, y, z])

# print("Initial array")
# print(arr)
# print(arr.shape)

# arr = tf.convert_to_tensor(arr)

# bnorm = tf.keras.layers.BatchNormalization()(arr)
# print("Normalized array")
# print(bnorm.numpy())

d = next(iter(Dataset.load("train").take(1)))[0][0, :, :, :]
print(d.shape)

# inp = tf.keras.layers.Resizing(4, 4)(d)
print(d)
print(d.shape)
inp1 = tf.keras.layers.BatchNormalization(axis=0)(d)
inp2 = tf.keras.layers.BatchNormalization(axis=1)(d)
inp3 = tf.keras.layers.BatchNormalization(axis=2)(d)

print((inp1 == inp2).numpy().all())

