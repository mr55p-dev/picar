# %%
import tensorflow as tf
from PyCrashed.data import Data
# %%
i = tf.keras.Input(shape=(224, 224, 3))
l = tf.keras.layers.BatchNormalization()(i)
l = tf.keras.layers.Conv2D(24, (5, 5), strides=(2, 2))(l)
l = tf.keras.layers.Activation('relu')(l)
l = tf.keras.layers.Conv2D(36, (5, 5), strides=(2, 2))(l)
l = tf.keras.layers.Activation('relu')(l)
l = tf.keras.layers.Conv2D(48, (5, 5), strides=(2, 2))(l)
l = tf.keras.layers.Activation('relu')(l)
l = tf.keras.layers.MaxPool2D((2, 2))(l)
l = tf.keras.layers.BatchNormalization()(l)
l = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1))(l)
l = tf.keras.layers.Activation('relu')(l)
l = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1))(l)
l = tf.keras.layers.Activation('relu')(l)
l = tf.keras.layers.MaxPool2D((2, 2))(l)

l = tf.keras.layers.Flatten()(l)

l = tf.keras.layers.Dense(128)(l)
l = tf.keras.layers.Activation('relu')(l)
l = tf.keras.layers.Dropout(0.25)(l)
l = tf.keras.layers.Dense(64)(l)
l = tf.keras.layers.Activation('relu')(l)
o = tf.keras.layers.Dense(2)(l)
# %%
train, val = Data.training(.75, .25, 64)

def angle_filter(img, label, weight):
    return img, label[:, 0:1], weight
def speed_filter(img, label, weight):
    return img, label[:, 1:2], weight

train_angle = train.map(angle_filter)
train_speed = train.map(speed_filter)
val_angle = val.map(angle_filter)
val_speed = val.map(speed_filter)
# %%
model_angle = tf.keras.Model(inputs=i, outputs=o, name="angle_model")
model_angle.compile(
    optimizer="adam",
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.RootMeanSquaredError()],
)
model_speed = tf.keras.Model(inputs=i, outputs=o, name="speed_model")
model_speed.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy()],
)
# %%
fit_metrics_angle = model_angle.fit(
    train_angle,
    epochs=1,
    validation_data=val_angle
)
fit_metrics_speed = model_speed.fit(
    train_speed,
    epochs=2,
    validation_data=val_speed
)
# %%
