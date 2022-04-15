# %%
from tensorflow import lite as tflite
from tensorflow import image
import tensorflow as tf
from PyCrashed import predict
from PyCrashed.predict import Data

# %%
ds, _ = Data.training(0.7, 0.3, 8, False, False)
# %%
tf.keras.layers.BatchNormalization()
# %%
batch = next(iter(ds))
# %%
loss = tf.keras.losses.BinaryCrossentropy()
# %%
class Model:
    __slots__ = ['interpreter', 'inp_tensor_idx', 'out_tensor_idx']
    def __init__(self):
        self.interpreter = tflite.Interpreter("products/Nvidia/model.tflite")
        self.interpreter.allocate_tensors()
        self.inp_tensor_idx = self.interpreter.get_input_details()[0]["index"]
        self.out_tensor_idx = self.interpreter.get_output_details()[0]["index"]

    def predict(self, img):
        img = image.convert_image_dtype(img, tf.float32)
        img = image.resize(img, (224, 224))
        img = image.rgb_to_yuv(img)
        img = tf.expand_dims(img, axis=0)

        self.interpreter.set_tensor(self.inp_tensor_idx, img)
        self.interpreter.invoke()
        pred = self.interpreter.get_tensor(self.out_tensor_idx).squeeze()

        angle, speed = tf.clip_by_value(pred, 0, 1)
        speed = tf.math.rint(speed)

        angle = (angle * 80) + 50
        speed = speed * 35

        return angle, speed
# %%
m = Model()
m.predict(image.decode_png(tf.io.read_file("data/training_data/training_data/1.png")))
# %%
