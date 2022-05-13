import cv2
from numpy import clip, expand_dims, rint
# import tensorflow.lite as tflite
# from tensorflow.lite.experimental import load_delegate
import tflite_runtime.interpreter as tflite
from tflite_runtime.interpreter import load_delegate


class Model:
    __slots__ = ['interpreter', 'inp_tensor_idx', 'out_tensor_idx']
    def __init__(self):
        model_path = "./autopilot/models/pycrashed/model.tflite"
        self.interpreter = tflite.Interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.inp_tensor_idx = self.interpreter.get_input_details()[0]["index"]
        self.out_tensor_idx = self.interpreter.get_output_details()[0]["index"]

    def predict(self, img):
        img = img.astype('float32')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img = expand_dims(img, 0)

        self.interpreter.set_tensor(self.inp_tensor_idx, img)
        self.interpreter.invoke()
        pred = self.interpreter.get_tensor(self.out_tensor_idx).squeeze()
        print(pred)
        angle, speed = clip(pred, 0, 1)
        speed = 1 if speed > 0.4 else 0

        angle = (angle * 80) + 50
        speed = speed * 35
        print((angle, speed))

        return angle, speed

