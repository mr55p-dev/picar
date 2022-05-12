import cv2
from numpy import clip, expand_dims, rint
# import tensorflow.lite as tflite
# from tensorflow.lite.experimental import load_delegate
import tflite_runtime.interpreter as tflite
from tflite_runtime.interpreter import load_delegate


class Model:
    __slots__ = ['interpreter', 'inp_tensor_idx', 'out_angle_tensor_idx', 'out_speed_tensor_idx']
    def __init__(self):
        model_path = "./autopilot/models/pycrashed/model.tflite"
        self.interpreter = tflite.Interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.inp_tensor_idx = self.interpreter.get_input_details()[0]["index"]
        self.out_angle_tensor_idx = self.interpreter.get_output_details()[0]["index"]

    def predict(self, img):
        #if not img:
            #print("No image passed")
            #return 0, 90
        img = img.astype('float32')
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img = expand_dims(img, 0)

        self.interpreter.set_tensor(self.inp_tensor_idx, img)
        self.interpreter.invoke()
        pred_angle = self.interpreter.get_tensor(self.out_angle_tensor_idx).squeeze()
        angle, speed = clip(pred_angle, 0, 1)
        speed = rint(speed)

        angle = (angle * 80) + 50
        speed = speed * 35

        return angle, speed

