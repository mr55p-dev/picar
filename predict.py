# %%
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
import cv2

# %%
model = tf.keras.models.load_model("products/Nvidia_split/model")
# %%
def get_image_id(path: Path):
    name = 
    return 
# %%
file_path = Path("data/test_data/test_data/")
files = sorted(list(file_path.glob("*.png")), key=get_image_id)
# %%
def open_image(path):
    img_decoded = tf.image.decode_image(
        tf.io.read_file(str(path)), channels=3, dtype=tf.float32
    )
    return tf.image.resize(tf.image.rgb_to_yuv(img_decoded), (224, 224))
# %%
images = map(open_image, files)
# %%
img1 = next(images)
# %%
img1
# %%
img1_proper = open_image()