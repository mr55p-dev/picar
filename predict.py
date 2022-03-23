# %% tags=["parameters"]
# declare a list tasks whose products you want to use as inputs
upstream = ['load-data', 'nvidia-model']

# %%
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
import cv2

# %%
images = np.load(upstream['load-data']['images'])
labels = np.load(upstream['load-data']['labels'])
model = tf.keras.models.load_model(upstream['nvidia-model']['model'])

# %%
def show(idx: int) -> None:
    img_data = images[idx-1, :, :, :]
    img = cv2.cvtColor(
        (np.swapaxes(img_data, 1, 0) * 255).astype('uint8'),
        cv2.COLOR_YUV2RGB
    )
    pred = model.predict(images[idx-1:idx, :, :, :])
    print(f"ACTUAL:\tSpeed: {labels[idx-1, 0]:.4f}\tSteering angle: {labels[idx-1, 1]:.4f}")
    print(f"PRED:\tSpeed: {pred[0, 1]:.4f}\tSteering angle: {pred[0, 1]:.4f}")
    plt.imshow(img)

# %%
widgets.interact(show, idx=widgets.IntSlider(min=1, max=images.shape[0]))
