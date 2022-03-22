# %% tags=["parameters"]
# declare a list tasks whose products you want to use as inputs
upstream = None

# %%
from math import ceil
import numpy as np
import pandas as pd
import cv2
import pathlib
from tqdm.notebook import tqdm

# %%
N_SAMPLES = 200
SCALE = .125

# %%
image_labels = pd.read_csv("data/training_norm.csv")
image_dir = pathlib.Path("data/training_data/training_data/")
image_paths = [
        (int(f.name[:-4]), f)
        for f in image_dir.glob("*.png")
        if f.stat().st_size > 0
]

# Get an index of the files (to be used with the image labels)
index = np.array(sorted([i[0] for i in image_paths]), dtype="int")
assert (image_labels["image_id"].values == index).all()

# Allow n to be a fraction or integer number
label_matrix = image_labels.values
n_vals = index.shape[0]
if N_SAMPLES == 0:
    instances = n_vals
elif N_SAMPLES < 1:
    instances = int(ceil(n_vals * N_SAMPLES))
else:
    instances = N_SAMPLES


# Select some random indices
rng = np.random.default_rng()
subset_indices = rng.choice(index, size=instances, replace=False).astype("float")

# Take the rows corresponding to those indices (how is this so hard??)
s_labels = np.delete(
    label_matrix[            # Get only the elements of label_matrix matching the following
        np.apply_along_axis( # Generate a boolean mask of elements which have been selected or not
            lambda x: x[0] in subset_indices, 1, label_matrix
        )
    ], 0, 1                  # Delete the 0th element from axis 0 (the "image_id" column which we no longer need)
)

# Take the file paths corresponding to those indices
s_paths = filter(lambda x: x[0] in subset_indices, image_paths)

img_shape = list((SCALE * np.array([320, 480], dtype="int")).astype("int"))

# Load images from the list of paths
def load_img(p):
    return cv2.resize(
        cv2.cvtColor(
            cv2.imread(str(p[1])), # Load the image from the file path
            cv2.COLOR_RGB2YUV     # Flatten the colours into grayscale
        ),
        img_shape                    # Resize the image to the "RESCALE" shape
    ) / 255 # Normalize the image

s_img = [load_img(f) for f in tqdm(s_paths, total=instances)]
s_img = np.swapaxes(np.stack(s_img), 1, 2)

# %%
s_img.shape

# %%
np.save("products/img-loaded", s_img)
np.save("products/lab-loaded", s_labels)
