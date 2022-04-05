# %%
import numpy as np
import pandas as pd
import tensorflow as tf

from PyCrashed.data import Data
# %%
t, _ = Data.training(.7, .3, 4)
# %%
item = next(iter(t))
# %%
img, lab, weight = item
# %%
