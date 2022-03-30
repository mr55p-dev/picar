# %%
from PyCrashed.pipeline import Dataset
# %%
ds = Dataset.load("train")
# %%
len(list(ds))

# %%
ds2 = Dataset.load("val")
len(list(ds2))
# %%
i = i[0, :, :, :]
# %%
import matplotlib.pyplot as plt
# %%
plt.imshow(i)
# %%
