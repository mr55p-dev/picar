# %%
from PyCrashed.pipeline import Dataset
# %%
ds = Dataset.load("train")
# %%
x = next(iter(ds.take(1)))

# %%
i = x[0]
# %%
i = i[0, :, :, :]
# %%
import matplotlib.pyplot as plt
# %%
plt.imshow(i)
# %%
