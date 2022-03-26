# %%
from PyCrashed.models import ImageNetPretrained
# %%
model_imgnet = ImageNetPretrained()
model_imgnet.build().summary()
# %%
model_imgnet.fit(n_epochs=1)
model_imgnet.test()
# %%
model_imgnet.save()
# %%
model_imgnet.fit_metrics.history
# %%
