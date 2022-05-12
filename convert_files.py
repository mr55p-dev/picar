# %%
from pathlib import Path
from typing import Tuple
import re
import pandas as pd

# %%
# Get the files
paths = Path("data/Track1").glob("*.png")
data = []

# %%
# Rename all the files
for idx, f in enumerate(paths, start=40000):
    (angle, speed), = re.findall(r"\d+_(\d+)_(\d+)\.png", f.name)

    angle = ( int(angle) - 50 ) / 80
    speed = int(speed)
    speed = 1 if speed else 0

    data.append((idx, angle, speed))

    f.rename(f"{idx}.png")

# %%
d = pd.DataFrame(data, columns=["image_id", "angle", "speed"])
d.to_csv("track_1.csv")

# %%
