# %%
from pathlib import Path
from typing import Tuple
import re
import pandas as pd

# %%
training_dir = Path("data/training_data")

def normalize_angle(angle: int) -> float:
    return (angle - 50) / 80

def extract_metadata(path: Path) -> Tuple:
    timestamp, angle, speed = re.search(r'(\d+)_(\d+)_(\d)', path.name).groups()
    return int(timestamp), normalize_angle(int(angle)), int(speed), str(path.relative_to(training_dir))

def extract_files(path: Path, start: int) -> pd.DataFrame:
    files = path.glob('*.png')
    iterable = list(map(extract_metadata, files))
    return pd.DataFrame(
        iterable,
        columns=["timestamp", "angle", "speed", "path"],
        index=pd.RangeIndex(
            start=start,
            stop=start + len(iterable)
            )
        )

# %%
# Get all the subdirs
subdirs = filter(lambda x: x.is_dir() and x.name != "training_data", training_dir.iterdir())

# Find the end of the base training data
base_data = Path("data/training_data/training_data")
start = max(map(lambda x: int(x.name.split('.')[0]), base_data.glob('*.png')))

# Load the training data labels
df = pd.read_csv(Path("data/training_norm.csv"))
df["path"] = df["image_id"].apply(lambda x: f"training_data/{x}.png")
df.drop("image_id", inplace=True, axis=1)

for subdir in subdirs:
    new_df = extract_files(subdir, start)
    df = pd.concat([df, new_df])
    start = df.index.max()