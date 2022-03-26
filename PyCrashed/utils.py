from pathlib import Path
from PyCrashed.models import NVidia, ImageNetPretrained, MultiHeaded, Model
from PyCrashed.pipeline import Dataset

import numpy as np
import pandas as pd
import tensorflow as tf

from tabulate import tabulate


def normal_to_raw(x: np.array) -> np.array:
    angle = np.rint((80 * x[0]) + 50)
    angle = 5 * np.rint(angle / 5)
    speed = 35 * np.rint(x[1]).astype(int)
    return np.array([angle, speed])

def raw_to_normal(x: np.array) -> np.array:
    angle = (x[0] - 50) / 80
    speed = x[1] / 35
    return np.array([angle, speed])

def get_printf(verbose):
    return print if verbose else lambda *args, **kwargs: None


models = {
    "nvidia": NVidia,
    "imagenet": ImageNetPretrained,
    "multiheaded": MultiHeaded
}

def list_models(args):
    if args.model:
        models[args.model]().build().summary()
    else:
        print('\n'.join(models.keys()))

def train_model(args):
    printf = get_printf(args.verbose)

    printf("Configuring data pipeline... ", end="")
    if args.n_train:
        Dataset.set("N_TRAIN", args.n_train)
    if args.batch:
        Dataset.set("N_VAL", args.n_val)
    if args.n_val:
        Dataset.set("BATCH_SIZE", args.batch)
    printf("Done!")

    printf("Instantiating model... ", end="")
    model: Model = models[args.model](
        use_logging=args.logging,
        use_early_stopping=args.earlystopping,
        use_checkpoints=args.checkpoints,
        verbose=args.verbose
    )
    printf("Done!")

    printf("Building model... ", end="")
    m = model.build()
    printf("Done!")

    fmt = lambda k: k.replace('_', ' ').title()
    printf("Executing model using the following dataset configuration")
    printf(tabulate({fmt(k): [v] for k, v in Dataset._props.items()}, headers="keys", tablefmt="fancy_grid"))

    if args.train:
        printf("Training model")
        model.fit(n_epochs=args.epochs)

    if args.test and args.train:
        printf("Testing model")
        model.test()

    if args.save and args.train:
        printf("Saving model")
        model.save()

def predict(args):
    printf = get_printf(args.verbose)

    # Load a model
    printf("Loading model... ", end="")
    model_path = Path(args.path)
    model = tf.keras.models.load_model(model_path)

    printf("Done!")
    printf("Loading dataset... ", end="")
    # Load the correct dataset
    ds = Dataset.load_test()
    printf("Done!")

    # Perform inference
    printf("Performing inference... ", end="")
    predictions = model.predict(ds)
    printf("Done!")

    # # Adjust values
    printf("Adjusting values... ", end="")
    predictions = np.apply_along_axis(normal_to_raw, 1, predictions)
    predictions = np.apply_along_axis(raw_to_normal, 1, predictions)
    printf("Done!")

    # Write to csv
    printf("Writing output... ", end="")
    output_path = args.output or Path.joinpath(model_path.parent, "predicions.csv")
    df = pd.DataFrame(
        predictions,
        columns=["angle", "speed"],
        index=pd.RangeIndex(1, predictions.shape[0] + 1, name="image_id")
    )
    df["speed"] = df["speed"].round(0).astype(int)
    df.to_csv(output_path)
    printf("Done!")
