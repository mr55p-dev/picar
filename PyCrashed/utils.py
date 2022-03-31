from pathlib import Path
from PyCrashed.models import NVidia, MobileNetPT, MultiHeaded, NVidiaSplit, ResNetPT, EfficientNetPT, Model
from PyCrashed.pipeline import Dataset

import numpy as np
import pandas as pd
import tensorflow as tf

import wandb
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
    "nvidia_split": NVidiaSplit,
    "mobilenet": MobileNetPT,
    "efficientnet": EfficientNetPT,
    "resnet": ResNetPT,
    "multiheaded": MultiHeaded,
}

def list_models(args):
    if args.model:
        models[args.model](use_wandb=False).build().summary()
    else:
        print('\n'.join(models.keys()))

def restore_model(args):
    model: Model = models[args.model]()
    model.build()
    model.restore(args.path)

def train_model(args):
    wandb.init(project="PyCrashed", entity="mr55p", config=args)

    printf = get_printf(args.verbose)

    # Find the device GPUs and make them available for the mirrored strategy
    # tf.debugging.set_log_device_placement(True) # Enable device placement debug messages
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)

    printf("Configuring data pipeline... ", end="")
    Dataset.n_train = args.train
    Dataset.n_val = args.val
    Dataset.n_test = 1 - (args.train + args.val)
    Dataset.batch_size = strategy.num_replicas_in_sync * args.batch
    ds = Dataset.load("train")
    printf("Done!")

    # Compile the model within the scope
    printf("Building model... ", end="")
    printf("Instantiating model... ", end="")
    with strategy.scope():
        model: Model = models[args.model](
            verbose=args.verbose,
            paitence=args.paitence or args.epochs,
            kernel_width=args.kernel_width,
            network_width=args.network_width,
            dropout_rate=args.dropout,
            activation=args.activation,
        )

        # Change these properties also in scope if they are defined
        if args.loss:       model.set_loss(args.loss)
        if args.optimizer:  model.set_optimizer(args.optimizer)
        if args.activation: model.set_activation(args.activation)

        # Call model.compile()
        model.build()

        if args.restore: model.restore()
    printf("Done!")

    fmt = lambda k: k.replace('_', ' ').title()
    printf("Executing model using the following dataset configuration")
    # printf(tabulate({fmt(k): [v] for k, v in Dataset._props.items()}, headers="keys", tablefmt="fancy_grid"))

    printf("Training model")
    model.fit(n_epochs=args.epochs, data=ds)

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
