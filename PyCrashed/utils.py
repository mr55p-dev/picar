from pathlib import Path
from PyCrashed.models import NVidia, MobileNetPT, MultiHeaded, NVidiaSplit, ResNetPT, EfficientNetPT, Model
from PyCrashed.predict import Data

import numpy as np
import pandas as pd
import tensorflow as tf

import wandb

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

    # Compile the model within the scope
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

    printf("Configuring data pipeline... ", end="")
    train_ds, val_ds = Data.training(args.train, args.val, args.batch)
    printf("Done!")

    printf("Training model")
    model.fit(n_epochs=args.epochs, data=train_ds, validation_data=val_ds)

    printf("Saving model")
    model.save()

def predict(args):
    printf = get_printf(args.verbose)

    # Load a model
    model_path = Path(args.path)
    model = tf.keras.models.load_model(model_path)

    # Load the correct dataset
    kaggle_dataset = Data.testing(1)

    # Make the predictions
    predictions = model.predict(kaggle_dataset)

    # Format the predictions
    predictions = tf.clip_by_value(predictions, 0, 1)
    predictions = np.stack((predictions[:, 0], np.rint(predictions[:, 1]))).T

    # Bring the predictions dataframe
    predictions = pd.DataFrame(
        predictions,
        index=pd.RangeIndex(1, 1021),
        columns=["angle", "speed"]
    )
    predictions.index.name = "image_id"
    predictions["speed"] = predictions["speed"].astype("int")

    # Write out to file
    output_path = args.output or Path.joinpath(model_path.parent, "predicions.csv")
    predictions.to_csv(output_path)
    printf("Done!")
