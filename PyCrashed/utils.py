from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import wandb

from PyCrashed.data import Data, clean_predictions
from PyCrashed.models import (BaseModel, NVidia, NVidiaBig, MultiHeaded, Resnet50Img, Resnet50Bare, Resnet101Img, Efficientnet, InceptionResnet)


def get_printf(verbose):
    """Returns the print function to use"""
    return print if verbose else lambda *args, **kwargs: None


# Each model from PyCrashed.models must be added here, kind of laborious but ust easier
models = {
    "nvidia": NVidia,
    "nvidia_big": NVidiaBig,
    "multiheaded": MultiHeaded,
    "efficientnet": Efficientnet,
    "resnet_50_imagenet": Resnet50Img,
    "resnet_50_bare": Resnet50Bare,
    "resnet_101_imagenet": Resnet50Img,
    "inceptionresnet": InceptionResnet,
}

def list_models(args):
    """Either prints out the models available, or compiles and summarises a specific model"""
    if args.model:
        models[args.model](use_wandb=False).build().summary()
    else:
        print('\n'.join(models.keys()))

def restore_model(args):
    """Returns a keras model from a path"""
    model: BaseModel = models[args.model]()
    model.build()
    model.restore(args.path)
    return model

def train_model(args):
    """Trains a model based on the argparse arguments passed"""
    wandb.init(project="PyCrashed", entity="mr55p", config=args)

    printf = get_printf(args.verbose)

    # Find the device GPUs and make them available for the mirrored strategy
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)

    # Compile the model within the scope
    printf("Instantiating model... ", end="")
    with strategy.scope():
        model: BaseModel = models[args.model](
            verbose=args.verbose,
            paitence=args.paitence or args.epochs,
            dropout_rate=args.dropout,
            activation=args.activation,
        )

        # Change these properties also in scope if they are defined
        if args.loss:       model.set_loss(args.loss)
        if args.optimizer:  model.set_optimizer(args.optimizer)
        if args.activation: model.set_activation(args.activation)

        # Call model.compile()
        model.build()

        # if args.restore: model.restore()
    printf("Done!")

    # Setup the training and validation datasets
    printf("Configuring data pipeline... ", end="")
    batch = args.batch * strategy.num_replicas_in_sync
    train_ds, val_ds = Data.training(args.train, args.val, batch , multiheaded=model.is_split)
    printf("Done!")

    # Fit the model
    printf("Training model")
    model.fit(n_epochs=args.epochs, data=train_ds, validation_data=val_ds)

    printf("Saving model")
    model.save()

def predict(args):
    """Perform inference using a model at the specified path"""
    printf = get_printf(args.verbose)

    # Load a model
    model_path = Path(args.path)
    print(model_path)
    model = tf.keras.models.load_model(str(model_path))

    # Load the testing dataset, with a batch size of 1
    kaggle_dataset = Data.testing(batch_size=1)

    # Make the predictions
    predictions = model.predict(kaggle_dataset)

    # Convert multiheaded output back into a (N, 2) vector
    if isinstance(predictions, tuple):
        predictions = np.hstack(predictions)

    # Clean the predictions
    predictions = clean_predictions(predictions)

    # Create a dataframe
    predictions = pd.DataFrame(
        predictions,
        index=pd.RangeIndex(1, 1021),
        columns=["angle", "speed"]
    )
    predictions.index.name = "image_id"
    predictions["speed"] = predictions["speed"].astype("int")

    # Write out to file
    output_path = args.output or Path.joinpath(model_path.parent, "predictions.csv")
    predictions.to_csv(output_path)
    printf("Done!")

def convert(args):
    """Load, convert and write a model tflite binary"""
    model_path = Path(args.path)
    output_file = args.output or Path.joinpath(model_path.parent, "model.tflite")
    model_converter = tf.lite.TFLiteConverter.from_saved_model(str(model_path))
    model = model_converter.convert()
    with open(str(output_file), 'wb') as f:
        f.write(model)
