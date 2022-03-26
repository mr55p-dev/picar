import argparse
from PyCrashed.models import NVidia, ImageNetPretrained, MultiHeaded, Model

if __name__ == "__main__":
    models = {
        "nvidia": NVidia,
        "imagenet": ImageNetPretrained,
        "multiheaded": MultiHeaded
    }

    parser = argparse.ArgumentParser(description='Train some models')
    parser.add_argument("model")
    args = parser.parse_args()

    print("Instantiating model")
    model: Model = models[args.model]

    print("Building model")
    model.build().summary()

    print("Training model")
    model.fit()

    print("Testing model")
    model.test()

    print("Saving model")
    model.save()

