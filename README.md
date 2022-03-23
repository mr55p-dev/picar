# picar
Requires [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

## Setup

```sh
# NOTE: if running ploomber <0.16, remove the --create-env argument
ploomber install --create-env
# activate conda environment
conda activate picar

```

## Code editor integration

* If using Jupyter, [click here](https://docs.ploomber.io/en/latest/user-guide/jupyter.html)
* If using VSCode, PyCharm, or Spyder, [click here](https://docs.ploomber.io/en/latest/user-guide/editors.html)



## Running the pipeline

Create an `env.yaml` file which defines the following:
```yaml
n_samples:	<number of items to take from the training data>
n_epochs: 	<number of epochs to train for>
batch_size:	<number of elements in each batch>
```

```sh
ploomber build
```

## Help

* Need help? [Ask us anything on Slack!](https://ploomber.io/community)
