# Isospectral Neural Networks

## Requirements

The code depends on jax:
```
pip install --upgrade -q https://storage.googleapis.com/jax-wheels/cuda$(echo $CUDA_VERSION | sed -e 's/\.//' -e 's/\..*//')/jaxlib-0.1.11-cp36-none-linux_x86_64.whl
pip install --upgrade -q jax
```

## Example of use

The following code learns how to classify the MNIST dataset with a neural network with the following architecture

```
input image --> Isospectral ODE layer on a 10x10 Matrix --> Dense Layer of size 10 --> Softmax
```
and stores tensorboard results in `/tmp/checkpoints`:

```
python learn_mnist.py --tag="some_name_for_tensorboard_logging"\
                       --layers="[Flatten, IsospectralODE(10), Flatten, Dense(10), Softmax]"\
                        --num_epochs=5
                        ```
