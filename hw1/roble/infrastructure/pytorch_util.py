from typing import Union

import torch
from torch import nn

Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


def build_mlp(
        input_size: int,
        output_size: int,
        **kwargs
    ):
    """
    Builds a feedforward neural network

    arguments:
    n_layers: number of hidden layers
    size: dimension of each hidden layer
    activation: activation of each hidden layer
    input_size: size of the input layer
    output_size: size of the output layer
    output_activation: activation of the output layer

    returns:
        MLP (nn.Module)
    """
    if isinstance(kwargs["params"]["output_activation"], str):
        output_activation = _str_to_activation[kwargs["params"]["output_activation"]]

    mlp = nn.Sequential()

    n_lyrs = len(kwargs["params"]["layer_sizes"])
    for i, lyr_sz, act in zip(range(n_lyrs), kwargs["params"]["layer_sizes"], kwargs["params"]["activations"]):
        mlp.add_module(f"lyr{i}", nn.Linear(input_size, lyr_sz))
        activation = _str_to_activation[act]
        mlp.add_module(f"act{i}", activation)
        input_size = lyr_sz

    mlp.add_module("out_lyr", nn.Linear(lyr_sz, output_size))
    mlp.add_module(f"out_act", output_activation)
    return mlp

device = None

def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")



def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)

def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.to('cpu').detach().numpy()
    else:
        raise TypeError("Expected input type torch.Tensor, but got {}".format(type(tensor)))
