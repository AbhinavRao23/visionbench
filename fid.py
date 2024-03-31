from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax  
import einops

from utils import load_mnist, dataloader