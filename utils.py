import array
import functools as ft
import gzip
import os
import struct
import urllib.request

import jax
import jax.numpy as jnp

def load_mnist(dtype = jnp.uint8):
    filename = "train-images-idx3-ubyte.gz"
    url_dir = "https://storage.googleapis.com/cvdf-datasets/mnist"
    target_dir = os.getcwd() + "/data/mnist"
    url = f"{url_dir}/{filename}"
    target = f"{target_dir}/{filename}"

    if not os.path.exists(target):
        os.makedirs(target_dir, exist_ok=True)
        urllib.request.urlretrieve(url, target)
        print(f"Downloaded {url} to {target}")

    with gzip.open(target, "rb") as fh:
        _, batch, rows, cols = struct.unpack(">IIII", fh.read(16))
        shape = (batch, 1, rows, cols)
        return jnp.array(array.array("B", fh.read()), dtype=dtype).reshape(shape)


def dataloader(data, batch_size, *, key):
    '''todo - managing labels!'''
    dataset_size = data.shape[0]
    indices = jnp.arange(dataset_size)
    while True:
        key, subkey = jax.random.split(key, 2)
        perm = jax.random.permutation(subkey, indices)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield data[batch_perm]
            start = end
            end = start + batch_size

def dataloader_with_labels(data, labels, batch_size, *, key):
    '''todo - managing labels!'''
    dataset_size = data.shape[0]
    indices = jnp.arange(dataset_size)
    while True:
        key, subkey = jax.random.split(key, 2)
        perm = jax.random.permutation(subkey, indices)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield data[batch_perm], labels[batch_perm]
            start = end
            end = start + batch_size


if __name__ == '__main__':
    data = load_mnist()
    print(data.shape)