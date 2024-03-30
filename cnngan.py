from functools import partial

from typing import Union, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax  
import einops

from utils import load_mnist, dataloader

LATENT_SIZE = 100

class Generator(eqx.Module):
    layers: list
    def __init__(
        self,
        input_shape: int,
        output_shape: tuple[int, int, int],
        key: jax.random.PRNGKey):

        channels, height, width = output_shape
        keys = jax.random.split(key, 5)
        
        ConvT = partial(eqx.nn.ConvTranspose2d, kernel_size=4, use_bias=False)
        self.layers = [
            ConvT(input_shape, width * 8, stride=2, padding=0, key=keys[0]),
            eqx.nn.BatchNorm(input_size=width * 8, axis_name="batch"),
            jax.nn.relu,
            ConvT(width * 8, width * 4, stride=2, padding=0, key=keys[1]),
            eqx.nn.BatchNorm(input_size=width * 4, axis_name="batch"),
            jax.nn.relu,
            ConvT(width * 4, width * 2, stride=2, padding=0, key=keys[2]),
            eqx.nn.BatchNorm(input_size=width * 2, axis_name="batch"),
            jax.nn.relu,
            ConvT(width * 2, width, stride=1, padding=0, key=keys[3]),
            eqx.nn.BatchNorm(input_size=width, axis_name="batch"),
            jax.nn.relu,
            ConvT(width, channels, stride=1, padding=0, key=keys[4]),
            jax.nn.tanh]

    def __call__(self, x, state):
        for layer in self.layers:
            if isinstance(layer, eqx.nn.BatchNorm):
                x, state = layer(x, state)
            else:
                x = layer(x)

        return x, state
    
class Discriminator(eqx.Module):
    layers: list[Union[eqx.nn.Conv2d, eqx.nn.PReLU, eqx.nn.BatchNorm, Callable]]

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        key: jax.random.PRNGKey):

        channels, height, width = input_shape
        keys = jax.random.split(key, 5)
        Conv = partial(eqx.nn.Conv2d, kernel_size=4, stride=2, use_bias=False)

        self.layers = [
            Conv(channels, width, padding=1, key=keys[0]), # 28 14 14
            eqx.nn.PReLU(0.2),
            Conv(width, width * 2, padding=1, key=keys[1]), # 56 7 7
            eqx.nn.BatchNorm(width * 2, axis_name="batch"),
            eqx.nn.PReLU(0.2),
            Conv(width * 2, width * 4, padding=1, key=keys[2]), # 112 3 3
            eqx.nn.BatchNorm(width * 4, axis_name="batch"),
            eqx.nn.PReLU(0.2),
            Conv(width * 4, width * 8, padding=1, key=keys[3]), # 224 1 1
            eqx.nn.BatchNorm(width * 8, axis_name="batch"),
            eqx.nn.PReLU(0.2),
            Conv(width * 8, 1, padding=2, key=keys[4]), # 1 1 1
        ]

    def __call__(self, x, state):
        for layer in self.layers:
            if isinstance(layer, eqx.nn.BatchNorm):
                x, state = layer(x, state=state)
            else:
                x = layer(x)
        return x, state
    
def loss_d(discriminator, generator, discriminator_state, generator_state, 
           real_batch, key):
    
    batch_size = real_batch.shape[0]
    fake_labels = jnp.zeros(batch_size)
    real_labels = jnp.ones(batch_size)
    
    noise = jax.random.normal(key, (batch_size, LATENT_SIZE, 1, 1))
    fake_batch, generator_state = jax.vmap(
        generator, axis_name="batch", in_axes=(0, None), out_axes=(0, None))(
            noise, generator_state)

    pred_y, discriminator_state = jax.vmap(
        discriminator, axis_name="batch", in_axes=(0, None), out_axes=(0, None))(
            fake_batch, discriminator_state)
    
    loss1 = optax.sigmoid_binary_cross_entropy(pred_y, fake_labels).mean()

    pred_y, discriminator_state = jax.vmap(
        discriminator, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
    )(real_batch, discriminator_state)
    loss2 = optax.sigmoid_binary_cross_entropy(pred_y, real_labels).mean()

    return (loss1 + loss2) / 2, (discriminator_state, generator_state)


def loss_g(generator, discriminator, generator_state, discriminator_state, 
           batch_size, key):
    noise = jax.random.normal(key, (batch_size, LATENT_SIZE, 1, 1))
    real_labels = jnp.ones(batch_size)

    fake_batch, generator_state = jax.vmap(
        generator, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
    )(noise, generator_state)

    pred_y, discriminator_state = jax.vmap(
        discriminator, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
    )(fake_batch, discriminator_state)
    loss = optax.sigmoid_binary_cross_entropy(pred_y, real_labels).mean()

    return loss, (generator_state, discriminator_state)


@eqx.filter_jit
def step_discriminator(discriminator, generator, discriminator_state, 
                       generator_state, discriminator_optimizer, 
                       discriminator_opt_state, real_batch, key):
    
    (loss, (discriminator_state, generator_state)), grads = \
    eqx.filter_value_and_grad(loss_d, has_aux=True)(
        discriminator, generator, discriminator_state, generator_state, 
        real_batch, key)

    updates, opt_state = discriminator_optimizer.update(
        grads, discriminator_opt_state, discriminator)
    discriminator = eqx.apply_updates(discriminator, updates)

    return loss, discriminator, discriminator_state, generator_state, opt_state


@eqx.filter_jit
def step_generator(generator, discriminator, generator_state, 
                   discriminator_state, generator_optimizer, 
                   generator_opt_state, batch_size, key):
    
    (loss, (generator_state, discriminator_state)), grads  = \
        eqx.filter_value_and_grad(loss_g, has_aux=True)(
            generator, discriminator, generator_state, discriminator_state, 
            batch_size, key)

    updates, opt_state = generator_optimizer.update(grads, generator_opt_state)
    generator = eqx.apply_updates(generator, updates)
    
    return loss, generator, generator_state, discriminator_state, opt_state


if __name__ == '__main__':
    
    ############ HYPERPARAMS #############

    # Model hyperparameters
    image_size = (1, 28, 28)
    channels, height, width  = image_size
    # Optimisation hyperparameters
    lr = 0.0001
    batch_size = 128
    num_steps = 500_000
    # Sampling hyperparameters
    print_every=100
    sample_size=10

    ############### DATA ################

    key = jax.random.PRNGKey(17456)
    model_key, train_key, loader_key, sample_key = jax.random.split(key, 4)
    data = load_mnist(dtype=jnp.float32)
    data_mean = jnp.mean(data)
    data_std = jnp.std(data)
    data_max = jnp.max(data)
    data_min = jnp.min(data)
    data_shape = data.shape[1:]
    data = (data - data_mean) / data_std

    ############## MODEL #################

    key, gen_key, disc_key = jax.random.split(key, 3)
    generator = Generator(input_shape=LATENT_SIZE, output_shape=image_size, 
                          key=gen_key)
    discriminator = Discriminator(input_shape=image_size, key=disc_key)
    generator_state = eqx.nn.State(generator)
    discriminator_state = eqx.nn.State(discriminator)

    ############### OPTIM ################

    generator_optimizer = optax.adam(lr)
    discriminator_optimizer = optax.adam(lr)

    generator_opt_state = generator_optimizer.init(
        eqx.filter(generator, eqx.is_array))
    discriminator_opt_state = discriminator_optimizer.init(
        eqx.filter(discriminator, eqx.is_array))

    ############### TRAINING ###############

    generator_losses = []
    discriminator_losses = []
    key, train_key, data_key = jax.random.split(key, 3)
    for step, data in zip(range(num_steps), 
                          dataloader(data, batch_size, key=data_key)):

        images = data 
        key, gen_key, disc_key = jax.random.split(key, 3)
        (discriminator_loss, discriminator, discriminator_state, 
        generator_state, discriminator_opt_state) = step_discriminator(
            discriminator, generator, discriminator_state, generator_state, 
            discriminator_optimizer, discriminator_opt_state, images, gen_key)

        (generator_loss, generator, generator_state, discriminator_state,
         generator_opt_state) = step_generator(
             generator, discriminator, generator_state, discriminator_state, 
             generator_optimizer, generator_opt_state, batch_size, disc_key)

        generator_losses.append(generator_loss)
        discriminator_losses.append(discriminator_loss)

        if (step % print_every) == 0 or step == num_steps - 1:
            print(
                f"Step={step}/{num_steps}, Generator loss: {generator_loss}, "
                f"Discriminator loss: {discriminator_loss}")
    
    fig, ax = plt.subplots()
    ax.plot(generator_losses, label = 'generator losses')
    ax.plot(discriminator_losses, label = 'discriminator losses')
    ax.set(title = 'CNN-GAN losses vs iter', xlabel = 'iter', ylabel = 'loss')
    ax.legend()
    fig.savefig('./assets/cnngan_losses.png', dpi=300)

    ############ LOAD PRETRAINED ############

    inference_generator = eqx.nn.inference_mode(generator)
    inference_generator = eqx.Partial(inference_generator, 
                                      state=generator_state)

    # inference_generator = eqx.tree_deserialise_leaves('./models/cnngan.eqx', 
    #                                                   inference_generator)

    ############### INFERENCE ###############

    @eqx.filter_jit
    def evaluate(model, xs):
        out, _ = jax.vmap(model)(xs)
        return out
            
    z = jax.random.normal(sample_key, shape=(sample_size**2, LATENT_SIZE,1,1))
    sample = evaluate(inference_generator, z)
                      
    sample = data_mean + data_std * sample
    sample = jnp.clip(sample, data_min, data_max)
    sample = einops.rearrange(sample, "(n1 n2) 1 h w -> (n1 h) (n2 w)", 
                              n1=sample_size, n2=sample_size)
    

    fig, ax = plt.subplots()
    ax.imshow(sample, cmap="Greys")
    ax.set_title('CNN-GAN')
    ax.axis("off")
    fig.tight_layout()
    fig.savefig('./assets/cnngan_sample.png', dpi=300)
    plt.close()

    ################ SAVING ##################

    eqx.tree_serialise_leaves('./models/cnngan.eqx', inference_generator)