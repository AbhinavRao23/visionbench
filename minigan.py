from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax  
import einops

from utils import load_mnist, dataloader


class Discriminator(eqx.Module):
    model: eqx.nn.Sequential
    data_size: int
    l_relu: float
    dropout_rate: float
    
    def __init__(self, data_size, l_relu, dropout_rate, key):
        self.data_size = data_size
        self.l_relu = l_relu
        self.dropout_rate = dropout_rate
        keys = jax.random.split(key, 5)
        self.model = eqx.nn.Sequential([
            eqx.nn.Linear(data_size, 1024, key=keys[1]), 
            eqx.nn.Lambda(partial(jax.nn.leaky_relu,negative_slope=self.l_relu)),
            eqx.nn.Linear(1024, 512, key=keys[2]), 
            eqx.nn.Lambda(partial(jax.nn.leaky_relu,negative_slope=self.l_relu)),
            eqx.nn.Linear(512, 256, key=keys[3]), 
            eqx.nn.Lambda(partial(jax.nn.leaky_relu,negative_slope=self.l_relu)),
            eqx.nn.Linear(256, 1, key=keys[4]),
            eqx.nn.Lambda(jax.nn.sigmoid)
            ]
        )

    def __call__(self, x):
        return self.model(x)
    
class Generator(eqx.Module):
    model: eqx.nn.Sequential
    latent_size: int
    data_size: int
    l_relu: float

    def __init__(self, data_size, latent_size, l_relu, key):
        self.data_size = data_size
        self.latent_size = latent_size
        self.l_relu = l_relu
        keys = jax.random.split(key, 5)
        self.model = eqx.nn.Sequential([
            eqx.nn.Linear(latent_size, 256, key=keys[1]),
            eqx.nn.Lambda(partial(jax.nn.leaky_relu,negative_slope=self.l_relu)),
            eqx.nn.Linear(256 , 512, key=keys[2]),
            eqx.nn.Lambda(partial(jax.nn.leaky_relu,negative_slope=self.l_relu)),
            eqx.nn.Linear(512 , 1024, key=keys[3]),
            eqx.nn.Lambda(partial(jax.nn.leaky_relu,negative_slope=self.l_relu)),
            eqx.nn.Linear(1024 , data_size, key=keys[4]),
            eqx.nn.Lambda(jax.nn.tanh)
            ]
        )
    
    def __call__(self, z):
        return self.model(z)
    
def loss_d(discriminator, generator, real_batch, latent_size, key):
    batch_size = real_batch.shape[0]
    fake_labels = jnp.zeros(batch_size)
    real_labels = jnp.ones(batch_size)
    
    z = jax.random.normal(key, (batch_size, latent_size))
    fake_batch = jax.vmap(generator)(z)
    pred_y = jax.vmap(discriminator)(fake_batch).flatten()
    loss1 = optax.sigmoid_binary_cross_entropy(pred_y, fake_labels).mean()

    pred_y = jax.vmap(discriminator)(real_batch).flatten()
    loss2 = optax.sigmoid_binary_cross_entropy(pred_y, real_labels).mean()

    return (loss1 + loss2) / 2


def loss_g(generator, discriminator, batch_size, latent_size, key):
    z = jax.random.normal(key, (batch_size, latent_size))
    real_labels = jnp.ones(batch_size)
    fake_batch = jax.vmap(generator)(z)
    pred_y = jax.vmap(discriminator)(fake_batch).flatten()
    loss = optax.sigmoid_binary_cross_entropy(pred_y, real_labels).mean()

    return loss


@eqx.filter_jit
def step_discriminator(discriminator, generator, d_optimizer,  d_opt_state, 
                       real_data, latent_size, key):
    
    loss, grads = eqx.filter_value_and_grad(loss_d)(
        discriminator, generator, real_data, latent_size, key)

    updates, d_opt_state = d_optimizer.update(grads, d_opt_state, discriminator)
    discriminator = eqx.apply_updates(discriminator, updates)

    return loss, discriminator, d_opt_state


@eqx.filter_jit
def step_generator(generator, discriminator, g_optimizer, g_opt_state, 
                   batch_size, latent_size, key):
    
    loss, grads  = eqx.filter_value_and_grad(loss_g)(
        generator, discriminator, batch_size, latent_size, key)

    updates, g_opt_state = g_optimizer.update(grads, g_opt_state)
    generator = eqx.apply_updates(generator, updates)
    
    return loss, generator, g_opt_state


if __name__ == '__main__':

    ############ HYPERPARAMS #############

    # Model hyperparameters
    data_size = 784
    latent_size = 100
    l_relu = 0.2
    dropout_rate = 0.5
    # Optimisation hyperparameters
    lr = 0.001
    batch_size = 128
    num_steps = 100000
    # Sampling hyperparameters
    print_every=1000
    sample_size=10


    ############### DATA ################

    key = jax.random.PRNGKey(1736)
    data = load_mnist(dtype=jnp.float32)
    data_mean = jnp.mean(data)
    data_std = jnp.std(data)
    data_max = jnp.max(data)
    data_min = jnp.min(data)
    data_shape = data.shape[1:]
    data = (data - data_mean) / data_std
    data = einops.rearrange(data, "b 1 h w -> b (h w)")


    ############## MODEL #################

    key, d_key, g_key = jax.random.split(key, 3)
    discriminator = Discriminator( data_size, l_relu, dropout_rate, d_key)
    generator = Generator(data_size, latent_size, l_relu, g_key)


    ############### OPTIM ################

    g_optimizer = optax.adam(lr)
    d_optimizer = optax.adam(lr)

    g_opt_state = g_optimizer.init(eqx.filter(generator, eqx.is_array))
    d_opt_state = d_optimizer.init(eqx.filter(discriminator, eqx.is_array))


    ############### TRAINING ###############

    g_losses, d_losses = [], []
    key, train_key, data_key = jax.random.split(key, 3)
    for step, data in zip(range(num_steps), 
                          dataloader(data, batch_size, key=data_key)):
        
        train_key, g_key, d_key = jax.random.split(train_key, 3)
        
        (d_loss, discriminator, d_opt_state) = step_discriminator(
            discriminator, generator, d_optimizer,  d_opt_state, data, 
            latent_size, d_key)

        (g_loss, generator, g_opt_state) = step_generator(
            generator, discriminator, g_optimizer, g_opt_state, batch_size, 
            latent_size, g_key)

        g_losses.append(g_loss)
        d_losses.append(d_loss)

        if (step % print_every) == 0 or step == num_steps - 1:
            print(
                f"Step={step}/{num_steps}, Generator loss: {g_loss}, "
                f"Discriminator loss: {d_loss}")
    
    fig, ax = plt.subplots()
    ax.plot(g_losses, label = 'generator losses')
    ax.plot(d_losses, label = 'discriminator losses')
    ax.set(title = 'GAN losses vs iter', xlabel = 'iter', ylabel = 'loss')
    ax.legend()
    fig.savefig('./assets/minigan_losses.png', dpi=300)
    
    ########## LOADING PRETRAINED ###########

    # generator = eqx.tree_deserialise_leaves('./models/minigan.eqx', generator)


    ############### INFERENCE ###############

    @eqx.filter_jit
    def evaluate(model, xs):
        out = jax.vmap(model)(xs)
        return out
    
    key, sample_key = jax.random.split(key)
    z = jax.random.normal(sample_key, shape=(sample_size**2, latent_size))
    sample = evaluate(generator, z)
    sample = data_mean + data_std * sample
    sample = jnp.clip(sample, data_min, data_max).reshape(-1, 1, 28, 28)
    sample = einops.rearrange(sample, "(n1 n2) 1 h w -> (n1 h) (n2 w)", 
                              n1=sample_size, n2=sample_size)
    
    fig, ax = plt.subplots()
    ax.imshow(sample, cmap="Greys")
    ax.set_title('Mini-GAN')
    ax.axis("off")
    fig.tight_layout()
    fig.savefig('./assets/minigan_sample.png', dpi=300)
    plt.close()
    
    ################ SAVING ##################

    eqx.tree_serialise_leaves('./models/minigan.eqx', generator)

