# Benchmarking of Generative Vision Models

Exploring generative modeling, tuning and benchmarking pipeline. Within purview of this (personal) exploration:

1. Build models with different structures for MNIST type digit image generation. 
2. Build models of different sizes.
3. Build a custom benchmark - FID benchmark for MNIST type image generation.
4. Build a basic model training health visualization platform.
5. Do all of this in Jax.

## Diffusion model:

Influenced by this [implementation](https://docs.kidger.site/equinox/examples/score_based_diffusion/) of score based diffusion, with modification for personal hardware, score function, hyperparameters, and minor structure modifiction.

![alt text](assets/diffusion_sample.png)

## CNN-GAN model: 

Inspired by this [implementation](https://docs.kidger.site/equinox/examples/deep_convolutional_gan/) but modified structured for 28 x 28 MNIST instead of 64 x 64, modified training process among other.

![alt text](assets/cnngan_sample.png)

## Mini-GAN model: 

Small MLP-GAN for basic control. (Also to analyze the training pathologies of GANs in a smaller domain). Based on my own torch [implementation](https://github.com/AbhinavRao23/conditionalGANs). 

![alt text](assets/minigan_sample.png)

