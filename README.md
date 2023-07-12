# Discrete Generative Adversarial Network
This repo contains all the code necessary to train Generative Adversarial Networks (GANs) with discrete outputs.

Usually this is not possible as discrete outputs are not differentiable and thus the error is not "back-propagatable".

In order to overcome this problem 2 solutions are proposed in this repo.

## Reinforcement Learning
The first solution is to use reinforcement learning to estimate the generator gradient, considering the reward the negative loss, and sampling over the generator output.

![RL formula](https://latex.codecogs.com/gif.latex?%5Cdpi%7B300%7D%20%5Chuge%20%5Cnabla_%5Ctheta%5Cmathcal%7BL%7D_G%20%5Capprox%5Cln%20D_%5Cphi%28I%29%5C%2C%5C%2C%5Cnabla_%5Ctheta%5Cln%20G_%5Ctheta%28I%7C%5Cepsilon%29)

## Straight-through-estimator
The second method used is the straight-through-estimator, used in VQVAE in order to have a discrete latent space (codebook), which consist in adding a residual to the continuous output of the generator in order to make it look discrete, and since additions don't break the gradient, it can be trained as a normal GAN
![Alt text](https://latex.codecogs.com/gif.latex?%5Cdpi%7B300%7D%20%5Chuge%20%5Cbegin%7Balign*%7D%20%5Cnabla_%5Ctheta%20%5Cmathcal%7BL%7D_G%20%26%3D%20%5Cnabla_%7BG_%5Ctheta%28%5Cgamma%29%7D%20%5Cln%20D_%5Cphi%28G_%5Ctheta%28%5Cgamma%29%20&plus;%20%5Cepsilon%29%20%5C%2C%5Ccdot%20%5C%2C%5Cnabla_%5Ctheta%20G_%5Ctheta%28%5Cgamma%29%5C%5C%20%5Cepsilon%20%26%3D%20%28I%20%5Csim%20G%28%5Cgamma%29%29%20-%20G%28%5Cgamma%29%20%5Cend%7Balign*%7D)

## Results
https://github.com/AlbertoSinigaglia/discrete-gans/assets/25763924/d4d5dfd4-a61d-4273-8df6-0186ebca616e

