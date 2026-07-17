# DDPM on MNIST: From ELBO to epsilon prediction

## Table of Content 

1. Forward diffusion
2. Closed-form q(x_t | x_0)
3. Reverse process and posterior q(x_{t-1}|x_t,x_0)
4. ELBO decomposition
5. Why L_simple works
6. Code mapping: formula -> train_mnist_ddpm.py
7. Engineering tricks: EMA / ResBlock / GroupNorm / posterior variance
8. MNIST experiment results and failure analysis
9. What DDPM teaches for LPM

## Formward Diffusion 

- 从 q(x_t|x_{t-1}) 推出 q(x_t|x_0)

## Sampling 

## Tricks 

- ELBO decomposition 
- L_simple 

## Code Mapping / Example 

## MNIST Experiment Results 

- Spikes 20260623 results 
- With EMA no spike 20260628 results and more epoches 

## DDPM for LPM 

- image 和 video generation 的基石


