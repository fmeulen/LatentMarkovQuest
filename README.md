# LatentMarkovQuest

Julia based implementation of a latent Markov model for analysing questionnaire data. 
**WIP**
The present implementation is merely meant to play around with Turing.jl for obtaining an efficient algorithm for a model similar to that proposed in 

Bartolucci, Lupparelli and Montanari  *Latent Markov model for longitudinal binary data: An application to the performance evaluation of nursing homes*
The Annals of Applied Statistics, 611--636, 2009.

In this implementation, the likelihood is computed from the backward information filter and latent states are marginalised out.
Frequentist estimates (based on the EM-algorithm) can be obtained from the LMest package in R. 
