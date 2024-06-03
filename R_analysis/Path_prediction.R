library(ggplot2)
library(tidyverse)
library(readxl)
library(LMest)
library(HMM)
library(ggpubr)
library(grid)
library(glue)
library(gridExtra)

# Load covariates
cov <- read_csv("Covariates.csv")

# The posterior distribution of the latent states at the end of the follow up (t=79)
posterior_dist <- read_csv("Posterior.csv")

# Design matrix Gamma (right side of the equation 5.6 in the manuscript)
Design_matrix_Gamma <- function(x,u_present){
  estimates <- as.matrix(m8_1$Ga[,,u_present])
  G <- t(x) %*% estimates
  GG <- exp(G)
  return(GG)
}

B <- 1000
t <- 10
k <- 3
x<-t(cov)

u <- array(0,dim=c(B,t,nrow(posterior_dist)))

for (id in 1:nrow(posterior_dist)){
  p <- as.vector(posterior_dist[id,2:4],mode="numeric")
  #u = matrix(0, nrow = B, ncol = t)
  #u <- array(0,dim=c(B,t,nrow(posterior_dist)))
  
  for (i in 1:t){
    trans_matrix <- matrix(0,ncol = k,nrow = k)
    for (z in 1:k) {
      GG <- Design_matrix_Gamma(x[i,],z)
      if(z==1){
        a <- matrix(c(1,1,1,-GG[1],1,0,-GG[2],0,1),nrow = k,byrow = TRUE)
        b <- c(1,0,0)
        solution <- solve(a,b)
        trans_matrix[z,] <- solution
      }else if(z==k){
        a <- matrix(c(1,1,1,1,0,-GG[1],0,1,-GG[2]),nrow = k,byrow = TRUE)
        b <- c(1,0,0)
        solution <- solve(a,b)
        trans_matrix[z,] <- solution
      }else{
        a <- matrix(c(1,1,1,1,-GG[1],0,0,-GG[2],1),nrow = k,byrow = TRUE)
        b <- c(1,0,0)
        solution <- solve(a,b)
        trans_matrix[z,] <- solution
      }
      
    }
    
    N_sim1 = round(p[1]*B,0)
    if (N_sim1 > 0){
      for (j in 1:N_sim1) {
        #u[j,i] <- sample(1:k, 1, prob = trans_matrix[1,],replace = FALSE)
        u[j,i,id] <- sample(1:k, 1, prob = trans_matrix[1,],replace = FALSE)
      }
    }
    
    N_sim2 = round(p[2]*B,0)
    if (N_sim2 > 0){
      for (j in (N_sim1 + 1):(N_sim1 + N_sim2)) {
        #u[j,i] <- sample(1:k, 1, prob = trans_matrix[2,], replace = FALSE)
        u[j,i,id] <- sample(1:k, 1, prob = trans_matrix[2,], replace = FALSE)
      }
    }
    
    N_sim3 = round(p[3]*B,0)
    if (N_sim3 > 0){
      for (j in (N_sim1 + N_sim2 + 1):B) {
        #u[j,i] <- sample(1:k, 1, prob = trans_matrix[3,], replace = FALSE)
        u[j,i,id] <- sample(1:k, 1, prob = trans_matrix[3,], replace = FALSE)
      }
    }
  }
}

