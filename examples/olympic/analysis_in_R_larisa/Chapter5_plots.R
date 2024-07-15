library(ggplot2)
library(readxl)
library(tidyverse)
library(dplyr)

# Plot betas

betas <- read.csv("Betas.csv")

betas$parameter <- factor(betas$parameter,
                             levels = c("b1", "b2", "b3", "b4"),
                             ordered = TRUE)

b <- ggplot(betas,aes(x=value,y=parameter, colour = parameter)) + 
  geom_errorbar(aes(xmin=value-se,xmax=value+se), width=.1) +
  geom_line() +
  geom_point() +
  geom_vline(xintercept = 0) +
  scale_x_continuous(labels = function(x) format(x, nsmall = 2)) +
  scale_y_discrete(labels = c("b1" = "\u03b2 1","b2" = "\u03b2 2","b3" = "\u03b2 3","b4" = "\u03b2 4")) +
  ylab("Parameter") +
  theme(axis.title.x=element_blank(),axis.title.y = element_text(angle = 90),legend.position = "bottom") +
  facet_wrap(~transition)
b + scale_colour_discrete(name="Covariate", limits = c("b1","b2","b3","b4"), labels=c("Intercept","Sport","Strength","Competition"))

#ggsave(filename = "Betas_LMest.png",width = 8, height = 9, dpi = 300)

# Plot gammas

gammas <- read.csv("Gammas.csv")

gammas$parameter <- factor(gammas$parameter,
                          levels = c("g1", "g2", "g3", "g4"),
                          ordered = TRUE)

transition_names <- c("12" = "1 -> 2",
                      "21" = "2 -> 1",
                      "13" = "1 -> 3",
                      "23" = "2 -> 3",
                      "31" = "3 -> 1",
                      "32" = "3 -> 2")

g <- ggplot(gammas,aes(x=value,y=parameter, colour = parameter)) + 
  geom_errorbar(aes(xmin=value-se,xmax=value+se), width=.1) +
  geom_line() +
  geom_point() +
  geom_vline(xintercept = 0) +
  scale_x_continuous(labels = function(x) format(x, nsmall = 2)) +
  xlim(-3.5,1) +
  scale_y_discrete(labels = c("g1" = "\u03b3 1","g2" = "\u03b3 2","g3" = "\u03b3 3","g4" = "\u03b3 4")) +
  ylab("Parameter") +
  theme(axis.title.x=element_blank(),axis.title.y = element_text(angle = 90),legend.position = "bottom") +
  facet_wrap(~transition, labeller = as_labeller(transition_names), nrow = 3, ncol = 2)
g + scale_colour_discrete(name="Covariate", limits = c("g1","g2","g3","g4"), labels=c("Intercept","Sport","Strength","Competition"))

#ggsave(filename = "Gammas_LMest.png",width = 8, height = 9, dpi = 300)

# Plot path

path_pred <- read.csv("PathPrediction_data.csv")

# Transition probabilities for three participants
transition_probabilities <- ggplot(path_pred, aes(fill=as.factor(state), y=prob, x=as.factor(week))) + 
  geom_bar(position="stack", stat="identity", width = 0.5) + 
  ylab("Probability (%)") + 
  xlab("Week") +
  scale_fill_discrete(labels=c('State 1', 'State 2', 'State 3')) + 
  #theme(axis.title.x = element_blank(), axis.text.x = element_blank()) +
  theme(legend.title=element_blank()) +
  facet_wrap(~id,nrow = 3, ncol = 1)
transition_probabilities

covariates <- read.csv("Covariates.csv")

# Training load (same for everyone) for the next 10 weeks
training_load <- ggplot(covariates, aes(fill=training, y=training_load, x=week)) + 
  geom_bar(position="dodge", stat="identity", width = 0.5) + 
  xlab("Week") + 
  ylab("Training load (hours)") + 
  xlim(1,10) +
  scale_x_discrete(limits = c("1","2","3","4","5","6","7","8","9","10")) +
  theme(legend.title=element_blank())
training_load
