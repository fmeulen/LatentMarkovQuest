library(ggplot2)
library(readxl)
library(tidyverse)
library(dplyr)

mytheme = theme_bw()
theme_set(mytheme)


cbPallete <- c("Competition"= "#E69F00", "Sport"="#56B4E9", "Strength"="#009E73", "Intercept"= "#0072B2")



# Plot betas

betas <- read.csv("Betas.csv")

betas <- betas %>% mutate(parameter = fct_recode(parameter,
                                "Intercept"    =   "b1",
                                "Sport"    =   "b2",
                                "Strength"    =   "b3",
                                "Competition"    =   "b4")) 

b <- ggplot(betas,aes(x=value,y=parameter, colour = parameter)) + 
  geom_errorbar(aes(xmin=value-se,xmax=value+se), width=.1) +
  geom_point() +
  geom_vline(xintercept = 0) +
  scale_colour_manual(values = cbPallete) +
  scale_x_continuous(labels = function(x) format(x, nsmall = 2)) +
  scale_y_discrete(labels = c("Intercept" = "\u03b2 1","Sport" = "\u03b2 2","Strength" = "\u03b2 3","Competition" = "\u03b2 4")) +
  ylab("parameter") +
  theme(axis.title.x=element_blank(),axis.title.y = element_text(angle = 90),legend.position = "bottom") +
  facet_wrap(~transition, scales="free") 
b


ggsave(filename = "Betas_LMest.png",width = 7, height = 3, dpi = 300)

# Plot gammas

gammas <- read.csv("Gammas.csv")
gammas <- gammas %>% mutate(parameter = fct_recode(parameter,
                                  "Intercept"    =   "g1",
                                  "Sport"    =   "g2",
                                  "Strength"    =   "g3",
                                  "Competition"    =   "g4")) 

transition_names <- c("12" = "1 -> 2",
                      "21" = "2 -> 1",
                      "13" = "1 -> 3",
                      "23" = "2 -> 3",
                      "31" = "3 -> 1",
                      "32" = "3 -> 2")

g <- ggplot(gammas,aes(x=value,y=parameter, colour = parameter)) + 
  geom_errorbar(aes(xmin=value-se,xmax=value+se), width=.1) +
  geom_point() +
  geom_vline(xintercept = 0) +
  scale_colour_manual(values = cbPallete) +
  scale_x_continuous(labels = function(x) format(x, nsmall = 2)) +
  #xlim(-3.5,1) +
  scale_y_discrete(labels = c("Intercept" = "\u03b3 1","Sport" = "\u03b3 2","Strength" = "\u03b3 3","Competition" = "\u03b3 4")) +
  ylab("parameter") +
  theme(axis.title.x=element_blank(),axis.title.y = element_text(angle = 90),legend.position = "bottom") +
  facet_wrap(~transition, labeller = as_labeller(transition_names), nrow = 3, ncol = 2)

ggsave(filename = "Gammas_LMest.png",width = 7, height = 7, dpi = 300)


# also without gamma_1 (intercept estimate)
g2 <- gammas %>% dplyr::filter(parameter %in% c("Sport", "Strength", "Competition")) %>% 
  ggplot(aes(x=value,y=parameter, colour = parameter)) + 
  geom_errorbar(aes(xmin=value-se,xmax=value+se), width=.1) +
  geom_point() +
  geom_vline(xintercept = 0) +
  scale_colour_manual(values = cbPallete) +
  scale_x_continuous(labels = function(x) format(x, nsmall = 2)) +
  scale_y_discrete(labels = c("Sport" = "\u03b3 2","Strength" = "\u03b3 3","Competition" = "\u03b3 4")) +
  ylab("parameter") +
  theme(axis.title.x=element_blank(),axis.title.y = element_text(angle = 90),legend.position = "bottom") +
  facet_wrap(~transition, labeller = as_labeller(transition_names), nrow = 3, ncol = 2, scales="free")
g2

ggsave(filename = "Gammas_LMest_withoutintercept.png",width = 7, height = 7, dpi = 300)






# Plot path

path_pred <- read.csv("PathPrediction_data.csv")

# Forward paths for three participants
forward_paths <- ggplot(path_pred, aes(fill=as.factor(state), y=prob, x=as.factor(week))) + 
  geom_bar(position="stack", stat="identity", width = 0.5) + 
  ylab("probability (%)") + 
  xlab("week") +
  scale_fill_discrete(labels=c('State 1', 'State 2', 'State 3')) + 
#  scale_fill_manual(values = cbPallete) +
  #theme(axis.title.x = element_blank(), axis.text.x = element_blank()) +
  theme(legend.title=element_blank()) +
  facet_wrap(~id,nrow = 3, ncol = 1)
forward_paths

ggsave(filename = "forward_paths.png",width = 7, height = 7, dpi = 300)

covariates <- read.csv("Covariates.csv")

# Training load (same for everyone) for the next 10 weeks
training_load <- ggplot(covariates, aes(fill=training, y=training_load, x=week)) + 
  geom_bar(position="dodge", stat="identity", width = 0.5) + 
  scale_fill_manual(values = cbPallete) +
  xlab("week") + 
  ylab("training load (hours)") + 
  xlim(1,10) +
  scale_x_discrete(limits = c("1","2","3","4","5","6","7","8","9","10")) +
  theme(legend.title=element_blank())
training_load

ggsave(filename = "training_load.png",width = 7, height = 3, dpi = 300)
