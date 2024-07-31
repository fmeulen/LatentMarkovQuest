library(tidyverse)
library(forecast)
library(ggpubr)
library(GGally)
setwd("~/.julia/dev/LatentMarkovQuest/examples/olympic")

theme_set(theme_bw())

library(readr)


# traceplots

iterates_full <- read_csv("figs/iterates.csv", col_types = cols(chain = col_character()))
iterates <- iterates_full %>% dplyr::select(1:31)
iterates_long <- pivot_longer(iterates, 3:31, names_to="parameter", values_to="value")

# for gammas
iterates_long %>% dplyr::filter(parameter %in% 
      c("σ²","γ12[1]", "γ12[2]","γ12[3]","γ12[4]","γ32[1]", "γ32[2]"    ,"γ32[3]"   ,"γ32[4]"    )) %>% 
  ggplot(aes(x=iteration, y=value, colour=chain)) + facet_wrap(~parameter,scales="free",ncol=2) + geom_line() + labs(y="")
ggsave("figs/traceplots1.png", width=7, height=8)

# for Zs
iterates_long %>% dplyr::filter(parameter %in% c("Z1[1]","Z1[2]","Z1[3]","Z2[1]","Z2[2]","Z2[3]"  )) %>% 
  ggplot(aes(x=iteration, y=value, colour=chain)) + facet_wrap(~parameter,scales="free",ncol=2) + geom_line() + labs(y="")
ggsave("figs/traceplots1.png", width=7, height=6)

# posterior summary plots

posterior_summary <- read_csv("figs/posterior_summary.csv") %>% 
  mutate(type=c("sigma",
                 rep("1 -> 2",4), rep("2 -> 3",4), rep("2 -> 1",4), rep("3 -> 2",4), 
        rep("Z1",3), rep("Z2",3), rep("Z3",3), rep("Z4",3)    ))
posterior_summary <- posterior_summary %>% mutate(lower=mean-std, upper=mean+std)

# all pars
posterior_summary %>% ggplot(aes(y=parameters, x=mean)) + geom_point() +
  geom_errorbar(aes(xmin=lower, xmax=upper), width=.1) +
  labs(y="parameter", x="")  +
  geom_vline(xintercept = 0) 
ggsave("figs/all_estimates.png", width=7, height=10)

# alternative displayx
library(ggdist)
iterates_long %>% ggplot(aes(x=value, y=parameter)) + stat_halfeye(fill="lightblue")+
  geom_vline(xintercept = 0) 

# gammas
cbPallete <- c("Competition"= "#E69F00", "Sport"="#56B4E9", "Strength"="#009E73", "Intercept"= "#0072B2")

covariates = c("Intercept", "Sport","Strength","Competition")
gammas <- posterior_summary %>% dplyr::filter(str_detect(parameters, "γ")) %>% 
  mutate(covariate=as.factor(rep(covariates,4)))   %>%
  mutate(covariate=fct_relevel(covariate, "Intercept", "Sport", "Strength", "Competition", after=0)) %>% 
  mutate(parameter=covariate) # ugly fix, but will do
  
  gammas %>% 
  ggplot(aes(y=parameter, x=mean, colour=parameter)) +
  geom_point() +
  geom_errorbar(aes(xmin=lower, xmax=upper), width=.1) +
  scale_y_discrete(labels = c("Intercept" = "\u03b3 1","Sport" = "\u03b3 2","Strength" = "\u03b3 3","Competition" = "\u03b3 4")) +
  labs(x="parameter", y="") + 
  geom_vline(xintercept = 0) +
  geom_hline(aes(yintercept=0)) +
  scale_colour_manual(values = cbPallete)  +
  facet_wrap(~type, scales="free_y")+
    theme(legend.position = "bottom")
ggsave("figs/Gammas_Bayes_estimates.png", width=7, height=4.67)

# gammas without intercept and free scales

gammas %>% dplyr::filter(parameter %in% c("Sport", "Strength", "Competition")) %>% 
  ggplot(aes(y=parameter, x=mean, colour=parameter)) +
  geom_point() +
  geom_errorbar(aes(xmin=lower, xmax=upper), width=.1) +
  scale_y_discrete(labels = c("Intercept" = "\u03b3 1","Sport" = "\u03b3 2","Strength" = "\u03b3 3","Competition" = "\u03b3 4")) +
  labs(x="parameter", y="") + 
  geom_vline(xintercept = 0) +
  geom_hline(aes(yintercept=0)) +
  scale_colour_manual(values = cbPallete)  +
  facet_wrap(~type,scales="free")+
    theme(legend.position = "bottom")
ggsave("figs/Gammas_Bayes_estimates_withoutintercept.png", width=7, height=4.67)



# forward paths


pathpred = read_csv("figs/simulated_scenarios.csv", col_types = cols(week = col_character(), state = col_character()))
                                                              

# Forward paths for three participants
forward_paths <- ggplot(pathpred, aes(fill=as.factor(state), y=prob, x=as.factor(week))) + 
  geom_bar(position="stack", stat="identity", width = 0.5) + 
  ylab("probability (%)") + 
  xlab("week") +
  scale_fill_discrete(labels=c('State 1', 'State 2', 'State 3')) + 
  theme(axis.title.x=element_blank(),axis.title.y = element_text(angle = 90),legend.position = "bottom") +
  #  scale_fill_manual(values = cbPallete) +
  #theme(axis.title.x = element_blank(), axis.text.x = element_blank()) +
  theme(legend.title=element_blank()) +
  facet_wrap(~athlete,nrow = 3, ncol = 1)
forward_paths

ggsave(filename = "figs/forward_paths_Bayes.png",width = 7, height = 6, dpi = 300)






# plots by chain

# iterates_long %>% group_by(chain, parameter) %>% summarise(m = mean(value), s =sd(value)) %>% 
#   ggplot(aes(y=parameter, x=m, colour=chain)) + geom_jitter(width=0.0, height=0.3) #+ facet_wrap(~parameter)
