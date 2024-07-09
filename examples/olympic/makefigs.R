library(tidyverse)
library(forecast)
library(ggpubr)
library(GGally)
setwd("~/.julia/dev/LatentMarkovQuest/examples/olympic")

theme_set(theme_bw())

library(readr)
iterates_full <- read_csv("figs/iterates.csv", col_types = cols(chain = col_character()))
#View(iterates)
iterates <- iterates_full %>% select(1:31)

#ggpairs(iterates, mapping = aes(color = chain), columns = 3:20)



iterates_long <- pivot_longer(iterates, 3:31, names_to="parameter", values_to="value")


iterates_long %>% dplyr::filter(parameter %in% c("σ²","γ12[1]", "γ12[2]", "γ32[3]"  ,"γ32[1]" ,"γ32[2]"   )) %>% 
  ggplot(aes(x=iteration, y=value, colour=chain)) + facet_wrap(~parameter,scales="free",ncol=1) + geom_line() + labs(y="")


iterates_long %>% dplyr::filter(parameter %in% c("Z1[1]","Z1[2]","Z1[3]","Z2[1]","Z2[2]","Z2[3]"  )) %>% 
  ggplot(aes(x=iteration, y=value, colour=chain)) + facet_wrap(~parameter,scales="free",nrow=2) + geom_line() + labs(y="")


df_acf <- iterates %>% group_by(chain) %>% reframe(acf=acf(`γup[1]`,plot=F)$acf, lag=acf(`γup[1]`,plot=F)$lag) 
df_acf %>% ggplot(aes(x=lag, y=acf)) +
  geom_bar(stat = "identity", position = "identity") + facet_wrap(~chain)


posterior_summary <- read_csv("figs/posterior_summary.csv") %>% 
  mutate(type=c("sigma", rep("from 1 to 2",4), rep("from 2 to 3",4), rep("from 2 to 1",4), rep("from 3 to 2",4), 
        rep("Z1",3), rep("Z2",3), rep("Z3",3), rep("Z4",3)    ))



posterior_summary <- posterior_summary %>% mutate(lower=mean-2*std, upper=mean+2*std)
posterior_summary %>% ggplot(aes(x=parameters, y=mean, colour=type)) + geom_point() +
  geom_errorbar(aes(ymin=lower, ymax=upper)) +  coord_flip()


covariates = c("intercept", "sport","strength","competition")
posterior_summary %>% dplyr::filter(str_detect(parameters, "γ")) %>% 
  mutate(covariate=rep(covariates,4)) %>% 
  ggplot(aes(x=parameters, y=mean, colour=covariate)) + geom_point() +
  geom_errorbar(aes(ymin=lower, ymax=upper)) +
  labs(x="parameter", y="") + geom_hline(aes(yintercept=0)) +  facet_wrap(~type,scales="free_y")+ coord_flip() + 
  theme(legend.position = "bottom")