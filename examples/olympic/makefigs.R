library(tidyverse)
library(forecast)
library(ggpubr)
setwd("~/.julia/dev/LatentMarkovQuest/examples/olympic")

theme_set(theme_bw())

library(readr)
iterates_full <- read_csv("figs/iterates.csv", col_types = cols(chain = col_character()))
#View(iterates)
iterates <- iterates_full %>% select(1:70)
iterates_long <- pivot_longer(iterates, 3:70, names_to="parameter", values_to="value")


iterates_long %>% dplyr::filter(parameter %in% c("σα²", "σ²","αup[15]","γup[1]","Z1[1]","Z1[2]","Z1[3]","αdown[23]","γdown[1]",  "γdown[2]",  "γdown[3]"  )) %>% 
  ggplot(aes(x=iteration, y=value, colour=chain)) + facet_wrap(~parameter,scales="free") + geom_line() + labs(y="")

iterates_long %>% dplyr::filter(parameter %in% c("αup[1]","αdown[1]","αup[2]","αdown[2]","αup[3]","αdown[3]","αup[23]","αdown[23]"  )) %>% 
  ggplot(aes(x=iteration, y=value, colour=chain)) + facet_wrap(~parameter,scales="free",nrow=2) + geom_line() + labs(y="")

iterates_long %>% dplyr::filter(parameter %in% c("Z1[1]","Z1[2]","Z1[3]","Z2[1]","Z2[2]","Z2[3]"  )) %>% 
  ggplot(aes(x=iteration, y=value, colour=chain)) + facet_wrap(~parameter,scales="free",nrow=2) + geom_line() + labs(y="")


df_acf <- iterates %>% group_by(chain) %>% reframe(acf=acf(`γup[1]`,plot=F)$acf, lag=acf(`γup[1]`,plot=F)$lag) 
df_acf %>% ggplot(aes(x=lag, y=acf)) +
  geom_bar(stat = "identity", position = "identity") + facet_wrap(~chain)


posterior_summary <- read_csv("figs/posterior_summary.csv")

posterior_summary %>% dplyr::filter(rhat >1.6) %>%  ggplot(aes(x=rhat,y=parameters)) + geom_point()

posterior_summary <- posterior_summary %>% mutate(lower=mean-2*std, upper=mean+2*std)
posterior_summary %>% ggplot(aes(x=parameters, y=mean)) + geom_point() +
  geom_errorbar(aes(ymin=lower, ymax=upper),colour='red') + coord_flip()

posterior_summary %>% dplyr::filter(str_detect(parameters, "α")) %>% 
  ggplot(aes(x=parameters, y=mean)) + geom_point() +
  geom_errorbar(aes(ymin=lower, ymax=upper),colour='red') + coord_flip()
