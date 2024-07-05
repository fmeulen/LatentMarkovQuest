setwd("~/.julia/dev/LatentMarkovQuest/examples/olympic")

library(tidyverse)
library(readr)
library(fastDummies)



olymp <- read_csv("olympic_athletes.csv", 
                  col_types = cols(ID = col_character())) %>% 
        mutate(Competition=as.numeric(Competition))

olymp_dummy <- fastDummies::dummy_cols(olymp) 

colnames(olymp_dummy)

write.csv(olymp_dummy, "olympic_athletes_dummyfied.csv", row.names=FALSE)


olymp_standardised <- olymp %>% mutate(Sport=scale(Sport), Strength=scale(Strength), Competition=scale(Competition))
write.csv(olymp_standardised, "olympic_athletes_standardised.csv", row.names=FALSE)

olymp %>% summarise(m=mean(Sport, na.rm=TRUE), std=sd(Sport, na.rm=TRUE))
olymp %>% summarise(m=mean(Strength, na.rm=TRUE), std=sd(Strength, na.rm=TRUE))
olymp %>% summarise(m=mean(Competition, na.rm=TRUE), std=sd(Competition, na.rm=TRUE))

