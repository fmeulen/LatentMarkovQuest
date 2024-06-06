setwd("~/.julia/dev/LatentMarkovQuest/examples/olympic")

library(tidyverse)
library(readr)
library(fastDummies)



olymp <- read_csv("olympic_athletes.csv", 
                  col_types = cols(ID = col_character())) %>% 
        mutate(Competition=as.numeric(Competition))

olymp_dummy <- fastDummies::dummy_cols(olymp) 

colnames(olymp_dummy)

write.csv(olymp_dummy, "olympic_athletes_dummyfied.csv")
