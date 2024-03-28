packdir = "/Users/frankvandermeulen/.julia/dev/LatentMarkovQuest"
cd(packdir)
wdir = joinpath(packdir, "src")#@__DIR__
cd(wdir)
#outdir= joinpath(wdir, "out")

using StatsBase, Plots, LinearAlgebra
using Optim
using ForwardDiff
using Distributions
using ComponentArrays
using StatsFuns  # for softmax
using Random
using DynamicHMC
using UnPack
using Turing
using StatsPlots # required for Turing plots
using BenchmarkTools
using StaticArrays
using NNlib # for softmax
using DataFrames
using Parameters

import StatsBase.sample


# fix these settings for now
@with_kw struct Pars
    NUM_HIDDENSTATES::Int = 3  # 3 hidden states (don't change)
    DIM_COVARIATES::Int = 3    # 2 predictors, but include intercept; this one can be changed (nothing hardcoded)
    DIM_RESPONSE::Int = 4      # 4 questions (don't change)
end

p = Pars()

include("lmq_funcs.jl")
include("converting_turing_output.jl")