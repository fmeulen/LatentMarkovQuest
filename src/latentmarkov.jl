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
    NUM_HIDDENSTATES::Int = 3  # 3 hidden states
    DIM_COVARIATES::Int = 3    # 2 predictors, but include intercept
    DIM_RESPONSE::Int = 4      # 4 questions
end

p = Pars()

include("lmq_funcs.jl")