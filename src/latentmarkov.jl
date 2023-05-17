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

# const NUM_HIDDENSTATES = 3
# const DIM_COVARIATES = 2
# const DIM_RESPONSE = 4

@with_kw struct Pars
    NUM_HIDDENSTATES::Int = 3
    DIM_COVARIATES::Int = 3 # include intercept
    DIM_RESPONSE::Int = 4
end

p = Pars()

include("lmq_funcs.jl")