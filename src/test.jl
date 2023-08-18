using StatsBase, Plots, LinearAlgebra
using Optim
using ForwardDiff
using Distributions
using ComponentArrays
using StatsFuns  # for softmax
using Random
using DynamicHMC
using Turing
using StatsPlots # required for Turing plots

v(k, α) = k^(-0.5-α)
γ(k, θ) = exp(-π^2 * k^2 * θ)

@model function logtarget(y,z, n, α)
    m = length(y)
    θ ~ Uniform(0,1)
    f = Vector{Float64}(undef, m)
    for k ∈ 1:m
        f[k] ~ Normal(0, v(k, α))
        y[k] ~ Normal(f[k], 1.0/sqrt(n))
        z[k] ~ Normal(γ(k, θ) * f[k], 1.0/sqrt(n))
    end
end


# @model function logtarget2(y,z, n, α)
#     m = length(y)
#     θ ~ Uniform(0,1)
#     f ~ arraydist([Normal(0, v(k, α)) for k in 1:m])
#     y ~ arraydist([Normal(f[k], 1.0/sqrt(n)) for k in 1:m])
#     z ~ arraydist([Normal(γ(k, θ) * f[k], 1.0/sqrt(n)) for k in 1:m])
# end





# generate data
Random.seed!(5)
θ₀ = 0.01
n = 10000
m = 50
f = [1.0/k^2 for k in 1:m]
y = [rand(Normal(f[k], 1.0/sqrt(n))) for k in 1:m]
z = [rand(Normal(γ(k, θ₀)* f[k], 1.0/sqrt(n))) for k in 1:m]

α = 5.0

model = logtarget(y, z,  n, α)
#model2 = logtarget2(y, z,  n, α)

# sampler =  NUTS() 
# @time chain = sample(model, sampler, MCMCDistributed(), 1000, 3; progress=true);
# chain = sample(model, HMC(0.007,10), 1000)

sampler = Gibbs(ESS(:f), MH(:θ))
@time chain = sample(model, sampler, 100_000)
#@time chain2 = sample(model2, sampler, 25_000)

BI = 1000
chain 
plot(chain[:θ][BI:end])
histogram(chain[:θ][BI:end],bins=25)

@show  mean((chain[:θ][BI:end]))
