# to show full output of x, do   "show(stdout, "text/plain", x)"

using CSV
using DataFrames
using JLD2 
using ReverseDiff
using Tidier
#using Zygote

wd = @__DIR__
cd(wd)

# Prior on root node (x can be inital state)
Πroot(_, p) = (@SVector ones(p.NUM_HIDDENSTATES))/p.NUM_HIDDENSTATES    

isanymissing(x) = maximum(ismissing.(x))  # helper function to deal with missing vals

#--------------------------- read water polo data
dfull = CSV.read("olympic_athletes_standardised.csv", DataFrame; delim=",", missingstring="NA",
types= Dict(3=>Float64,4=>Float64,5=>Float64,6=>Int64,7=>Int64,8=>Int64,9=>Int64)) 
# x: (sport,strength,competition), cols 3:5
# y: (participation, modification, performance, symptoms), cols 6:9 (on binary scale)

# remove rows where we have no data, these are rows where columns 3:10 are missing
# originally, missing values were added because LMest can only work with rectangular data
dfull_miss = ismissing.(dfull)
keep = [sum(x) <8 for x in eachrow(dfull_miss)]  # if equal to 1 keep that data
d = dfull[keep .== 1, :]


n = 24
p = Pars(NUM_HIDDENSTATES = 3, DIM_COVARIATES= 3, DIM_RESPONSE = 4) 

# turn the data into a vector of type ObservationTrajectory
TX = Union{Missing, SVector{p.DIM_COVARIATES+1,Float64}} # indien er missing vals zijn 
TY = Union{Missing, SVector{p.DIM_RESPONSE, Int64}}

n = 24  # nr of athletes
𝒪s = ObservationTrajectory{TX,TY}[]
for i ∈ 1:n
    di = d[d.ID .== i, :]
    X = TX[]
    Y = TY[]
    for r in eachrow(di)
        x = SA[1.0, r[3], r[4], r[5]]  # include intercept
        #x = SA[r[3], r[4], r[5]]  # no intercept
        if isanymissing(x)
            push!(X, missing)
        else
            push!(X, x)
        end
        y = SA[r[6]+1, r[7]+1, r[8]+1, r[9]+1]
        if isanymissing(y)
            push!(Y, missing)
        else
            push!(Y, y)
        end
    end
    push!(𝒪s, ObservationTrajectory(X,Y))
end    

map(x-> length(x.X), 𝒪s)

# count missing values in either covariates or responses
countmissing(x,y) = [mean(ismissing.(x)), mean(ismissing.(y))]
miss = map(o -> countmissing(o.X, o.Y), 𝒪s)
@show miss
#scatter(first.(miss), last.(miss))


𝒪s_small = copy(𝒪s)
deleteat!(𝒪s_small, 2:24)

restricted = false
ztype = restricted ? Restricted() : Unrestricted() 
model = logtarget(ztype, 𝒪s, p);

#--------------- map -----------------------
@time map_estimate = maximum_a_posteriori(model)
show(stdout, "text/plain", map_estimate.values)


mle_estimate = maximum_likelihood(model)
coeftable(mle_estimate)


show(stdout, "text/plain", mle_estimate.values)

θmap = convert_turingoutput(ztype, map_estimate, p);
@show θmap[:γ12] 
@show θmap[:γ21] 
@show mapallZtoλ(θmap)'

# ----------- mcmc ---------------------------
sampler = Turing.NUTS(adtype=AutoReverseDiff())

sampler = Turing.NUTS()
@time chain = sample(model, sampler, MCMCThreads(), 2000, 3; progress=true)#, initial_params=map_estimate.values.array);

plot(chain)
savefig(wd*"/figs/olympic_histograms_traces.pdf")

histogram(chain)
savefig(wd*"/figs/histograms_traces.pdf")

CSV.write(wd*"/figs/iterates.csv", DataFrame(chain))

summ = summarize(chain; sections=[:parameters])
show(stdout, "text/plain", summ)

CSV.write(wd*"/figs/posterior_summary.csv", DataFrame(summ))

# extract posterior mean from mcmc output
θs = describe(chain)[1].nt.mean
names = String.(describe(chain)[1].nt.parameters)

@warn "We assume here 4 questions (hence Z1,...,Z4). Adapt if different"

σ²_ = θs[occursin.("σ²", names)]
σα²_ = θs[occursin.("σα²", names)]
αup_ = θs[occursin.("αup", names)]
αdown_ = θs[occursin.("αdown", names)]
γup_ = θs[occursin.("γup", names)]
γdown_ = θs[occursin.("γdown", names)]
if restricted 
    Z1_ = θs[occursin.("Z0", names)]
    Z2_ = θs[occursin.("Z0", names)]
    Z3_ = θs[occursin.("Z0", names)]
    Z4_ = θs[occursin.("Z0", names)]
else
    Z1_ = θs[occursin.("Z1", names)]
    Z2_ = θs[occursin.("Z2", names)]
    Z3_ = θs[occursin.("Z3", names)]
    Z4_ = θs[occursin.("Z4", names)]
end
θpm = ComponentArray(σ²=σ²_, σα²=σα²_, αup=αup_, αdown=αdown_,  γ12=γup_, γ21=γdown_, Z1=Z1_, Z2=Z2_, Z3=Z3_, Z4=Z4_)
   

λs = mapallZtoλ(θpm)'
@show λs

@show θpm[:γ12]

@show θpm[:γ21]

# save objects 
jldsave("ex_olympicathletes.jld2"; 𝒪s, model, θpm, λs, chain, ztype, map_estimate)
jldsave("ex_olympicathletes_large.jld2"; 𝒪s, model, θpm, λs, chain, ztype)

### to open again
aa = jldopen("ex_olympicathletes.jld2")
aa["𝒪s"]
###



#---------------- generating forward scenarios -------------------------

# extract info from HMC
# chain.value.data 500×32×3 Array{Float64, 3}, so this contains the iterates from the 3 chains

chain = aa["chain"] # if read via jldopen
# vals = vcat(chain.value.data[:,:,1], chain.value.data[:,:,2], chain.value.data[:,:,3]) # merge all iterates
vals = DataFrame(chain)
names = vcat(chain.name_map.parameters, chain.name_map.internals)


# set training load and initial latent status
trainingload = CSV.read("Covariates.csv", DataFrame)
X = [SA[x...] for x in eachcol(trainingload)]
U0 = 3 # presently assumed latent state

scenarios = Vector{Int64}[]
for i ∈ 1:size(vals)[1]
    γup = vals[i, 3:6]
    γdown = vals[i, 7:10]
    θ =  ComponentArray(γ12=γup, γ21=γdown, γ23=γup, γ32=γdown)
    latentpath = sample_latent(θ, X, U0, p)
    push!(scenarios, latentpath)
end

# compute proportions at each future time instance
prs = []
L = length(scenarios[1])
for k in 1:L
    x = getindex.(scenarios, k)
    pr = proportions(x,p.NUM_HIDDENSTATES) 
    push!(prs, pr)
end


#----------- make a barplot using R's ggplot -------------------
dbar = DataFrame(y= vcat(prs...), 
                 x = repeat(0:10, inner=3), 
                 state= repeat(["1", "2", "3"], outer=11))
using RCall

@rput dbar
R"""
library(tidyverse)
mytheme = theme_bw()
theme_set(mytheme)  

dbar %>%  ggplot(aes(x=x,y=y, fill=state)) + geom_bar(stat="identity") + labs(x="time", y="state") +
scale_x_continuous(breaks=0:10)
ggsave("forward_latent.pdf", width=6, height=2.5)
"""



# following converts all Z values to λ values
λiters = [vcat( mapZtoλ(vals[i,11:13]), 
              mapZtoλ(vals[i,14:16]), 
              mapZtoλ(vals[i,17:19]), 
              mapZtoλ(vals[i,20:22]))  
        for i in 1:size(vals)[1] ]

λiters =  hcat(λiters...)'





𝒪 = 𝒪s[24]
pp1 = plot(getindex.(𝒪.X,2))
pp2 = plot(getindex.(𝒪.X,3))
pp3 = plot(getindex.(𝒪.X,4))
pp4 = plot(getindex.(𝒪.Y,1))
pp5 = plot(getindex.(𝒪.Y,2))
pp6 = plot(getindex.(𝒪.Y,3))
pp7 = plot(getindex.(𝒪.Y,4))
plot(pp1, pp2, pp3, pp4, pp5, pp6, pp7)

# few changes: 1(0), 14(1), 16(1), 