using CSV
using DataFrames
using JLD2 

wd = @__DIR__
cd(wd)

# Prior on root node (x can be inital state)
Πroot(_, p) = (@SVector ones(p.NUM_HIDDENSTATES))/p.NUM_HIDDENSTATES    


restricted = false
ztype = restricted ? Restricted() : Unrestricted() 

# read water polo data
d = CSV.read("olympic_athletes.csv", DataFrame; delim=",", missingstring="NA",
types= Dict(3=>Float64,4=>Float64,5=>Float64,6=>Int64,7=>Int64,8=>Int64,9=>Int64)) 

# x: (sport,strength,competition), cols 3:5
# y: (participation, modification, performance, symptoms), cols 6:9 (on binary scale)

n = 24
p = Pars(NUM_HIDDENSTATES = 3, DIM_COVARIATES= 27, DIM_RESPONSE = 4)

TX = Union{Missing, SVector{p.DIM_COVARIATES,Float64}} # indien er missing vals zijn 
TY = Union{Missing, SVector{p.DIM_RESPONSE, Int64}}

isanymissing(x) = maximum(ismissing.(x))  

n = 24  # nr of athletes
count_missingX = 0
count_missingY = 0
𝒪s = ObservationTrajectory{TX,TY}[]
for i ∈ 1:n
    di = d[d.ID .== i, :]
    X = TX[]
    Y = TY[]
    for r in eachrow(di)
        x = SA[1.0, r[3], r[4], r[5]]  # include intercept
        if isanymissing(x)
            push!(X, missing)
            count_missingX += 1
        else
            push!(X, x)
        end
        y = SA[r[6]+1, r[7]+1, r[8]+1, r[9]+1]
        if isanymissing(y)
            push!(Y, missing)
            count_missingY += 1
        else
            push!(Y, y)
        end
    end
    push!(𝒪s, ObservationTrajectory(X,Y))
end    

model = logtarget(ztype, 𝒪s, p);
#model = logtarget_large(ztype, 𝒪s, p);

#--------------- map -----------------------
@time map_estimate = optimize(model, MAP());
#coeftable(map_estimate)

θmap = convert_turingoutput(ztype, map_estimate, p);

@show θmap[:γ12] 
@show θmap[:γ21] 

@show mapallZtoλ(θmap)'

# ----------- mcmc ---------------------------
sampler =  NUTS() 
@time chain = sample(model, sampler, MCMCDistributed(), 500, 3; progress=true);

plot(chain)
savefig(wd*"/figs/olympic_histograms_traces.pdf")

histogram(chain)
savefig(wd*"/figs/histograms_traces.pdf")

summarize(chain; sections=[:parameters])

# extract posterior mean from mcmc output
θs = describe(chain)[1].nt.mean
names = String.(describe(chain)[1].nt.parameters)

@warn "We assume here 4 questions (hence Z1,...,Z4). Adapt if different"
if restricted 
    γup_ = θs[occursin.("γup", names)]
    γdown_ = θs[occursin.("γdown", names)]
    Z1_ = θs[occursin.("Z0", names)]
    Z2_ = θs[occursin.("Z0", names)]
    Z3_ = θs[occursin.("Z0", names)]
    Z4_ = θs[occursin.("Z0", names)]
    θpm = ComponentArray(γ12=γup_, γ21=γdown_, Z1=Z1_, Z2=Z2_, Z3=Z3_, Z4=Z4_)
else
    γup_ = θs[occursin.("γup", names)]
    γdown_ = θs[occursin.("γdown", names)]
    Z1_ = θs[occursin.("Z1", names)]
    Z2_ = θs[occursin.("Z2", names)]
    Z3_ = θs[occursin.("Z3", names)]
    Z4_ = θs[occursin.("Z4", names)]
    θpm = ComponentArray(γ12=γup_, γ21=γdown_, Z1=Z1_, Z2=Z2_, Z3=Z3_, Z4=Z4_)
end

λs = mapallZtoλ(θpm)'
@show λs

@show θpm[:γ12]

@show θpm[:γ21]

# save objects 
jldsave("ex_olympicathletes.jld2"; 𝒪s, model, θpm, λs, chain, ztype)
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