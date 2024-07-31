# to show full output of x, do   "show(stdout, "text/plain", x)"

using CSV
using DataFrames
using JLD2 
using RCall


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
p = Pars(NUM_HIDDENSTATES = 3, DIM_COVARIATES= 4, DIM_RESPONSE = 4) 

# turn the data into a vector of type ObservationTrajectory
TX = Union{Missing, SVector{p.DIM_COVARIATES,Float64}} # indien er missing vals zijn 
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

# visualisation of the data
𝒪 = 𝒪s[1]
pp1 = plot(getindex.(𝒪.X,2))
pp2 = plot(getindex.(𝒪.X,3))
pp3 = plot(getindex.(𝒪.X,4))
pp4 = plot(getindex.(𝒪.Y,1))
pp5 = plot(getindex.(𝒪.Y,2))
pp6 = plot(getindex.(𝒪.Y,3))
pp7 = plot(getindex.(𝒪.Y,4))
plot(pp1, pp2, pp3, pp4, pp5, pp6, pp7)
for i in eachindex(𝒪s)
    𝒪 = 𝒪s[i]
    c23 = cor(getindex.(𝒪.X,2), getindex.(𝒪.X,3))
    c34 = cor(getindex.(𝒪.X,3), getindex.(𝒪.X,4))
    c24 = cor(getindex.(𝒪.X,2), getindex.(𝒪.X,4))
    @show [c23, c34, c24]
end



##### define the model
restricted = false
ztype = restricted ? Restricted() : Unrestricted() 
model = logtarget(ztype, 𝒪s, p);









#--------------- map -----------------------
@time map_estimate = maximum_a_posteriori(model)
show(stdout, "text/plain", map_estimate.values)

# extract estimates in component array
names_map = String.(names(map_estimate.values)[1])
θmapval = map_estimate.values
θmap = getpars(θmapval, names_map)
@show mapallZtoλ(θmap)'

#--------------- mle -----------------------
mle_estimate = maximum_likelihood(model)

names_mle = String.(names(mle_estimate.values)[1])
θmleval = mle_estimate.values
θmle = getpars(θmleval, names_mle)
@show mapallZtoλ(θmle)'



# ----------- mcmc ---------------------------
sampler = Turing.NUTS()
@time chain = sample(model, sampler, MCMCThreads(), 2000, 4; progress=true)

plot(chain)
savefig(wd*"/figs/olympic_histograms_traces.pdf")


# write output to CSV
CSV.write(wd*"/figs/iterates.csv", DataFrame(chain))
summ = summarize(chain; sections=[:parameters])
posterior_summary = DataFrame(summ)
CSV.write(wd*"/figs/posterior_summary.csv", posterior_summary)

# show summary on console
show(stdout, "text/plain", summ)

# extract posterior mean from mcmc output
θs = describe(chain)[1].nt.mean
names_par = String.(describe(chain)[1].nt.parameters)
θpm = getpars(θs, names_par)
λs = mapallZtoλ(θpm)'
@show λs


# save objects 
jldsave("ex_olympicathletes_final.jld2"; 𝒪s, model, θpm, λs, chain, ztype, map_estimate) 


### to open again
aa = jldopen("ex_olympicathletes_final.jld2")
aa["𝒪s"]
###




#---------------- generating forward scenarios -------------------------

# extract info from HMC
# chain.value.data 500×32×3 Array{Float64, 3}, so this contains the iterates from the 3 chains

chain = aa["chain"] # if read via jldopen
vals = DataFrame(chain)
names_ = vcat(chain.name_map.parameters, chain.name_map.internals)


# set training load and initial latent status
trainingload = CSV.read("Covariates_standardised.csv", DataFrame)
X = [SA[x...] for x in eachrow(trainingload)]


"""
    sample_future_scenarios(chain, X, U0, p)

    chain: output of HMC
    X: vector of training scenarios
    U0 ∈ {1,2,3}: initial latent state

    returns tuple with scenarios and marginal probabilities at each future time instance 
"""
function sample_future_scenarios(chain, X, U0, p)
    Random.seed!(2)
    vals = DataFrame(chain)
    scenarios = Vector{Int64}[]
    for i ∈ 1:size(vals)[1]
        γ12 = collect(vals[i, 4:7])
        γ23 = collect(vals[i, 8:11])
        γ21 = collect(vals[i, 12:15])
        γ32 = collect(vals[i, 16:19])

        θ =  ComponentArray(γ12=γ12, γ21=γ21, γ23=γ23, γ32=γ32)
        
        latentpath = sample_latent(θ, X, U0, p, 1)
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
    (scenarios=scenarios, proportions=prs)
end

U0 = [1, 2, 3] # presently assumed latent states
S1 = sample_future_scenarios(chain, X, U0[1], p)
S2 = sample_future_scenarios(chain, X, U0[2], p)
S3 = sample_future_scenarios(chain, X, U0[3], p)
#----------- make a barplot using R's ggplot -------------------
L = length(X)

dbar1 = DataFrame(prob= vcat(S1.proportions...), week = repeat(0:L, inner=3), state= repeat(["1", "2", "3"], outer=L+1), athlete=fill("Athlete 1",3*(L+1)))
dbar2 = DataFrame(prob= vcat(S2.proportions...), week = repeat(0:L, inner=3), state= repeat(["1", "2", "3"], outer=L+1), athlete=fill("Athlete 2",3*(L+1)))
dbar3 = DataFrame(prob= vcat(S3.proportions...), week = repeat(0:L, inner=3), state= repeat(["1", "2", "3"], outer=L+1), athlete=fill("Athlete 3",3*(L+1)))

CSV.write(wd*"/figs/simulated_scenarios.csv", vcat(dbar1, dbar2, dbar3))


# @rput dbar
# R"""
# library(tidyverse)
# mytheme = theme_bw()
# theme_set(mytheme)  

# dbar %>%  ggplot(aes(x=week,y=prob, fill=state)) + geom_bar(stat="identity") + labs(x="time", y="state") +
# scale_x_continuous(breaks=0:10)
# ggsave("figs/forward_latent.pdf", width=6, height=2.5)
# """



# following converts all Z values to λ values
# λiters = [vcat( mapZtoλ(vals[i,11:13]), 
#               mapZtoλ(vals[i,14:16]), 
#               mapZtoλ(vals[i,17:19]), 
#               mapZtoλ(vals[i,20:22]))  
#         for i in 1:size(vals)[1] ]

# λiters =  hcat(λiters...)'








