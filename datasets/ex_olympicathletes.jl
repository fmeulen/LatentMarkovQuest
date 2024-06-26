using CSV
using DataFrames
using JLD2

mkpath("olympic")
olympdir = packdir*"/src/olympic"

restricted = false
ztype = restricted ? Restricted() : Unrestricted() 

# read water polo data
d = CSV.read(joinpath(packdir, "datasets/olympic_athletes.csv"), DataFrame; delim=",", missingstring="NA",
types= Dict(3=>Float64,4=>Float64,5=>Float64,6=>Int64,7=>Int64,8=>Int64,9=>Int64)) 

# x: (sport,strength,competition), cols 3:5
# y: (participation, modification, performance, symptoms), cols 6:9 (on binary scale)

p = Pars(NUM_HIDDENSTATES = 3, DIM_COVARIATES= 4, DIM_RESPONSE = 4)
TX = Union{Missing, SVector{p.DIM_COVARIATES,Float64}} # indien er missing vals zijn 
TY = Union{Missing, SVector{p.DIM_RESPONSE, Int64}}

isanymissing(x) = maximum(ismissing.(x))  

n = 24
𝒪s = ObservationTrajectory{TX,TY}[]
for i ∈ 1:n
    di = d[d.ID .== i, :]
    X = TX[]
    Y = TY[]
    for r in eachrow(di)
        x = SA[1.0, r[3], r[4], r[5]]  # include intercept
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

model = logtarget(ztype, 𝒪s, p);


#--------------- map -----------------------
@time map_estimate = optimize(model, MAP());
θmap = convert_turingoutput(ztype, map_estimate, p);

@show θmap[:γ12] 
@show θmap[:γ21] 

@show mapallZtoλ(θmap)'


sampler =  NUTS() 
@time chain = sample(model, sampler, MCMCDistributed(), 500, 3; progress=true);

plot(chain)
savefig(joinpath(packdir,"/src/olympicathletes/olympic_histograms_traces.pdf"))

histogram(chain)
savefig(joinpath(packdir,"/sir/olympicathletes/histograms_traces.pdf"))


describe(chain)[1]
θpm = describe(chain)[1].nt.mean
pDC = p.DIM_COVARIATES

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




pNHS = p.NUM_HIDDENSTATES
if restricted 
    θpm = ComponentArray(γ12=θpm[1:pDC], γ21=θpm[pDC+1:2pDC], Z1=θpm[2pDC+1:3pDC], 
                    Z2=θpm[2pDC+1:3pDC], Z3=θpm[2pDC+1:3pDC], Z4=θpm[2pDC+1:3pDC])
else
    θpm = ComponentArray(γ12=θpm[1:pDC], γ21=θpm[pDC+1:2pDC], Z1=θpm[2pDC+1:3pDC], 
                    Z2=θpm[(3pDC+1):(3pDC+pNHS)],Z3=θpm[(3pDC+pNHS+1):(3pDC+2pNHS)],Z4=θpm[(3pDC+2pNHS+1):(3pDC+3pNHS)])
end


@show mapallZtoλ(θpm)'

@show θpm[:γ12]

@show θpm[:γ21]

# save objects
jldsave("ex_olympicathletes.jld2"; 𝒪s, model, θmap, sampler, chain, θpm)