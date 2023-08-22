using CSV
using DataFrames

restricted = false
ztype = restricted ? Restricted() : Unrestricted() 

# read water polo data
d = CSV.read(joinpath(packdir, "datasets/olympic_athletes.csv"), DataFrame; delim=",", missingstring="NA",
types= Dict(3=>Float64,4=>Float64,5=>Float64,6=>Int64,7=>Int64,8=>Int64,9=>Int64)) 

# x: (sport,strength,competition), cols 3:5
# y: (participation, modification, performance, symptoms), cols 6:9 (on binary scale)

p = Pars(NUM_HIDDENSTATES = 3, DIM_COVARIATES= 3, DIM_RESPONSE = 4)
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
        x = SA[r[3], r[4], r[5]]
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
θmap = convert_turingoutput(ztype, map_estimate);

@show θmap[:γ12] 
@show θmap[:γ21] 

@show mapallZtoλ(θmap)'


sampler =  NUTS() 
@time chain = sample(model, sampler, MCMCDistributed(), 1000, 3; progress=true);

plot(chain)
savefig(joinpath(packdir,"figs/olympic_histograms_traces.pdf"))

histogram(chain)
savefig(joinpath(packdir,"figs/histograms_traces.pdf"))


describe(chain)[1]
θpm = describe(chain)[1].nt.mean
pDC = p.DIM_COVARIATES
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
