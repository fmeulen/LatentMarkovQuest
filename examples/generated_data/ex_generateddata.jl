########### An example, where data are generated from the model ####################
using RCall
using CSV
using JLD2

wd = @__DIR__
cd(wd)

# Prior on root node (x can be inital state)
Πroot(_, p) = (@SVector ones(p.NUM_HIDDENSTATES))/p.NUM_HIDDENSTATES    


# True parameter vector
γup = [2.0, 0.0]
γdown = [-0.5, -0.5]

p = Pars()

Z1 = [0.5, 1.0, 1.5]
Z2 = [0.5, 1.0, 1.5]
Z3 = [0.2, 1.0, 2.5]
Z4 = [0.5, 1.0, 1.5]

Z = [0.5, 1.0, 1.5]

restricted = true#false
θ0 =  restricted ? ComponentArray(γ12 = γup, γ21 = γdown, γ23 = γup, γ32 = γdown, Z1=Z, Z2=Z, Z3=Z, Z4=Z) : ComponentArray(γ12 = γup, γ21 = γdown, γ23 = γup, γ32 = γdown, Z1=Z1, Z2=Z2, Z3=Z3, Z4=Z4)

ztype = restricted ? Restricted() : Unrestricted() 

println("true vals", "  ", γup,"  ", γdown,"  ", Z1, Z2, Z3, Z4)

# generate covariates, el1 = intensity, el2 = gender
n = 25 # nr of subjects
T = 50 # nr of times at which we observe

# generate latent Markov process and observations (returns array of ObservationTrajectory)

INCLUDE_MISSING  = false
p = Pars(DIM_COVARIATES=2)

Random.seed!(9)

if INCLUDE_MISSING
    TX = Union{Missing, SVector{p.DIM_COVARIATES,Float64}} # indien er missing vals zijn 
    TY = Union{Missing, SVector{p.DIM_RESPONSE, Int64}}
  
    𝒪s = ObservationTrajectory{TX, TY}[]
    Us =  Vector{Int64}[]
    for i in 1:n
        #local X 
        X = TX[]   # next, we can push! elements to X
        if i ≤ 10 
            slope = rand(Uniform(-0.05,0.05))
            for t in 1: T
                push!(X, SA[slope*t + 0.1*randn(), 0.0])
            end
        else
            slope = rand(Uniform(-0.05,0.05))
            for t in 1: T
                push!(X, SA[slope*t + 0.1*randn(), 1.0])
            end
            X[3] = missing
        end
        U, Y =  sample(θ0, X, p) 
        push!(Us, U)
        YY = TY[]
        push!(YY, missing) 
        for t in  2:T
            push!(YY, Y[t]) 
        end    
        push!(𝒪s, ObservationTrajectory(X, YY))
    end
else 
    TX = SVector{p.DIM_COVARIATES,Float64}
    TY = SVector{p.DIM_RESPONSE, Int64}

    𝒪s = ObservationTrajectory{TX, TY}[]
    Us =  Vector{Int64}[]
    for i in 1:n
        #local X 
        X = TX[]   # next, we can push! elements to X
        if i ≤ 10 
            slope = rand(Uniform(-0.05,0.05))
            for t in 1: T
                push!(X, SA[slope*t + 0.1*randn(), 0.0])
            end
        else
            slope = rand(Uniform(-0.05,0.05))
            for t in 1: T
                push!(X, SA[slope*t + 0.1*randn(), 1.0])
            end
        end
        U, Y =  sample(θ0, X, p) 
        push!(Us, U)
        YY = TY[]
        for t in  1:T
            push!(YY, Y[t]) 
        end    
        push!(𝒪s, ObservationTrajectory(X, YY))
    end
end

#### convert the simulated data to a Julia-dataframe suitable for fitting in R's lmest
out = []
for i ∈ 1:n
    𝒪 = 𝒪s[i]
    @unpack X, Y = 𝒪
    Y = [Y[j] .- SA[1,1,1,1] for j in eachindex(Y)]
    xx=vcat(X'...)
    yy=vcat(Y'...)
    ni = size(yy)[1]
    push!(out, hcat(fill(i,ni),1:ni,xx,yy))
end

dout = DataFrame(vcat(out...), :auto)
colnames = ["subject", "time", "x1", "x2", "y1", "y2", "y3", "y4"]
rename!(dout, colnames)


#CSV.write(joinpath(packdir,"datasets/generated_testdata.csv"), dout)

dout = CSV.read("generated_testdata.csv",DataFrame)

#### Fit with LMest #####

@rput dout
R"""
library(LMest)

#require(LMest)
dt <- lmestData(data = dout, id = "subject", time="time")

lmestF <- lmestFormula(data=dout, response=6:9, 
                        LatentInitial=NULL, 
                        LatentTransition=3:5,
                        AddInterceptInitial = FALSE,
                        AddInterceptTransition = FALSE)

 
out0 = lmest(responsesFormula= lmestF$responsesFormula,
             latentFormula = lmestF$latentFormula,   
                index = c("subject", "time"),
                data = dt,
                k = 3,
                start = 0, # 0 deterministic, 1 random type of starting values
                modBasic = 1,
                seed = 123,
                tol = 1e-2) 

# out1 <- lmest(responsesFormula = y1 + y2 + y3 + y4 ~ NULL,
#               latentFormula = ~ 1 | x1 + x2,
#               index = c("subject", "time"),
#               data = dt,
#               k = 3,
#               start = 0, # 0 deterministic, 1 random type of starting values
#               modBasic = 1,
#               seed = 123,
#               tol = 1e-2,)
# summary(out1)
# lambdas = out1$Psi
# gammas = out1$Ga
"""

# important: in LMest output, the Psi is related to our lambdas (conditional respondse probabilities)
# the Ga is related to our gammas (parameters affecting the logit for the transition probabilities)
lmest_fit0 = @rget out0
#lmest_fit1 = @rget out1

lmest_fit0[:Ga]

lmest_fit0[:Piv]

@show lmest_fit0[:Psi] # bottom row should resemble are lambdas
# get the bottom rows, the following should be close (if estimates are good)
@show [lmest_fit0[:Psi][:,:,1][2,:],lmest_fit0[:Psi][:,:,2][2,:],lmest_fit0[:Psi][:,:,3][2,:],lmest_fit0[:Psi][:,:,4][2,:]]
@show mapallZtoλ(θ0)'


#################### Fitting with Turing.jl ##########################

model = logtarget(ztype, 𝒪s, p);


#--------------- map -----------------------
@time map_estimate = optimize(model, MAP());
θmap = convert_turingoutput(ztype, map_estimate, p);
@show mapallZtoλ(θ0)'
@show mapallZtoλ(θmap)'

@show θ0[:γ12] 
θmap[:γ12]

@show θ0[:γ21] 
θmap[:γ21]

#--------------- mle -----------------------
@time mle_estimate = optimize(model, MLE(), NelderMead())
#@edit optimize(model, MLE(), NelderMead())
θmle = convert_turingoutput(ztype, mle_estimate, p);
@show mapallZtoλ(θ0)'
@show mapallZtoλ(θmle)'

@show θ0[:γ12] 
θmle[:γ12]

@show θ0[:γ21] 
θmle[:γ21]


ForwardDiff.gradient(loglik(𝒪s, p), θ0)
ForwardDiff.gradient(loglik(𝒪s, p), θmle)
hess = ForwardDiff.hessian(loglik(𝒪s, p), θmle)
isposdef(-hess)
eigen(hess)
issymmetric(hess)

#--------------- NUTS sampler -----------------------

sampler =  NUTS() 
#sampler = HMC(0.01,10)

# if initial ϵ=0.05 works.
@time chain = sample(model, sampler, MCMCDistributed(), 500, 3; progress=true);

# plotting 
histogram(chain)
savefig(wd*"/figs/histograms.pdf")
plot(chain)
savefig(wd*"/figs/histograms_traces.pdf")

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

# compare posterior means to true values
@show mapallZtoλ(θpm)'
@show mapallZtoλ(θ0)'

@show θ0[:γ12] 
@show θpm[:γ12]

@show θ0[:γ21] 
@show θpm[:γ21]

# can finetune graphs like below
# plot(
#     traceplot(chain[Z1symb], title="traceplot"),
#     #meanplot(chain[Z1symb], title="meanplot"),
#     density(chain[Z1symb], title="density"),
#     #histogram(chain[Z1symb], title="histogram"),
#     #autocorplot(chain[Z1symb], title="autocorplot"),
#     dpi=300, size=(900, 900))


# γsymb=[Symbol("γup[1]"), Symbol("γup[2]"), Symbol("γdown[1]"), Symbol("γdown[2]")]

# plot(chain[γsymb])
# savefig(joinpath(packdir,"figs/gammas.pdf"))


#methodswith(MCMCChains.Chains) #to know methods name which we can apply on chain object


# compare LMest and this implementation (posterior mean)

# λs
@show mapallZtoλ(θ0)'
@show [lmest_fit0[:Psi][:,:,1][2,:],lmest_fit0[:Psi][:,:,2][2,:],lmest_fit0[:Psi][:,:,3][2,:],lmest_fit0[:Psi][:,:,4][2,:]]
@show mapallZtoλ(θpm)'

# γs
@show θ0[:γ12], θpm[:γ12]
@show θ0[:γ21], θpm[:γ21]
lmest_fit0[:Ga]

lmest_fit0[:Piv]





# save objects 
jldsave("ex_generateddata.jld2"; 𝒪s, model, θpm, λs, chain, ztype)


### to open again
# aa = jldopen("ex_generateddata.jld2")
# aa["𝒪s"]
###

