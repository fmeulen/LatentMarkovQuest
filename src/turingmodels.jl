
"""
    Ki(θ,x,p)

    create transition probability matrix when parameter is θ and state is x
"""    
function Ki(θ, x, p, i)   #not generic, restrict transitions to neighboring states
    αup = θ.αup[i]
    αdown = θ.αdown[i]
    γ12 = SA[αup, θ.γ12...]
    γ23 = SA[αup, θ.γ23...]
    γ21 = SA[αdown, θ.γ21...]
    γ32 = SA[αdown, θ.γ32...]
    SMatrix{p.NUM_HIDDENSTATES,p.NUM_HIDDENSTATES}
    (       NNlib.softmax([0.0 dot(x,γ12) -Inf64; 
                           dot(x,γ21) 0.0 dot(x,γ23);
                           -Inf64 dot(x,γ32) 0.0]
    ;dims=2) ) 
    # a12 = logistic(dot(x,γ12))
    # a21 = logistic(dot(x,γ21))
    # a23 = logistic(dot(x,γ23))
    # a32 = logistic(dot(x,γ32))
    # @SMatrix [1.0-a12  a12 0.0; 
    #           a21 1.0-a21-a23 a23;
    #           0.0 a32 1.0-a32]
end

Ki(_,::Missing, p, i) = SMatrix{p.NUM_HIDDENSTATES,p.NUM_HIDDENSTATES}(1.0I) #generic


abstract type Ztype end
struct Restricted <: Ztype end  # same λ for all questions (bit of toy model)
struct Unrestricted <: Ztype end # # separate λ for all questions

# model with λvector the same for all questions (less parameters)
@model function logtarget(::Restricted, 𝒪s, p; σ=3)
    n = length(𝒪s)
    σ² ~ InverseGamma(0.1, 0.1)
    σα² ~ InverseGamma(0.1, 0.1)
    
    γup ~ filldist(Normal(0,sqrt(σ²)), p.DIM_COVARIATES)
    γdown ~ filldist(Normal(0,sqrt(σ²)), p.DIM_COVARIATES)  
    αup ~ filldist(Normal(0,sqrt(σα²)), n)  
    αdown ~ filldist(Normal(0,sqrt(σα²)), n)  

    Z0 ~ filldist(Exponential(), p.NUM_HIDDENSTATES) 
    Turing.@addlogprob! loglik(ComponentArray(γ12 = γup, γ21 = γdown, γ23 = γup,
     γ32 = γdown, Z1=Z0, Z2=Z0, Z3=Z0, Z4=Z0, αup=αup, αdown=αdown), 𝒪s, p)
end

@model function logtarget(::Unrestricted, 𝒪s, p; σ=3)
    n = length(𝒪s)
    σ² ~ InverseGamma(0.1, 0.1)
    σα² ~ InverseGamma(0.1, 0.1)
    
    γup ~ filldist(Normal(0,sqrt(σ²)), p.DIM_COVARIATES)
    γdown ~ filldist(Normal(0,sqrt(σ²)), p.DIM_COVARIATES)  
    αup ~ filldist(Normal(0,sqrt(σα²)), n)  
    αdown ~ filldist(Normal(0,sqrt(σα²)), n)  

    Z1 ~ filldist(Exponential(), p.NUM_HIDDENSTATES) 
    Z2 ~ filldist(Exponential(), p.NUM_HIDDENSTATES) 
    Z3 ~ filldist(Exponential(), p.NUM_HIDDENSTATES) 
    Z4 ~ filldist(Exponential(), p.NUM_HIDDENSTATES) 
    Turing.@addlogprob! loglik(ComponentArray(γ12 = γup, γ21 = γdown, γ23 = γup, 
        γ32 = γdown, Z1=Z1, Z2=Z2, Z3=Z3, Z4=Z4, αup=αup, αdown=αdown), 𝒪s, p)
end

# full model with state dependend γs
@model function logtarget_large(::Unrestricted, 𝒪s, p; σ=3)
    n = length(𝒪s)

    σ² ~ InverseGamma(0.1, 0.1)
    σα² ~ InverseGamma(0.1, 0.1)

    γ12 ~ filldist(Normal(0,sqrt(σ²)), p.DIM_COVARIATES)
    γ13 ~ filldist(Normal(0,sqrt(σ²)), p.DIM_COVARIATES)
    γ21 ~ filldist(Normal(0,sqrt(σ²)), p.DIM_COVARIATES)
    γ23 ~ filldist(Normal(0,sqrt(σ²)), p.DIM_COVARIATES)
    γ31 ~ filldist(Normal(0,sqrt(σ²)), p.DIM_COVARIATES)
    γ32 ~ filldist(Normal(0,sqrt(σ²)), p.DIM_COVARIATES)
    αup ~ filldist(Normal(0,sqrt(σα²)), n)  
    αdown ~ filldist(Normal(0,sqrt(σα²)), n)  
    

    Z1 ~ filldist(Exponential(), p.NUM_HIDDENSTATES) 
    Z2 ~ filldist(Exponential(), p.NUM_HIDDENSTATES) 
    Z3 ~ filldist(Exponential(), p.NUM_HIDDENSTATES) 
    Z4 ~ filldist(Exponential(), p.NUM_HIDDENSTATES) 
    Turing.@addlogprob! loglik(ComponentArray(γ12 = γ12, γ13 = γ13, γ21 = γ21, γ23 = γ23, γ31 = γ31, γ32 = γ32, 
                                            Z1=Z1, Z2=Z2, Z3=Z3, Z4=Z4, αup=αup, αdown=αdown), 𝒪s, p)
end


