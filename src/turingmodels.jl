abstract type Ztype end
struct Restricted <: Ztype end  # same λ for all questions (bit of toy model)
struct Unrestricted <: Ztype end # # separate λ for all questions

# model with λvector the same for all questions (less parameters)
@model function logtarget(::Restricted, 𝒪s, p)
    γup ~ filldist(Normal(0,5), p.DIM_COVARIATES)
    γdown ~ filldist(Normal(0,5), p.DIM_COVARIATES)  
    Z0 ~ filldist(Exponential(), p.NUM_HIDDENSTATES) 
    Turing.@addlogprob! loglik(ComponentArray(γ12 = γup, γ21 = γdown, γ23 = γup, γ32 = γdown, Z1=Z0, Z2=Z0, Z3=Z0, Z4=Z0), 𝒪s, p)
end

@model function logtarget(::Unrestricted, 𝒪s, p)
    γup ~ filldist(Normal(0,5), p.DIM_COVARIATES)
    γdown ~ filldist(Normal(0,5), p.DIM_COVARIATES)  
    
    Z1 ~ filldist(Exponential(), p.NUM_HIDDENSTATES) 
    Z2 ~ filldist(Exponential(), p.NUM_HIDDENSTATES) 
    Z3 ~ filldist(Exponential(), p.NUM_HIDDENSTATES) 
    Z4 ~ filldist(Exponential(), p.NUM_HIDDENSTATES) 
    Turing.@addlogprob! loglik(ComponentArray(γ12 = γup, γ21 = γdown, γ23 = γup, γ32 = γdown, Z1=Z1, Z2=Z2, Z3=Z3, Z4=Z4), 𝒪s, p)
end

# full model with state dependend γs
@model function logtarget_large(::Unrestricted, 𝒪s, p)
    γ12 ~ filldist(Normal(0,5), p.DIM_COVARIATES)
    γ13 ~ filldist(Normal(0,5), p.DIM_COVARIATES)
    γ21 ~ filldist(Normal(0,5), p.DIM_COVARIATES)
    γ23 ~ filldist(Normal(0,5), p.DIM_COVARIATES)
    γ31 ~ filldist(Normal(0,5), p.DIM_COVARIATES)
    γ32 ~ filldist(Normal(0,5), p.DIM_COVARIATES)

    Z1 ~ filldist(Exponential(), p.NUM_HIDDENSTATES) 
    Z2 ~ filldist(Exponential(), p.NUM_HIDDENSTATES) 
    Z3 ~ filldist(Exponential(), p.NUM_HIDDENSTATES) 
    Z4 ~ filldist(Exponential(), p.NUM_HIDDENSTATES) 
    Turing.@addlogprob! loglik(ComponentArray(γ12 = γ12, γ13 = γ13, γ21 = γ21, γ23 = γ23, γ31 = γ31, γ32 = γ32, 
                                            Z1=Z1, Z2=Z2, Z3=Z3, Z4=Z4), 𝒪s, p)
end


