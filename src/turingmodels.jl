
"""
    Ki(θ,x,p)

    create transition probability matrix when parameter is θ and state is x
"""    
function Ki(θ, x, p, i)   #not generic, restrict transitions to neighboring states
       SMatrix{p.NUM_HIDDENSTATES,p.NUM_HIDDENSTATES}
    (       NNlib.softmax([0.0 dot(x,θ.γ12) -Inf64; 
                           dot(x,θ.γ21) 0.0 dot(x,θ.γ23);
                           -Inf64 dot(x,θ.γ32) 0.0]
    ;dims=2) ) 
end
 
Ki(_,::Missing, p, i) = SMatrix{p.NUM_HIDDENSTATES,p.NUM_HIDDENSTATES}(1.0I) #generic


abstract type Ztype end
struct Restricted <: Ztype end  # same λ for all questions (bit of toy model)
struct Unrestricted <: Ztype end # # separate λ for all questions


@model function logtarget(::Unrestricted, 𝒪s, p)
    σ ~ Exponential(3.0)
    
    γ12 ~ filldist(Normal(0.0, σ), p.DIM_COVARIATES)
    γ23 ~ filldist(Normal(0.0, σ), p.DIM_COVARIATES)
    γ21 ~ filldist(Normal(0.0, σ), p.DIM_COVARIATES)  
    γ32 ~ filldist(Normal(0.0, σ), p.DIM_COVARIATES)  


    Z1 ~ filldist(Exponential(), p.NUM_HIDDENSTATES) 
    Z2 ~ filldist(Exponential(), p.NUM_HIDDENSTATES) 
    Z3 ~ filldist(Exponential(), p.NUM_HIDDENSTATES) 
    Z4 ~ filldist(Exponential(), p.NUM_HIDDENSTATES) 

    θ = ComponentArray(γ12 = γ12, γ23 = γ23, γ21 = γ21, γ32 = γ32, Z1=Z1, Z2=Z2, Z3=Z3, Z4=Z4)

    Turing.@addlogprob! loglik(θ, 𝒪s, p)
end
