# Y_i depends on U_i
# U_i depends on U_{i-1}, X_i
# u_1 depends on Πroot(X1)



struct ObservationTrajectory{S,T}
    X::Vector{S}  # vector of covariates (each element of X contains the covariates at a particular time instance)
    Y::Vector{T}  # vector of responses (each element of Y contains a K-vector of responses to the K questions)
end
#ObservationTrajectory(X,  DIM_RESPONSE) = ObservationTrajectory(X, fill(fill(1,DIM_RESPONSE), length(X)))  # constructor if only X is given

ObservationTrajectory(X, _) = ObservationTrajectory(X, fill(SA[1,1,1,1], length(X)))  # constructor if only X is given

# Prior on root node (x can be inital state)
Πroot(_, p) = (@SVector ones(p.NUM_HIDDENSTATES))/p.NUM_HIDDENSTATES    

# transition kernel of the latent chain assuming 3 latent states
#Ki(θ,x) = [StatsFuns.softmax([0.0, dot(x,θ.γ12), -Inf])' ; StatsFuns.softmax([dot(x,θ.γ21), 0.0, dot(x,θ.γ23)])' ; StatsFuns.softmax([-Inf, dot(x,θ.γ32), 0])']
# to avoid type instability, both Ki methods should return an element of the same type 
# slightly faster, though almost double allocation

Ki(θ,x,p)= SMatrix{p.NUM_HIDDENSTATES,p.NUM_HIDDENSTATES}( 
    NNlib.softmax([0.0 dot(x,θ.γ12) -Inf64; dot(x,θ.γ21) 0.0 dot(x,θ.γ23) ; -Inf64 dot(x,θ.γ32) 0.0];dims=2) ) 
    
Ki(_,::Missing,p) = SMatrix{p.NUM_HIDDENSTATES,p.NUM_HIDDENSTATES}(1.0I)
 
scaledandshifted_logistic(x) = 2.0logistic(x) -1.0 # function that maps [0,∞) to [0,1)

"""
    pullback(θ,x,h)  
        
    returns  Ki(θ,x)*h
"""
function pullback(θ,x,h) 
    a1 = dot(StatsFuns.softmax(SA[0.0, dot(x,θ.γ12), -Inf64]),h)
    a2 = dot(StatsFuns.softmax(SA[dot(x,θ.γ21), 0.0 ,dot(x,θ.γ23)]),h)
    a3 = dot(StatsFuns.softmax(SA[-Inf64, dot(x,θ.γ32), 0.0]),h)
    SA[a1,a2,a3]
end

"""
    pullback(_, ::Missing, h)

    returns pullback in case covariates are missing
    as we assume no state change in this case, this simply returns h
"""
pullback(_, ::Missing, h) = h


mapZtoλ(x) = scaledandshifted_logistic.(cumsum(x))

"""
    response(Z) 
    
    make matrix [1.0-λ[1] λ[1]; 1.0-λ[2] λ[2]; 1.0-λ[3] λ[3]] (if 3 latent vars)

    # construct transition kernel Λ to observations
    # λ1, λ2, λ3 is formed by setting λi = logistic(cumsum(Z)[i])
"""
function response(Z) 
        λ = mapZtoλ(Z)
        SA[ one(λ[1])-λ[1] λ[1];  one(λ[2])-λ[2] λ[2];  one(λ[3])-λ[3] λ[3]]
end

# function response(Z::Vector{T},p) where T  # slightly slower, but generic
#     λ = mapZtoλ(Z)
#     v1 = SVector{p.NUM_HIDDENSTATES,T}(λ)
#     v2 = SVector{p.NUM_HIDDENSTATES,T}(one(T) .- λ)
#     hcat(v2, v1)
# end


Λi(θ) = SA[ response(θ.Z1), response(θ.Z2), response(θ.Z3), response(θ.Z4)    ]    # assume 4 questions


#sample_observation(Λ, u) =  [sample(Weights(Λ[i][u,:])) for i in eachindex(Λ)] # sample Y | U
sample_observation(Λ, u) =  SA[sample(Weights(Λ[1][u,:])), sample(Weights(Λ[2][u,:])), sample(Weights(Λ[3][u,:])), sample(Weights(Λ[4][u,:])) ] # sample Y | U

"""
    sample(θ, X)             

    X: vector of covariates, say of length n
    
    samples U_1,..., U_n and Y_1,..., Y_n, where 
    U_1 ~ Πroot
    for i ≥ 2 
        U_i | X_{i}, U_{i-1} ~ Row_{U_{i-1}} K(θ,X_{i})
    (thus, last element of X are not used)

"""
function sample(θ, X, p)            # Generate exact track + observations
    Λ = Λi(θ)
    uprev = sample(Weights(Πroot(X[1],p)))                  # sample u1 (possibly depending on X[1])
    U = [uprev]
    for i in eachindex(X[2:end])
        u = sample(Weights(Ki(θ,X[i],p)[uprev,:]))         # Generate sample from previous state
        push!(U, copy(u))
        uprev = u
    end
    Y = [sample_observation(Λ, u) for u ∈ U]
    U, Y
end


h_from_one_observation(Λ, i::Int) = Λ[:,i]

"""
    h_from_observation(θ, y) 

    returns message sent by observation y, when the parameter vector is θ
"""
function h_from_observation(θ, y, _) 
    Λ = Λi(θ)
    h_from_one_observation(Λ[1],y[1]) .* h_from_one_observation(Λ[2],y[2]) .* h_from_one_observation(Λ[3],y[3]) .* h_from_one_observation(Λ[4],y[4])
end

# above one is twice faster and allocates half, the one below is generic
# function h_from_observation(θ, y, p) 
#     Λ = Λi(θ)
#     a1 = [h_from_one_observation(Λ[i],y[i]) for i in eachindex(y)] 
#     SA[[prod(getindex.(a1,k)) for k in 1:p.NUM_HIDDENSTATES]...]
# end



h_from_observation(_, ::Missing,p) =  @SVector ones(p.NUM_HIDDENSTATES)


"""
    normalise!(x)

    inplace change of x to x/sum(x)
    returns log(sum(x))
"""
function normalise(x)
    s = sum(x)
    x/s, log(s)
end


"""
    loglik_and_bif(θ, 𝒪::ObservationTrajectory)

    Compute loglikelihood and h-vectors from the backward information filter, for one ObservationTrajectory
"""
function loglik_and_bif(θ, 𝒪::ObservationTrajectory,p)
    @unpack X, Y = 𝒪
    N = length(Y) 
    h = h_from_observation(θ, Y[N], p)
    H = [h]
    loglik = zero(θ[1][1])
    for i in N:-1:2
        h = pullback(θ, X[i], h) .* h_from_observation(θ, Y[i-1], p)
#        c = normalise!(h)
        h, c = normalise(h)
        loglik += c
        pushfirst!(H, copy(ForwardDiff.value.(h)))
        #hprev = h
    end
    loglik += log(dot(h, Πroot(X[1], p)))
    (ll=loglik, H=H)          
end

"""
    loglik(θ, 𝒪::ObservationTrajectory) 

    Returns loglikelihood at θ for one ObservationTrajectory
"""    
function loglik(θ, 𝒪::ObservationTrajectory, p) 
    @unpack X, Y = 𝒪
    N = length(Y) 
    h = h_from_observation(θ, Y[N], p)
    loglik = zero(θ[1][1])
    for i in N:-1:2
        h = pullback(θ, X[i], h) .* h_from_observation(θ, Y[i-1], p)
        #c = normalise!(h)
        h, c = normalise(h)
        loglik += c
    end
    loglik + log(dot(h, Πroot(X[1],p)))
end


"""
    loglik(θ, 𝒪s::Vector)

    Returns loglikelihood at θ for multplies ObservationTrajectories
"""    
function loglik(θ, 𝒪s::Vector, p)
    ll = zero(θ[1][1])
    for i ∈ eachindex(𝒪s)
        ll += loglik(θ, 𝒪s[i], p)
    end
    ll 
end

loglik(𝒪, p) = (θ) -> loglik(θ, 𝒪, p) 

∇loglik(𝒪, p) = (θ) -> ForwardDiff.gradient(loglik(𝒪), θ, p)

# check
function sample_guided(θ, 𝒪, H, p)# Generate approximate track
    X = 𝒪.X
    N = length(H) # check -1?
    uprev = sample(Weights(Πroot(X[1], p) .* H[1])) # Weighted prior distribution
    uᵒ = [uprev]
    for i=2:N
        w = Ki(θ,X[i], p)[uprev,:] .* H[i]         # Weighted transition density
        u = sample(Weights(w))
        push!(uᵒ, u)
        uprev = u
    end
    uᵒ
end

function unitvec(k,K)
    ee = zeros(K); 
    ee[k] = 1.0
    SVector{K}(ee)
end

function viterbi(θ, 𝒪::ObservationTrajectory, p) 
    @unpack NUM_HIDDENSTATES = p
    @unpack X, Y = 𝒪
    N = length(Y) 
    h = h_from_observation(θ, Y[N], p)
    mls = [argmax(h)]  # m(ost) l(ikely) s(tate)
    h = unitvec(mls[1], NUM_HIDDENSTATES)
    #loglik = zero(θ[1][1])
    for i in N:-1:2
        h = pullback(θ, X[i], h) .* h_from_observation(θ, Y[i-1], p)
        #c = normalise!(h)
        pushfirst!(mls, argmax(h))
        h = unitvec(mls[1], NUM_HIDDENSTATES)
     #   loglik += c
    end
    #loglik + log(dot(h, Πroot(X[1])))
    mls
end


############ use of Turing to sample from the posterior ################

# model with λvector the same for all questions (less parameters)
@model function logtarget(𝒪s, p)
    γup ~ filldist(Normal(0,5), p.DIM_COVARIATES)#MvNormal(fill(0.0, 2), 2.0 * I)
    γdown ~ filldist(Normal(0,5), p.DIM_COVARIATES)  #MvNormal(fill(0.0, 2), 2.0 * I)
    Z0 ~ filldist(Exponential(), p.NUM_HIDDENSTATES) 
    Turing.@addlogprob! loglik(ComponentArray(γ12 = γup, γ21 = γdown, γ23 = γup, γ32 = γdown, Z1=Z0, Z2=Z0, Z3=Z0, Z4=Z0), 𝒪s, p)
end

# model with unequal λvector for questions (less parameters)
@model function logtarget_large(𝒪s, p)
    γup ~ filldist(Normal(0,5), p.DIM_COVARIATES)#MvNormal(fill(0.0, 2), 2.0 * I)
    γdown ~ filldist(Normal(0,5), p.DIM_COVARIATES)  #MvNormal(fill(0.0, 2), 2.0 * I)
    
    Z1 ~ filldist(Exponential(), p.NUM_HIDDENSTATES) 
    Z2 ~ filldist(Exponential(), p.NUM_HIDDENSTATES) 
    Z3 ~ filldist(Exponential(), p.NUM_HIDDENSTATES) 
    Z4 ~ filldist(Exponential(), p.NUM_HIDDENSTATES) 
    Turing.@addlogprob! loglik(ComponentArray(γ12 = γup, γ21 = γdown, γ23 = γup, γ32 = γdown, Z1=Z1, Z2=Z2, Z3=Z3, Z4=Z4), 𝒪s, p)
end

# now with different gammas
@model function logtarget_large(𝒪s, p)
    γ12 ~ filldist(Normal(0,5), p.DIM_COVARIATES)#MvNormal(fill(0.0, 2), 2.0 * I)
    γ13 ~ filldist(Normal(0,5), p.DIM_COVARIATES)#MvNormal(fill(0.0, 2), 2.0 * I)
    γ21 ~ filldist(Normal(0,5), p.DIM_COVARIATES)  #MvNormal(fill(0.0, 2), 2.0 * I)
    γ23 ~ filldist(Normal(0,5), p.DIM_COVARIATES)  #MvNormal(fill(0.0, 2), 2.0 * I)
    γ31 ~ filldist(Normal(0,5), p.DIM_COVARIATES)  #MvNormal(fill(0.0, 2), 2.0 * I)
    γ32 ~ filldist(Normal(0,5), p.DIM_COVARIATES)  #MvNormal(fill(0.0, 2), 2.0 * I)

    Z1 ~ filldist(Exponential(), p.NUM_HIDDENSTATES) 
    Z2 ~ filldist(Exponential(), p.NUM_HIDDENSTATES) 
    Z3 ~ filldist(Exponential(), p.NUM_HIDDENSTATES) 
    Z4 ~ filldist(Exponential(), p.NUM_HIDDENSTATES) 
    Turing.@addlogprob! loglik(ComponentArray(γ12 = γ12, γ13 = γ13, γ21 = γ21, γ23 = γ23, γ31 = γ31, γ32 = γ32, Z1=Z1, Z2=Z2, Z3=Z3, Z4=Z4), 𝒪s, p)
end



mapallZtoλ(θ) = hcat(mapZtoλ(θ.Z1), mapZtoλ(θ.Z2), mapZtoλ(θ.Z3), mapZtoλ(θ.Z4))

"""
    convert_turingoutput(optimised_model)

    Converts values of Z vectors to λ vectors

    Example usage: 
    map_estimate = optimize(model, MAP())
    convert_turingoutput(map_estimate)
"""
function convert_turingoutput(optimised_model)  # function is not yet adapted to p
    θ =  optimised_model.values
    ComponentArray(γ12=[θ[Symbol("γup[1]")], θ[Symbol("γup[2]")], θ[Symbol("γup[3]")]],
                      γ21=[θ[Symbol("γdown[1]")], θ[Symbol("γdown[2]")], θ[Symbol("γdown[3]")]],
                      Z1=[θ[Symbol("Z1[1]")], θ[Symbol("Z1[2]")], θ[Symbol("Z1[3]")]],
                      Z2=[θ[Symbol("Z2[1]")], θ[Symbol("Z2[2]")], θ[Symbol("Z2[3]")]],
                      Z3=[θ[Symbol("Z3[1]")], θ[Symbol("Z3[2]")], θ[Symbol("Z3[3]")]],
                      Z4=[θ[Symbol("Z4[1]")], θ[Symbol("Z4[2]")], θ[Symbol("Z4[3]")]]
                      )
end
