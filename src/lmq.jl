# -------------------- model ----------------------------
    # observe Y_i, latent U_i, observe covariates X_i
    # Y_i depends on U_i
    # U_i depends on U_{i-1}, X_i
    # U_1 depends on Πroot(X1)

struct ObservationTrajectory{S,T}
    X::Vector{S}  # vector of covariates (each element of X contains the covariates at a particular time instance)
    Y::Vector{T}  # vector of responses (each element of Y contains a K-vector of responses to the K questions)
end

ObservationTrajectory(X, _) = ObservationTrajectory(X, fill(SA[1,1,1,1], length(X)))  # constructor if only X is given

"""
    Ki(θ,x,p)

    create transition probability matrix when parameter is θ and state is x
"""    
Ki(θ,x,p)= SMatrix{p.NUM_HIDDENSTATES,p.NUM_HIDDENSTATES}(  #not generic
    NNlib.softmax([0.0 dot(x,θ.γ12) -Inf64; 
                    dot(x,θ.γ21) 0.0 dot(x,θ.γ23);
                     -Inf64 dot(x,θ.γ32) 0.0];dims=2) ) 
    
Ki(_,::Missing,p) = SMatrix{p.NUM_HIDDENSTATES,p.NUM_HIDDENSTATES}(1.0I) #generic
 
"""
    pullback(θ,x,h)  
        
    returns  Ki(θ,x)*h
"""
function pullback(θ,x,h) # not generic
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

# mapping Z to λ (prior construction on λs)
scaledandshifted_logistic(x) = 2.0logistic(x) - 1.0 # function that maps [0,∞) to [0,1) 
mapZtoλ(x) = scaledandshifted_logistic.(cumsum(x))

"""
    response(Z) 
    
    make matrix [1.0-λ[1] λ[1]; 1.0-λ[2] λ[2]; 1.0-λ[3] λ[3]] (if 3 latent vars)

    # construct transition kernel Λ to observations
    # λ1, λ2, λ3 is formed by setting λi = logistic(cumsum(Z)[i])
"""
function response(Z) # not generic
        λ = mapZtoλ(Z)
        SA[ one(λ[1])-λ[1] λ[1];  one(λ[2])-λ[2] λ[2];  one(λ[3])-λ[3] λ[3]]
end


Λi(θ) = SA[ response(θ.Z1), response(θ.Z2), response(θ.Z3), response(θ.Z4)    ]    # not generic

"""
    sample_observation(Λ, u)
        sample Y | U
"""
sample_observation(Λ, u) =  SA[sample(Weights(Λ[1][u,:])), sample(Weights(Λ[2][u,:])), sample(Weights(Λ[3][u,:])), sample(Weights(Λ[4][u,:])) ] #not generic

"""
    sample(θ, X)             

    Generate track with initial value from prior + observations
    X: vector of covariates, say of length n
    
    samples U_1,..., U_n and Y_1,..., Y_n, where 
    U_1 ~ Πroot
    for i ≥ 2 
        U_i | X_{i}, U_{i-1} ~ Row_{U_{i-1}} K(θ,X_{i})
    (thus, last element of X are not used)

"""
function sample(θ, X, p)            # generic
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

"""
    sample_latent(θ, X, U0, p)

    # Generate track starting from U0 + for a future scenario contained in X
        γup = rand(4)
        γdown = rand(4)

        X = [SA[1,2,3,4], SA[4,5,6,7]]
        θ = ComponentArray(γ12=γup, γ21=γdown, γ23=γup, γ32=γdown)
        U0 = 2
        sample_latent(θ, X, U0, p)
"""
function sample_latent(θ, X, U0, p)            # generic
    uprev = U0
    U = [uprev]
    for i in eachindex(X)
        u = sample(Weights(Ki(θ,X[i],p)[uprev,:]))         # Generate sample from previous state
        push!(U, copy(u))
        uprev = u
    end    
    U
end

h_from_one_observation(Λ, i::Int) = Λ[:,i]

"""
    h_from_observation(θ, y) 

    returns message sent by observation y, when the parameter vector is θ
"""
function h_from_observation(θ, y, _) # not generic
    Λ = Λi(θ)
    h_from_one_observation(Λ[1],y[1]) .* h_from_one_observation(Λ[2],y[2]) .* h_from_one_observation(Λ[3],y[3]) .* h_from_one_observation(Λ[4],y[4])
end

h_from_observation(_, ::Missing,p) =  @SVector ones(p.NUM_HIDDENSTATES) # generic


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
function loglik_and_bif(θ, 𝒪::ObservationTrajectory,p) # generic
    @unpack X, Y = 𝒪
    N = length(Y) 
    h = h_from_observation(θ, Y[N], p)
    H = [h]
    loglik = zero(θ[1][1])
    for i in N:-1:2
        h = pullback(θ, X[i], h) .* h_from_observation(θ, Y[i-1], p)
        h, c = normalise(h)
        loglik += c
        pushfirst!(H, copy(ForwardDiff.value.(h)))
    end
    loglik += log(dot(h, Πroot(X[1], p)))
    (ll=loglik, H=H)          
end

"""
    loglik(θ, 𝒪::ObservationTrajectory) 

    Returns loglikelihood at θ for one ObservationTrajectory
"""    
function loglik(θ, 𝒪::ObservationTrajectory, p) # generic
    @unpack X, Y = 𝒪
    N = length(Y) 
    h = h_from_observation(θ, Y[N], p)
    loglik = zero(θ[1][1])
    for i in N:-1:2
        h = pullback(θ, X[i], h) .* h_from_observation(θ, Y[i-1], p)
        h, c = normalise(h)
        loglik += c
    end
    loglik + log(dot(h, Πroot(X[1],p)))
end


"""
    loglik(θ, 𝒪s::Vector)

    Returns loglikelihood at θ for multplies ObservationTrajectories
"""    
function loglik(θ, 𝒪s::Vector, p) # generic
    ll = zero(θ[1][1])
    for i ∈ eachindex(𝒪s)
        ll += loglik(θ, 𝒪s[i], p)
    end
    ll 
end

loglik(𝒪, p) = (θ) -> loglik(θ, 𝒪, p) 

∇loglik(𝒪, p) = (θ) -> ForwardDiff.gradient(loglik(𝒪, p), θ)

# check
function sample_guided(θ, 𝒪, H, p)              # Generate approximate track
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
