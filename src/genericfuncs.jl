# all is set up for efficiency with 3 latent states and 4 questions. 
# This file contains generic functions that tend to be slightly slower. 

ObservationTrajectory(X,  DIM_RESPONSE) = 
        ObservationTrajectory(X, fill(fill(1,DIM_RESPONSE), length(X)))  

function response(Z::Vector{T},p) where T  # slightly slower, but generic
    λ = mapZtoλ(Z)
    v1 = SVector{p.NUM_HIDDENSTATES,T}(λ)
    v2 = SVector{p.NUM_HIDDENSTATES,T}(one(T) .- λ)
    hcat(v2, v1)
end

sample_observation(Λ, u) =  [sample(Weights(Λ[i][u,:])) for i in eachindex(Λ)] # sample Y | U

# below is twice slower and allocates double compared to non generic
function h_from_observation(θ, y, p) 
    Λ = Λi(θ)
    a1 = [h_from_one_observation(Λ[i],y[i]) for i in eachindex(y)] 
    SA[[prod(getindex.(a1,k)) for k in 1:p.NUM_HIDDENSTATES]...]
end

