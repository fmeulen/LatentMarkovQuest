
function unitvec(k,K)
    ee = zeros(K); 
    ee[k] = 1.0
    SVector{K}(ee)
end

function viterbi(θ, 𝒪::ObservationTrajectory, p) # generic
    @unpack NUM_HIDDENSTATES = p
    @unpack X, Y = 𝒪
    N = length(Y) 
    h = h_from_observation(θ, Y[N], p)
    mls = [argmax(h)]  # m(ost) l(ikely) s(tate)
    h = unitvec(mls[1], NUM_HIDDENSTATES)
    #loglik = zero(θ[1][1])
    for i in N:-1:2
        h = pullback(θ, X[i], h) .* h_from_observation(θ, Y[i-1], p)
        pushfirst!(mls, argmax(h))
        h = unitvec(mls[1], NUM_HIDDENSTATES)
    end
    mls
end

