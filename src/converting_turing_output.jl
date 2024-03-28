# this works, but is somewhat ugly


mapallZtoλ(θ) = hcat(mapZtoλ(θ.Z1), mapZtoλ(θ.Z2), mapZtoλ(θ.Z3), mapZtoλ(θ.Z4))

"""
    convert_turingoutput(optimised_model)

    Function that turns a sample/mle/map from Turing into a ComponentVector.

    Example usage: 
    map_estimate = optimize(model, MAP())
    convert_turingoutput(Restricted(), map_estimate)
"""
function convert_turingoutput(::Restricted, optimised_model, p)  # function is not yet adapted to p
    θ =  optimised_model.values
    if p.DIM_COVARIATES==3
        out = ComponentArray(γ12=[θ[Symbol("γup[1]")], θ[Symbol("γup[2]")], θ[Symbol("γup[3]")]],
                        γ21=[θ[Symbol("γdown[1]")], θ[Symbol("γdown[2]")], θ[Symbol("γdown[3]")]],
                        γ23=[θ[Symbol("γup[1]")], θ[Symbol("γup[2]")], θ[Symbol("γup[3]")]],
                        γ32=[θ[Symbol("γdown[1]")], θ[Symbol("γdown[2]")], θ[Symbol("γdown[3]")]],
                        Z1=[θ[Symbol("Z0[1]")], θ[Symbol("Z0[2]")], θ[Symbol("Z0[3]")]],
                        Z2=[θ[Symbol("Z0[1]")], θ[Symbol("Z0[2]")], θ[Symbol("Z0[3]")]],
                        Z3=[θ[Symbol("Z0[1]")], θ[Symbol("Z0[2]")], θ[Symbol("Z0[3]")]],
                        Z4=[θ[Symbol("Z0[1]")], θ[Symbol("Z0[2]")], θ[Symbol("Z0[3]")]]
                        )
    elseif p.DIM_COVARIATES==4
        out = ComponentArray(γ12=[θ[Symbol("γup[1]")], θ[Symbol("γup[2]")], θ[Symbol("γup[3]")], θ[Symbol("γup[4]")]],
                        γ21=[θ[Symbol("γdown[1]")], θ[Symbol("γdown[2]")], θ[Symbol("γdown[3]")], θ[Symbol("γdown[4]")]],
                        γ23=[θ[Symbol("γup[1]")], θ[Symbol("γup[2]")], θ[Symbol("γup[3]")], θ[Symbol("γup[4]")]],
                        γ32=[θ[Symbol("γdown[1]")], θ[Symbol("γdown[2]")], θ[Symbol("γdown[3]")], θ[Symbol("γdown[4]")]],
                        Z1=[θ[Symbol("Z0[1]")], θ[Symbol("Z0[2]")], θ[Symbol("Z0[3]")]],
                        Z2=[θ[Symbol("Z0[1]")], θ[Symbol("Z0[2]")], θ[Symbol("Z0[3]")]],
                        Z3=[θ[Symbol("Z0[1]")], θ[Symbol("Z0[2]")], θ[Symbol("Z0[3]")]],
                        Z4=[θ[Symbol("Z0[1]")], θ[Symbol("Z0[2]")], θ[Symbol("Z0[3]")]]
                        )
    else 
        @error("convert_turingoutput has not been implemented for this number of covariates")
    end
    out 
end


function convert_turingoutput(::Unrestricted, optimised_model, p)  # function is not yet adapted to p
    θ =  optimised_model.values
    if p.DIM_COVARIATES==3
        out = ComponentArray(γ12=[θ[Symbol("γup[1]")], θ[Symbol("γup[2]")], θ[Symbol("γup[3]")]],
                    γ21=[θ[Symbol("γdown[1]")], θ[Symbol("γdown[2]")], θ[Symbol("γdown[3]")]],
                    γ23=[θ[Symbol("γup[1]")], θ[Symbol("γup[2]")], θ[Symbol("γup[3]")]],
                    γ32=[θ[Symbol("γdown[1]")], θ[Symbol("γdown[2]")], θ[Symbol("γdown[3]")]],
                      Z1=[θ[Symbol("Z1[1]")], θ[Symbol("Z1[2]")], θ[Symbol("Z1[3]")]],
                      Z2=[θ[Symbol("Z2[1]")], θ[Symbol("Z2[2]")], θ[Symbol("Z2[3]")]],
                      Z3=[θ[Symbol("Z3[1]")], θ[Symbol("Z3[2]")], θ[Symbol("Z3[3]")]],
                      Z4=[θ[Symbol("Z4[1]")], θ[Symbol("Z4[2]")], θ[Symbol("Z4[3]")]]
                      )
    elseif p.DIM_COVARIATES==4
        out = ComponentArray(γ12=[θ[Symbol("γup[1]")], θ[Symbol("γup[2]")], θ[Symbol("γup[3]")], θ[Symbol("γup[4]")]],
        γ21=[θ[Symbol("γdown[1]")], θ[Symbol("γdown[2]")], θ[Symbol("γdown[3]")], θ[Symbol("γdown[4]")]],
        γ23=[θ[Symbol("γup[1]")], θ[Symbol("γup[2]")], θ[Symbol("γup[3]")], θ[Symbol("γup[4]")]],
        γ32=[θ[Symbol("γdown[1]")], θ[Symbol("γdown[2]")], θ[Symbol("γdown[3]")], θ[Symbol("γdown[4]")]],
        Z1=[θ[Symbol("Z1[1]")], θ[Symbol("Z1[2]")], θ[Symbol("Z1[3]")]],
        Z2=[θ[Symbol("Z2[1]")], θ[Symbol("Z2[2]")], θ[Symbol("Z2[3]")]],
        Z3=[θ[Symbol("Z3[1]")], θ[Symbol("Z3[2]")], θ[Symbol("Z3[3]")]],
        Z4=[θ[Symbol("Z4[1]")], θ[Symbol("Z4[2]")], θ[Symbol("Z4[3]")]]
        )
    else 
        @error("convert_turingoutput has not been implemented for this number of covariates")
    end
    out 
end










# @time map_estimate = optimize(model, MAP());

# m = map_estimate.values
# m.dicts
# m.arraym


