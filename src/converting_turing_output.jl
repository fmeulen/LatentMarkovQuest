mapallZtoλ(θ) = hcat(mapZtoλ(θ.Z1), mapZtoλ(θ.Z2), mapZtoλ(θ.Z3), mapZtoλ(θ.Z4))


function getpars(θs, names_par; restricted=false)
    @warn "We assume here 4 questions (hence Z1,...,Z4). Adapt if different"

    σ²_ = θs[occursin.("σ²", names_par)]

    γ12_ = θs[occursin.("γ12", names_par)]
    γ23_ = θs[occursin.("γ23", names_par)]
    γ21_ = θs[occursin.("γ21", names_par)]
    γ32_ = θs[occursin.("γ32", names_par)]
    if restricted 
        Z1_ = θs[occursin.("Z0", names_par)]
        Z2_ = θs[occursin.("Z0", names_par)]
        Z3_ = θs[occursin.("Z0", names_par)]
        Z4_ = θs[occursin.("Z0", names_par)]
    else
        Z1_ = θs[occursin.("Z1", names_par)]
        Z2_ = θs[occursin.("Z2", names_par)]
        Z3_ = θs[occursin.("Z3", names_par)]
        Z4_ = θs[occursin.("Z4", names_par)]
    end
    ComponentArray(σ²=σ²_,   γ12=γ12_, γ23=γ23_, γ21=γ21_, γ32=γ32_, Z1=Z1_, Z2=Z2_, Z3=Z3_, Z4=Z4_)
end   



# """
#     convert_turingoutput(optimised_model)

#     Function that turns a sample/mle/map from Turing into a ComponentVector.

#     Example usage: 
#     map_estimate = optimize(model, MAP())
#     convert_turingoutput(Restricted(), map_estimate)
# """
# function convert_turingoutput(::Restricted, optimised_model, p)  
#     θ =  optimised_model.values
#     if p.DIM_COVARIATES==3
#         out = ComponentArray(γ12=[θ[Symbol("γup[1]")], θ[Symbol("γup[2]")], θ[Symbol("γup[3]")]],
#                         γ21=[θ[Symbol("γdown[1]")], θ[Symbol("γdown[2]")], θ[Symbol("γdown[3]")]],
#                         γ23=[θ[Symbol("γup[1]")], θ[Symbol("γup[2]")], θ[Symbol("γup[3]")]],
#                         γ32=[θ[Symbol("γdown[1]")], θ[Symbol("γdown[2]")], θ[Symbol("γdown[3]")]],
#                         Z1=[θ[Symbol("Z0[1]")], θ[Symbol("Z0[2]")], θ[Symbol("Z0[3]")]],
#                         Z2=[θ[Symbol("Z0[1]")], θ[Symbol("Z0[2]")], θ[Symbol("Z0[3]")]],
#                         Z3=[θ[Symbol("Z0[1]")], θ[Symbol("Z0[2]")], θ[Symbol("Z0[3]")]],
#                         Z4=[θ[Symbol("Z0[1]")], θ[Symbol("Z0[2]")], θ[Symbol("Z0[3]")]]
#                         )
#     elseif p.DIM_COVARIATES==4
#         out = ComponentArray(γ12=[θ[Symbol("γup[1]")], θ[Symbol("γup[2]")], θ[Symbol("γup[3]")], θ[Symbol("γup[4]")]],
#                         γ21=[θ[Symbol("γdown[1]")], θ[Symbol("γdown[2]")], θ[Symbol("γdown[3]")], θ[Symbol("γdown[4]")]],
#                         γ23=[θ[Symbol("γup[1]")], θ[Symbol("γup[2]")], θ[Symbol("γup[3]")], θ[Symbol("γup[4]")]],
#                         γ32=[θ[Symbol("γdown[1]")], θ[Symbol("γdown[2]")], θ[Symbol("γdown[3]")], θ[Symbol("γdown[4]")]],
#                         Z1=[θ[Symbol("Z0[1]")], θ[Symbol("Z0[2]")], θ[Symbol("Z0[3]")]],
#                         Z2=[θ[Symbol("Z0[1]")], θ[Symbol("Z0[2]")], θ[Symbol("Z0[3]")]],
#                         Z3=[θ[Symbol("Z0[1]")], θ[Symbol("Z0[2]")], θ[Symbol("Z0[3]")]],
#                         Z4=[θ[Symbol("Z0[1]")], θ[Symbol("Z0[2]")], θ[Symbol("Z0[3]")]]
#                         )
#     elseif p.DIM_COVARIATES==2
#         out = ComponentArray(γ12=[θ[Symbol("γup[1]")], θ[Symbol("γup[2]")]],
#         γ21=[θ[Symbol("γdown[1]")], θ[Symbol("γdown[2]")]],
#         γ23=[θ[Symbol("γup[1]")], θ[Symbol("γup[2]")]],
#         γ32=[θ[Symbol("γdown[1]")], θ[Symbol("γdown[2]")]],
#         Z1=[θ[Symbol("Z0[1]")], θ[Symbol("Z0[2]")], θ[Symbol("Z0[3]")]],
#         Z2=[θ[Symbol("Z0[1]")], θ[Symbol("Z0[2]")], θ[Symbol("Z0[3]")]],
#         Z3=[θ[Symbol("Z0[1]")], θ[Symbol("Z0[2]")], θ[Symbol("Z0[3]")]],
#         Z4=[θ[Symbol("Z0[1]")], θ[Symbol("Z0[2]")], θ[Symbol("Z0[3]")]]
#         )
#     else 
#         @error("convert_turingoutput has not been implemented for this number of covariates")
#     end
#     out 
# end


# function convert_turingoutput(::Unrestricted, optimised_model, p)  
#     θ =  optimised_model.values
#     if p.DIM_COVARIATES==3
#         out = ComponentArray(γ12=[θ[Symbol("γup[1]")], θ[Symbol("γup[2]")], θ[Symbol("γup[3]")]],
#                     γ21=[θ[Symbol("γdown[1]")], θ[Symbol("γdown[2]")], θ[Symbol("γdown[3]")]],
#                     γ23=[θ[Symbol("γup[1]")], θ[Symbol("γup[2]")], θ[Symbol("γup[3]")]],
#                     γ32=[θ[Symbol("γdown[1]")], θ[Symbol("γdown[2]")], θ[Symbol("γdown[3]")]],
#                       Z1=[θ[Symbol("Z1[1]")], θ[Symbol("Z1[2]")], θ[Symbol("Z1[3]")]],
#                       Z2=[θ[Symbol("Z2[1]")], θ[Symbol("Z2[2]")], θ[Symbol("Z2[3]")]],
#                       Z3=[θ[Symbol("Z3[1]")], θ[Symbol("Z3[2]")], θ[Symbol("Z3[3]")]],
#                       Z4=[θ[Symbol("Z4[1]")], θ[Symbol("Z4[2]")], θ[Symbol("Z4[3]")]]
#                       )
#     elseif p.DIM_COVARIATES==4
#         out = ComponentArray(γ12=[θ[Symbol("γup[1]")], θ[Symbol("γup[2]")], θ[Symbol("γup[3]")], θ[Symbol("γup[4]")]],
#         γ21=[θ[Symbol("γdown[1]")], θ[Symbol("γdown[2]")], θ[Symbol("γdown[3]")], θ[Symbol("γdown[4]")]],
#         γ23=[θ[Symbol("γup[1]")], θ[Symbol("γup[2]")], θ[Symbol("γup[3]")], θ[Symbol("γup[4]")]],
#         γ32=[θ[Symbol("γdown[1]")], θ[Symbol("γdown[2]")], θ[Symbol("γdown[3]")], θ[Symbol("γdown[4]")]],
#         Z1=[θ[Symbol("Z1[1]")], θ[Symbol("Z1[2]")], θ[Symbol("Z1[3]")]],
#         Z2=[θ[Symbol("Z2[1]")], θ[Symbol("Z2[2]")], θ[Symbol("Z2[3]")]],
#         Z3=[θ[Symbol("Z3[1]")], θ[Symbol("Z3[2]")], θ[Symbol("Z3[3]")]],
#         Z4=[θ[Symbol("Z4[1]")], θ[Symbol("Z4[2]")], θ[Symbol("Z4[3]")]]
#         )
#     elseif p.DIM_COVARIATES==2
#         out = ComponentArray(γ12=[θ[Symbol("γup[1]")], θ[Symbol("γup[2]")]],
#         γ21=[θ[Symbol("γdown[1]")], θ[Symbol("γdown[2]")]],
#         γ23=[θ[Symbol("γup[1]")], θ[Symbol("γup[2]")]],
#         γ32=[θ[Symbol("γdown[1]")], θ[Symbol("γdown[2]")]],
#         Z1=[θ[Symbol("Z1[1]")], θ[Symbol("Z1[2]")], θ[Symbol("Z1[3]")]],
#         Z2=[θ[Symbol("Z2[1]")], θ[Symbol("Z2[2]")], θ[Symbol("Z2[3]")]],
#         Z3=[θ[Symbol("Z3[1]")], θ[Symbol("Z3[2]")], θ[Symbol("Z3[3]")]],
#         Z4=[θ[Symbol("Z4[1]")], θ[Symbol("Z4[2]")], θ[Symbol("Z4[3]")]]
#         )
#     else 
#         @error("convert_turingoutput has not been implemented for this number of covariates")
#     end
#     out 
# end










# # @time map_estimate = optimize(model, MAP());

# # m = map_estimate.values
# # m.dicts
# # m.arraym


