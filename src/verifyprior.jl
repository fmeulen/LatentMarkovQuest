scaledandshifted_logistic(x) = 2.0logistic(.75*x) - 1.0 # function that maps [0,∞) to [0,1) 
#scaledandshifted_logistic(x) = 2.0logistic(x) - 1.0
mapZtoλ(x) = scaledandshifted_logistic.(cumsum(x))

s = [mapZtoλ(rand(Exponential(1.0),3)) for _ in 1:10_000]
#s = [mapZtoλ(rand(Gamma(2.0,1.0),3)) for _ in 1:10_000]

histogram(getindex.(s,1), alpha=0.2, label="λ_j(1)",
legend = :outertopright, size = (800, 300), normalize=true)
histogram!(getindex.(s,2), alpha=0.2, label="λ_j(2)", normalize=true)
histogram!(getindex.(s,3), alpha=0.2, label="λ_j(3)", normalize=true)

mean(getindex.(s,1))
mean(getindex.(s,2))
mean(getindex.(s,3))

png("priorlambda.png")
pdf("priorlambda.png")