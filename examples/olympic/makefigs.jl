### obsolete, better graphs in R

using DataFrames
using DataFramesMeta
using Plots
using Statistics
using TidierPlots

# Define the theme
#default(theme(:bw))

# Add the type column
types = ["sigma"; repeat(["from 1 to 2"], 4); repeat(["from 2 to 3"], 4); repeat(["from 2 to 1"], 4); repeat(["from 3 to 2"], 4); 
         repeat(["Z1"], 3); repeat(["Z2"], 3); repeat(["Z3"], 3); repeat(["Z4"], 3)]

posterior_summary.type = types[1:nrow(posterior_summary)]

# Convert parameters to string
posterior_summary.parameters = string.(posterior_summary.parameters)

# Add lower and upper columns
@chain posterior_summary begin
    @transform! :lower = :mean .- 2 * :std
    @transform! :upper = :mean .+ 2 * :std
end


# Plotting
p = plot(
    posterior_summary.mean, posterior_summary.parameters, 
    seriestype = :scatter, 
    xerr = (posterior_summary.mean .- posterior_summary.lower, posterior_summary.upper .- posterior_summary.mean),
    label = "",
    xlabel = "parameters",
    ylabel = "mean",
    legend = :outerright, 
    yticks = (1:nrow(posterior_summary), posterior_summary.parameters),
    margin = 5Plots.mm,
    size= (400, 1000)
)


# Adding color by type
for type in unique(posterior_summary.type)
    data_subset = posterior_summary[posterior_summary.type .== type, :]
    scatter!(data_subset.mean, data_subset.parameters,
             xerr = (data_subset.mean .- data_subset.lower, data_subset.upper .- data_subset.mean), 
             label = type)
end


# Save the plot
savefig(p, "figs/estimates.pdf")


#############


covariates = ["intercept", "sport", "strength", "competition"]
# Filter rows where parameters contain "γ"
filtered_posterior_summary = @chain posterior_summary begin
    @subset(occursin.("γ",:parameters))
    @transform! :covariate = repeat(covariates, inner=4)
    @transform! :lower = :mean .- 2 * :std
    @transform! :upper = :mean .+ 2 * :std
end

# Plotting
p = plot(
    filtered_posterior_summary.parameters, filtered_posterior_summary.mean,
    seriestype = :scatter,
    xerr = (filtered_posterior_summary.mean .- filtered_posterior_summary.lower, filtered_posterior_summary.upper .- filtered_posterior_summary.mean),
    xlabel = "Parameter",
    ylabel = "",
    legend = :bottom,
    yticks = (1:nrow(filtered_posterior_summary), filtered_posterior_summary.parameters),
    margin = 5Plots.mm,
    size = (800, 600)
)

# Adding horizontal line at y=0
hline!(p, [0], color=:black, linestyle=:dash)

# Adding color by covariate and facets by type
for covariate in unique(filtered_posterior_summary.covariate)
    data_subset = filtered_posterior_summary[filtered_posterior_summary.covariate .== covariate, :]
    scatter!(data_subset.parameters, data_subset.mean,
             xerr = (data_subset.mean .- data_subset.lower, data_subset.upper .- data_subset.mean),
             label = covariate)
end

# Facet by type
for t in unique(filtered_posterior_summary.type)
    data_subset = filtered_posterior_summary[filtered_posterior_summary.type .== t, :]
    scatter!(data_subset.parameters, data_subset.mean,
             xerr = (data_subset.mean .- data_subset.lower, data_subset.upper .- data_subset.mean),
             label = t)
end

# Save the plot
savefig(p, "figs/estimates_faceted.pdf")


#####


# Covariates
covariates = ["intercept", "sport", "strength", "competition"]

# Filter rows where parameters contain "γ"
filtered_posterior_summary = @chain posterior_summary begin
    @subset(occursin.("γ", :parameters))
    @transform! :covariate = repeat(covariates, inner=4)
    @transform! :lower = :mean .- 2 * :std
    @transform! :upper = :mean .+ 2 * :std
end

# Plotting
p = @tidy begin
    scatter(filtered_posterior_summary,
        x=:mean,
        y=:parameters,
        color=:covariate,
        xerror=:std,
        label="",
        xlabel="Mean",
        ylabel="Parameter",
        legend=:bottom
    )
    hline!([0], color=:black, linestyle=:dash)
end

# Save the plot
savefig(p, "figs/estimates_faceted.pdf")


############# with TidierPlots ################3

covariates = ["intercept", "sport","strength","competition"]
types = vcat(["σ²", fill("from 1 to 2",4), fill("from 2 to 3",4), 
            fill("from 2 to 1",4), fill("from 3 to 2",4), 
            fill("Z1",3), fill("Z2",3), fill("Z3",3), fill("Z4",3)]...)

            types = vcat([fill("from 1 to 2",4), fill("from 2 to 3",4), 
            fill("from 2 to 1",4), fill("from 3 to 2",4)]...)


# Filter rows where parameters contain "γ"
filtered_posterior_summary = @chain posterior_summary begin
    @subset(occursin.("γ", :parameters))
    @transform! :covariate = repeat(covariates, outer=4)
    @transform! :lower = :mean .- 2 * :std
    @transform! :upper = :mean .+ 2 * :std
    @transform! :type = types
end


ggplot(data=filtered_posterior_summary) +
  geom_point(@aes(x= parameters, y= mean, colour= covariate)) +
  geom_errorbar(@aes(x = parameters, ymin= lower, ymax= upper)) +
  labs(x="parameter", y="") + geom_hline(yintercept=0) +  
  facet_grid(.~type,scales="free_y")
  
  coord_flip() + 
  theme(legend.position = "bottom")
