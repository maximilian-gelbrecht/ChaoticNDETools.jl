using Pkg
Pkg.activate("scripts")

using JLD2, SlurmHyperopt, DataFrames

@load "hyperopt.jld2" sho
merge_results!(sho)
@save "hyperopt.jld2" sho

pars = get_params(sho) # get all the parameters 
res = get_results(sho) # get only the results 

df = DataFrame(sho) # return the pars and results as a DataFrame

@save "hyperopt-df.jld2" df
# your evaluation here 