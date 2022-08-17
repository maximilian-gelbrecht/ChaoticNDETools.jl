using EllipsisNotation

"""
    forecast_δ(prediction::AbstractArray{T,N}, truth::AbstractArray{T,N}, mode::String="both") where {T,N}

Assumes that the last dimension of the input arrays is the time dimension and `N_t` long. Returns an `N_t` long array, judging how accurate the prediction is. 

Supported modes: 
* `"mean"`: mean between the arrays
* `"maximum"`: maximum norm 
* `"norm"`: normalized, similar to the metric used in Pathak et al 
"""
function forecast_δ(prediction::AbstractArray{T,N}, truth::AbstractArray{T,N}, mode::String="norm") where {T,N}

    if !(mode in ["mean","largest","both","norm"])
        error("mode has to be either 'mean', 'largest' or 'both', 'norm'.")
    end

    δ = abs.(prediction .- truth)

    if mode == "mean"
        return mean(δ, dims=1:(N-1))
    elseif mode == "maximum"
        return maximum(δ, dims=1:(N-1))
    elseif mode == "norm"
        return sqrt.(sum((prediction .- truth).^2, dims=(1:(N-1))))./sqrt(mean(sum(abs2, truth, dims=(1:(N-1)))))
    else
        return (mean(δ, dims=1:(N-1)), maximum(δ, dims=1))
    end
end

"""
    average_forecast_length(predict, valid::NODEDataloader,N_t=300; λmax=0, mode="norm")

Returns the average forecast length on a NODEDataloader set (should be valid or test set) given a `(t, u0) -> prediction` function. `N_t` is the length of each forecast, has to be larger than the expected forecast length. If a `λmax` is given, the results are scaled with it (and `dt``)
"""
function average_forecast_length(predict, t::AbstractArray{T,1}, data::AbstractArray{T,S}, N_t=300; λ_max=0, mode="norm") where {T,S}

    N = length(t) - N_t
    @assert N >= 1 

    forecasts = zeros(N)

    if t <: AbstractRange 
        dt = step(t)
    else 
        dt = t[2] - t[1]
    end
    
    for i=1:N 

        δ = forecast_δ(predict(t[i:i+N_t], data[..,i]), data[..,i:i+N_t], mode)
        δ = δ[:] # return a 1x1...xN_t array, so we flatten it here
        if λ_max == 0
            forecasts[i] = findfirst(δ .> 0.4) 
        else 
            forecasts[i] = findfirst(δ .> 0.4) * dt * λ_max
        end
    end 
    return mean(forecasts)
end

average_forecast_length(predict, valid, N_t=300; kwargs...)  = average_forecast_length(predict, valid.t, valid.data, N_t; kwargs...) 
