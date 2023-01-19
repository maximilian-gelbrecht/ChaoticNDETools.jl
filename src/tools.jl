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
    forecast_lengths(model, valid::NODEDataloader,N_t=300; λmax=0, mode="norm")

Returns the forecast lengths of predictions on a NODEDataloader set (should be valid or test set) given a `(t, u0) -> prediction` function. `N_t` is the length of each forecast, has to be larger than the expected forecast length. If a `λmax` is given, the results are scaled with it (and `dt``)
"""
function forecast_lengths(model, t::AbstractArray{T,1}, data::AbstractArray{T,S}, N_t::Integer=300; λ_max=0, mode="norm", threshold=0.4) where {T,S}

    N = length(t) - N_t
    @assert N >= 1 

    forecasts = zeros(N)

    if typeof(t) <: AbstractRange 
        dt = step(t)
    else 
        dt = t[2] - t[1]
    end
    
    for i=1:N 
        δ = forecast_δ(model((t[i:i+N_t], data[..,i:i])), data[..,i:i+N_t], mode)
        δ = δ[:] # return a 1x1...xN_t array, so we flatten it here
        first_ind = findfirst(δ .> threshold) 

        if isnothing(first_ind) # in case no element of δ is larger than the threshold
            @warn "Prediction error smaller than threshold for the full predicted range, consider increasing N_t"
            first_ind = N_t + 1
        end 

        if λ_max == 0
            forecasts[i] = first_ind 
        else 
            forecasts[i] = first_ind * dt * λ_max
        end
    end 
    
    return forecasts
end
forecast_lengths(model, valid, N_t::Integer=300; kwargs...)  = forecast_lengths(model, valid.t, valid.data, N_t; kwargs...) 

"""
    average_forecast_length(predict, valid::NODEDataloader,N_t=300; λmax=0, mode="norm")

Returns the average forecast length on a NODEDataloader set (should be valid or test set) given a `(t, u0) -> prediction` function. `N_t` is the length of each forecast, has to be larger than the expected forecast length. If a `λmax` is given, the results are scaled with it (and `dt``)
"""
average_forecast_length(args...; kwargs...)  = mean(forecast_lengths(args...; kwargs...))

"""
    forecast_horizon_datagen(predict, ground_truth_predict, u0, t, N_t=300, λ_max=0, mode="norm", threshold=0.4)

Returns the forecast horizon, but also generates the ground truth data with a `(t,u0) -> data` function. Use this in cases in which the forecast horizon might be longer than the valid set
"""
function forecast_horizon_datagen(predict, ground_truth_predict, u0, t, λ_max=0, mode="norm", threshold=0.4)

    if typeof(t) <: AbstractRange 
        dt = step(t)
    else 
        dt = t[2] - t[1]
    end

    ground_truth = ground_truth_predict(t, u0) 
    prediction = predict(t, u0)

    δ = ChaoticNDETools.forecast_δ(prediction, ground_truth, mode)

    first_ind = findfirst(δ .> threshold)
    if isnothing(first_ind) # in case no element of δ is larger than the threshold
        @warn "Prediction error smaller than threshold for the full predicted range, consider increasing N_t"
        first_ind = length(t) + 1
    end 

    if λ_max == 0
        return first_ind 
    else 
        return first_ind * dt * λ_max
    end
end 






