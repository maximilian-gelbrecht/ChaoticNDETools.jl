var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = ChaoticNDETools","category":"page"},{"location":"#ChaoticNDETools","page":"Home","title":"ChaoticNDETools","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for ChaoticNDETools.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [ChaoticNDETools]","category":"page"},{"location":"#ChaoticNDETools.NablaSkipConnection","page":"Home","title":"ChaoticNDETools.NablaSkipConnection","text":"NablaSkipConnection\n\nWith the input x, passes w * ( * x) + (1 - w) * x , where winmathbbR win 01. Here Nabla is a finite difference derivative matrix\n\n\n\n\n\n","category":"type"},{"location":"#ChaoticNDETools.ParSkipConnection","page":"Home","title":"ChaoticNDETools.ParSkipConnection","text":"ParSkipConnection(layer, connection)\n\nA version of SkipConnection that is paramaterized with one single parameter, thus the output is connection(w .* layer(input), input)\n\n\n\n\n\n","category":"type"},{"location":"#ChaoticNDETools.average_forecast_length","page":"Home","title":"ChaoticNDETools.average_forecast_length","text":"average_forecast_length(predict, valid::NODEDataloader,N_t=300; λmax=0, mode=\"norm\")\n\nReturns the average forecast length on a NODEDataloader set (should be valid or test set) given a (t, u0) -> prediction function. N_t is the length of each forecast, has to be larger than the expected forecast length. If a λmax is given, the results are scaled with it (and dt`)\n\n\n\n\n\n","category":"function"},{"location":"#ChaoticNDETools.forecast_δ-Union{Tuple{N}, Tuple{T}, Tuple{AbstractArray{T, N}, AbstractArray{T, N}}, Tuple{AbstractArray{T, N}, AbstractArray{T, N}, String}} where {T, N}","page":"Home","title":"ChaoticNDETools.forecast_δ","text":"forecast_δ(prediction::AbstractArray{T,N}, truth::AbstractArray{T,N}, mode::String=\"both\") where {T,N}\n\nAssumes that the last dimension of the input arrays is the time dimension and N_t long. Returns an N_t long array, judging how accurate the prediction is. \n\nSupported modes: \n\n\"mean\": mean between the arrays\n\"maximum\": maximum norm \n\"norm\": normalized, similar to the metric used in Pathak et al \n\n\n\n\n\n","category":"method"},{"location":"#ChaoticNDETools.gpuoff-Tuple{}","page":"Home","title":"ChaoticNDETools.gpuoff","text":"gpuoff()\n\nManually toggle GPU use off\n\n\n\n\n\n","category":"method"},{"location":"#ChaoticNDETools.gpuon-Tuple{}","page":"Home","title":"ChaoticNDETools.gpuon","text":"gpuon()\n\nManually toggle GPU use on (if available)\n\n\n\n\n\n","category":"method"},{"location":"#ChaoticNDETools.∂x_PBC-Union{Tuple{T}, Tuple{Integer, T}} where T","page":"Home","title":"ChaoticNDETools.∂x_PBC","text":"∂x_PBC(n::Integer, dx::T)\n\n2nd order central finite difference matrix for 1d domains with periodic boundary conditions\n\n\n\n\n\n","category":"method"},{"location":"#ChaoticNDETools.∂x²_PBC-Union{Tuple{T}, Tuple{Integer, T}} where T","page":"Home","title":"ChaoticNDETools.∂x²_PBC","text":"∂x²_PBC(n::Integer, dx::T)\n\n2nd order central finite difference matrix of the second derivative for 1d domains with periodic boundary conditions\n\n\n\n\n\n","category":"method"},{"location":"#ChaoticNDETools.∂x⁴_PBC-Union{Tuple{T}, Tuple{Integer, T}} where T","page":"Home","title":"ChaoticNDETools.∂x⁴_PBC","text":"∂x⁴_PBC(n::Integer, dx::T)\n\n4th derivative finite difference matrix with periodic boundary conditions\n\n\n\n\n\n","category":"method"}]
}
