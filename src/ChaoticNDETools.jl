module ChaoticNDETools

using CUDA 

global const cuda_used = Ref(false)

function __init__() # automatically called at runtime to set cuda_used
    cuda_used[] = CUDA.functional()
end

include("gpu.jl")
include("layers.jl")
include("finite_differences.jl")
include("tools.jl")

export ParSkipConnection, NablaSkipConnection
export DeviceArray, DeviceSparseArray

end
