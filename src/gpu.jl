using Adapt
using CUDA.CUSPARSE, SparseArrays

DeviceArray(x) = cuda_used[] ? adapt(CuArray,x) : adapt(Array,x)
DeviceSparseArray(x) = cuda_used[] ? CUDA.CUSPARSE.CuSparseMatrixCSC(x) : sparse(x)

"""
    gpuon()

Manually toggle GPU use on (if available)
"""
function gpuon() # manually toggle GPU use on and off
    cuda_used[] = CUDA.functional()
end

"""
    gpuoff()

Manually toggle GPU use off
"""
function gpuoff()
    cuda_used[] = false
end