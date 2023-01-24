using Adapt
using CUDA.CUSPARSE, SparseArrays

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

abstract type AbstractDevice end 
abstract type AbstractGPUDevice <: AbstractDevice end 
struct CPUDevice <: AbstractDevice end 
struct CUDADevice <: AbstractGPUDevice end 

"""
    Device(; gpu::Union{Nothing, Bool}=nothing)   

Initializes the device that is used. Returns either `CPUDevice` or `CUDADevice`. If no `gpu` keyword argument is given, it determines automatically if a GPU is available.
"""
function Device(; gpu::Union{Nothing, Bool}=nothing)   
    if isnothing(gpu)
        dev = cuda_used[] ? CUDADevice() : CPUDevice()
    else 
        dev = gpu ? CUDADevice() : CPUDevice()
    end 
    return dev 
end 

DeviceArray(x) = cuda_used[] ? adapt(CuArray,x) : adapt(Array,x)
DeviceSparseArray(x) = cuda_used[] ? CUDA.CUSPARSE.CuSparseMatrixCSC(x) : sparse(x)

DeviceArray(dev::CUDADevice, x) = adapt(CuArray, x)
DeviceArray(dev::CPUDevice, x) = adapt(Array, x)
DeviceSparseArray(dev::CUDADevice) = CUDA.CUSPARSE.CuSparseMatrixCSC(x) 
DeviceSparseArray(dev::CPUDevice) = sparse(x)
