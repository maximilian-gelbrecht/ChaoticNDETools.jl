
using OrdinaryDiffEq, DiffEqSensitivity

abstract type AbstractChaoticNDEModel end 

"""
    ChaoticNDE{P,R,A,K} <: AbstractChaoticNDEModel

Model for setting up and training Chaotic Neural Differential Equations.

# Fields:

* `p` parameter vector 
* `prob` DEProblem 
* `alg` Algorithm to use for the `solve` command 
* `kwargs` any additional keyword arguments that should be handed over (e.g. `sensealg`)
* `device` the device the model is running on, either `CPUDevice` or `CUDADevice`, used for dispatiching if `Arrays` or `CuArrays` are used

# Constructor 

`ChaoticNDE(prob; alg=Tsit5(), kwargs...)`

# Input / call 

An instance of the model is called with a trajectory pair `(t,x)` in `t` are the timesteps that NDE is integrated for and `x` is a trajectory `N x ... x N_t` in which `x[:, ... , 1]` is taken as the initial condition. 
"""
struct ChaoticNDE{P,R,A,K,D} <: AbstractChaoticNDEModel
    p::P 
    prob::R 
    alg::A
    kwargs::K
    device::D
end 

function ChaoticNDE(prob; alg=Tsit5(), gpu=nothing, kwargs...)
    p = prob.p 
    device = Device(gpu=gpu)
    ChaoticNDE{typeof(p), typeof(prob), typeof(alg), typeof(kwargs), typeof(device)}(p, prob, alg, kwargs, device)
end 

Flux.@functor ChaoticNDE
Flux.trainable(m::ChaoticNDE) = (p=m.p,)

function (m::ChaoticNDE{P,R,A,K,D})(X,p=m.p) where {P,R,A,K,D}
    (t, x) = X 
    DeviceArray(m.device, solve(remake(m.prob; tspan=(t[1],t[end]),u0=selectdim(x, ndims(x), 1),p=p), m.alg; saveat=t, m.kwargs...))
end


 