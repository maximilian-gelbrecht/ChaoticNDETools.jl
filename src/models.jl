
using OrdinaryDiffEq, SciMLSensitivity, Optimisers

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

# Constructors 

* `ChaoticNDE(prob; alg=Tsit5(), kwargs...)`
* `ChaoticNDE(model::ChaoticNDE; alg=model.alg, kwargs...)` remake the model with different kwargs and solvers

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

ChaoticNDE(m::ChaoticNDE; alg=m.alg, kwargs...) = ChaoticNDE(m.p, m.prob, alg, kwargs, m.device)

Flux.@layer ChaoticNDE
Optimisers.trainable(m::ChaoticNDE) = (p=m.p,)

function (m::ChaoticNDE{P,R,A,K,D})(X,p=m.p) where {P,R,A,K,D}
    (t, x) = X 
    DeviceArray(m.device, solve(remake(m.prob; tspan=(t[1],t[end]),u0=selectdim(x, ndims(x), 1),p=p), m.alg; saveat=t, m.kwargs...))
end

"""
    set_params!(m::ChaoticNDE{P,R,A,K,D}, p::P) where {P,R,A,K,D}

Sets `p` as the parameters of model `m`. Used e.g. when loading parameter from trained models. 
"""
function set_params!(m::ChaoticNDE{P,R,A,K,D}, p::P) where {P,R,A,K,D}
    @assert size(m.p) == size(p) "Input parameter has to have the same size as the existing parameter vector of the model."
    m.p .= p
    return nothing 
end 

"""
    trajectory(m::ChaoticNDE, tspan, u0; alg=nothing, kwargs...)

Solves the model `m`, returns a SciML solution object. All keyword arguments are directly forwarded to the differential equation solver.
"""
function trajectory(m::ChaoticNDE, tspan, u0; alg=nothing, kwargs...)
    if isnothing(alg)
        alg = m.alg 
    end 

    solve(remake(m.prob; tspan=tspan, u0=u0, p=m.p), alg; kwargs...)
end 