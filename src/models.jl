
using OrdinaryDiffEq, DiffEqSensitivity

abstract type AbstractChaoticNDEModel end 

struct ChaoticNDE{P,R,S,K} <: AbstractChaoticNDEModel
    p::P 
    prob::R 
    sensealg::S
    kwargs::K
end 

function ChaoticNDE(prob; sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()), kwargs...)
    p = prob.p 
    ChaoticNDE{typeof(p), typeof(prob), typeof(sensealg), typeof(kwargs)}(p, prob, sensealg, kwargs)
end 

Flux.@functor ChaoticNDE
Flux.trainable(m::ChaoticNDE) = (p=m.p,)

function (m::ChaoticNDE)(X,p=m.p)
    (t, x) = X 
    DeviceArray(solve(remake(m.prob; tspan=(t[1],t[end]),u0=x[:,1],p=p), Tsit5(), saveat=t, sensealg=m.sensealg, m.kwargs...))
end


