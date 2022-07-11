# A script for fitting a neural de to a lotka volterra system, not present in the NJP paper, but kept here as presented in the university course

import Pkg
Pkg.activate("scripts") # change this to "." incase your "scripts" is already your working directory

using Flux, DiffEqFlux, CUDA, OrdinaryDiffEq, BenchmarkTools, JLD2, Plots, Random

# not registered packages, add them manually (see comment in the Readme.md)
using ChaoticNDETools, NODEData

Random.seed!(12345)
#=
this script can also be called from the command line with extra arguments (e.g. by a batch system such as SLURM), otherwise default values are used.

the extra arguments are
* 1) GPU, 1==true, 0==false
* 2) SAVE_NAME, base name of the saved files
* 3) N_EPOCHS
* 4) N_t, length of training dataset
* 5) τ_max
* 6) RELOAD, reload data
=#
begin
    parse_ARGS(i, ARGS, default) = length(ARGS) >= i ? parse(Int, ARGS[i]) : default

    GPU = parse_ARGS(1, ARGS, 0)
    GPU = GPU == 1 ? true : false

    SAVE_NAME = length(ARGS) >= 2 ? ARGS[2] : "local-test"
    N_epochs = parse_ARGS(3, ARGS, 30)

    N_t = parse_ARGS(4, ARGS, 250) 
    τ_max = parse_ARGS(5, ARGS, 3) 
    RELOAD = parse_ARGS(6, ARGS, 0) 
    RELOAD = RELOAD == 1 ? true : false

    N_WEIGHTS = 10
    dt = 0.1
    t_transient = 0.
    N_t_train = N_t
    N_t_valid = 8000
    N_t = N_t_train + N_t_valid + 1000
end 


begin 
    function lotka_volterra(x,p,t)
        α, β, γ, δ = p 
        [α*x[1] - β*x[1]*x[2], -γ*x[2] + δ*x[1]*x[2]]
    end
    
    α = 1.3
    β = 0.9
    γ = 0.8
    δ = 1.8
    p = [α, β, γ, δ] 
    tspan = (0.,50.)
    
    x0 = [0.44249296, 4.6280594] 
    
    prob = ODEProblem(lotka_volterra, x0, tspan, p) 
    sol = solve(prob, Tsit5(), saveat=dt)
end 

train, valid = NODEDataloader(sol, 10; dt=dt, valid_set=0.8)

nn = Chain(Dense(2, N_WEIGHTS, swish), Dense(N_WEIGHTS, N_WEIGHTS, swish), Dense(N_WEIGHTS, N_WEIGHTS, swish), Dense(N_WEIGHTS, 2)) |> gpu
p, re_nn = Flux.destructure(nn)

neural_ode(u, p, t) = re_nn(p)(u)

node_prob = ODEProblem(neural_ode, x0, (Float32(0.),Float32(dt)), p)

predict(t, u0; reltol=1e-5) = DeviceArray(solve(remake(node_prob; tspan=(t[1],t[end]),u0=u0, p=p), Tsit5(), dt=dt, saveat = t, reltol=reltol, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP())))
loss(t, u0) = sum(abs2, predict(t, view(u0,:,1)) - u0)
loss(train[1]...)

function plot_node()
    plt = plot(sol.t, Array(predict(sol.t,x0)'), label="Neural ODE")
    plot!(plt, sol.t, Array(sol)', label="Training Data")
    plot!(plt, [train[1][1][1],train[end][1][end]],zeros(2),label="Length of Training Set", linewidth=5, ylims=[0,5])
    display(plt)
end
plot_node()

opt = Flux.AdamW(1f-3)
# setup the training loop 

Flux.train!(loss, Flux.params(p), train, opt)
plot_node()

TRAIN = true
if TRAIN 
    println("starting training...")

    for i_e = 1:300
        Flux.train!(loss, Flux.params(p), train, opt)
        plot_node()

        if (i_e % 25) == 0  # reduce the learning rate every 25 epochs
            opt[1].eta /= 2
        end
    end
end

