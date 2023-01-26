# A script for fitting a neural de to a lotka volterra system, not present in the NJP paper, but kept here as presented in the university course

import Pkg
Pkg.activate("scripts") # change this to "." incase your "scripts" is already your working directory

using Flux, CUDA, OrdinaryDiffEq, BenchmarkTools, JLD2, Plots, Random, DiffEqSensitivity

# not registered packages, add them manually (see comment in the Readme.md)
using ChaoticNDETools, NODEData

Random.seed!(1234)
#=
this script can also be called from the command line with extra arguments (e.g. by a batch system such as SLURM), otherwise default values are used.

the extra arguments are
* 1) SAVE_NAME, base name of the saved files
* 2) N_EPOCHS
* 3) N_t, length of training dataset
* 4) τ_max
=#
begin
    parse_ARGS(i, ARGS, default) = length(ARGS) >= i ? parse(Int, ARGS[i]) : default

    SAVE_NAME = length(ARGS) >= 1 ? ARGS[1] : "local-test"
    N_epochs = parse_ARGS(2, ARGS, 30)

    N_t = parse_ARGS(3, ARGS, 250) 
    τ_max = parse_ARGS(4, ARGS, 3) 
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
basic_tgrad(u,p,t) = zero(u)
nnf = ODEFunction{false}(neural_ode,tgrad=basic_tgrad)
node_prob = ODEProblem(nnf, x0, (Float32(0.),Float32(dt)), p)

model = ChaoticNDE(node_prob)
model(train[1])

loss(x, y) = sum(abs2, x - y)
loss(model(train[1]), train[1][2]) 

function plot_node()
    plt = plot(sol.t, Array(model((sol.t,train[1][2])))', label="Neural ODE")
    plot!(plt, sol.t, Array(sol)', label="Training Data")
    plot!(plt, [train[1][1][1],train[end][1][end]],zeros(2),label="Length of Training Set", linewidth=5, ylims=[0,5])
    display(plt)
end
plot_node()

η = 1f-3
opt = Flux.AdamW(η)
opt_state = Flux.setup(opt, model)

# pre-compile adjoint code 
g = gradient(model) do m
    result = m(train[1])
    loss(result, train[1][2])
end

TRAIN = false
if TRAIN 
    println("starting training...")

    for i_e = 1:400

        Flux.train!(model, train, opt_state) do m, t, x
            result = m((t,x))
            loss(result, x)
        end 

        plot_node()

        if (i_e % 30) == 0  # reduce the learning rate every 30 epochs
            η /= 2
            Flux.adjust!(opt_state, η)
        end
    end
end

