# A script for fitting a neural de to a lotka volterra system, not present in the NJP paper, but kept here as presented in the university course

import Pkg
Pkg.activate("scripts") # change this to "." incase your "scripts" is already your working directory

using Flux, DiffEqFlux, CUDA, OrdinaryDiffEq, BenchmarkTools, JLD2, Plots, Random

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
    N_epochs = parse_ARGS(2, ARGS, 40)

    N_t = parse_ARGS(3, ARGS, 500) 
    τ_max = parse_ARGS(4, ARGS, 2) 

    N_WEIGHTS = 16
    dt = 0.1
    t_transient = 100.
    N_t_train = N_t
    N_t_valid = N_t_train*3
    N_t = N_t_train + N_t_valid
end 

begin 
    function lorenz63!(du,u,p,t)
        X,Y,Z = u 
        σ,r,b = p 
    
        du[1] = -σ * X + σ * Y
        du[2] = - X*Z + r*X - Y 
        du[3] = X*Y - b*Z
    end 
    
    σ, r, b = 10., 28., 8/3.
    p = [σ, r, b]
    u0 = rand(3)
    tspan = (0f0, Float32(t_transient + N_t * dt))

    prob = ODEProblem(lorenz63!, u0, tspan, p)
    
    sol = solve(prob, Tsit5(), saveat=t_transient:dt:t_transient + N_t * dt)
end 

begin 
    t_train = t_transient:dt:t_transient+N_t_train*dt
    data_train = DeviceArray(sol(t_train))

    t_valid = t_transient+N_t_train*dt:dt:t_transient+N_t_train*dt+N_t_valid*dt
    data_valid = DeviceArray(sol(t_valid))

    train = NODEDataloader(Float32.(data_train), t_train, 2)
    valid = NODEDataloader(Float32.(data_valid), t_valid, 2)
end

nn = Chain(Dense(3, N_WEIGHTS, relu), Dense(N_WEIGHTS, N_WEIGHTS, relu), Dense(N_WEIGHTS, N_WEIGHTS, relu), Dense(N_WEIGHTS, N_WEIGHTS, relu), Dense(N_WEIGHTS, N_WEIGHTS, relu), Dense(N_WEIGHTS, 1)) |> gpu
p, re_nn = Flux.destructure(nn)

const σ_const = σ 
const b_const = b 

function neural_lorenz(u,p,t)
    X,Y,Z = u 

    [-σ_const * X + σ_const * Y,
    - X*Z + re_nn(p)(u)[1] - Y, 
    X*Y - b_const*Z]
end

basic_tgrad(u,p,t) = zero(u)
odefunc = ODEFunction{false}(neural_lorenz,tgrad=basic_tgrad)
node_prob = ODEProblem(odefunc, u0, (Float32(0.),Float32(dt)), p)
# is the sensealg correct?
model = ChaoticNDE(node_prob, reltol=1e-5, dt=dt)

loss = Flux.Losses.mse 
loss(model(train[1]), train[1][2])

function plot_node()
    plt = plot(valid.t, model((valid.t, valid[1][2]))', label="Neural ODE", xlabel="Time t")
    plot!(plt, valid.t, valid.data', label="Training Data",xlims=[150,180])
    display(plt)
end
plot_node()

η = 1f-3
opt = Flux.AdamW(η)
opt_state = Flux.setup(opt, model)

λ_max = 0.9056 # maximum LE of the L63

TRAIN = true
if TRAIN 
    println("starting training...")
    for i_e = 1:N_epochs

        Flux.train!(model, train, opt_state) do m, t, x
            result = m((t,x))
            loss(result, x)
        end 

        plot_node()

        δ = ChaoticNDETools.forecast_δ(model((valid.t,valid[1][2])), valid.data)
        forecast_length = findall(δ .> 0.4)[1][2] * dt * λ_max
        println("forecast_length=", forecast_length)

        if (i_e % 5) == 0  # reduce the learning rate every 30 epochs
            global η /= 2
            Flux.adjust!(opt_state, η)
        end
    end
end

