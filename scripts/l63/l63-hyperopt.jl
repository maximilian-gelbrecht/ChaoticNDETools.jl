# A script for hyperparameter optimiziation for hybrid L63, this takes a long time to execute (I ran it on a HPC)

import Pkg
Pkg.activate(".") # change this to "." incase your "scripts" is already your working directory

using Flux, DiffEqFlux, CUDA, OrdinaryDiffEq, JLD2, Random

# not registered packages, add them manually (see comment in the Readme.md)
using ChaoticNDETools, NODEData, SlurmHyperopt

if length(ARGS) > 1
    println("Loading Hyperparameters...")
    @load "hyperopt.jld2" sho 
    i_job = parse(Int,ARGS[2])
    pars = sho[i_job]  
    N_weights = pars[:N_weights]
    N_hidden_layers = pars[:N_hidden_layers]
    τ_max = pars[:τ_max]
    func = pars[:activation]
    η = pars[:eta]
    η_decrease = pars[:eta_decrease]
    println("Hyperparameter:")
    println(pars)
    println("-----")
    if func == "relu"
        activation = relu
    elseif func == "selu"
        activation = selu 
    elseif func == "swish"
        activation = swish
    elseif func == "tanh"
        activation = tanh
    else
        error("unkonwn activation function")
    end
else 
    N_weights = 5
    N_hidden_layers = 1
    τ_max = 2
    activation = swish
    i_job = 0
    η = 1f-3 
    η_decrease = 5
end

begin
    N_epochs = 20
    N_t = 200
    dt = 0.1
    t_transient = 100.
    N_t_train = N_t
    N_t_valid = N_t_train*2
    N_t = N_t_train + N_t_valid
    η = 1f-3
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

    train = NODEDataloader(Float32.(data_train), Float32.(t_train), 2)
    valid = NODEDataloader(Float32.(data_valid), Float32.(t_valid), 2)
end

const σ_const = σ 
const b_const = b 
const λ_max = 0.9056 # maximum LE of the L63

"""
    train(N_epochs, N_weights, σ, τ_max, η)

Train the hybrid NODE with `N_weights`, activation function `σ`, until integration length `τ_max` with learning rate `η`
"""
function train_node(train, valid, N_epochs, N_weights, N_hidden_layers, activation, τ_max, η)


    hidden_layers = [Flux.Dense(N_weights, N_weights, activation) for i=1:N_hidden_layers]
    nn = Chain(Flux.Dense(3, N_weights, activation), hidden_layers...,  Flux.Dense(N_weights, 1)) |> gpu
    p, re_nn = Flux.destructure(nn)

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

    η = 1f-3
    opt = Flux.AdamW(η)
    opt_state = Flux.setup(opt, model)

    N_epochs = ceil(N_epochs)
    opt = Flux.AdamW(η)

    for i_τ = 2:τ_max
        println("starting training ]with N_EPOCHS= ",N_epochs, " - N_weights=",N_weights, " - activation=",activation, " - η=",η)
        N_epochs_i = i_τ == 2 ? 2*Int(ceil(N_epochs/τ_max)) : ceil(N_epochs/τ_max) # N_epochs sets the total amount of epochs 
        
        train_i = NODEDataloader(train, i_τ)
        for i_e = 1:N_epochs_i

            Flux.train!(model, train_i, opt_state) do m, t, x
                result = m((t,x))
                loss(result, x)
            end 

            if (i_e % 5) == 0  # reduce the learning rate every 30 epochs
                global η /= 2
                Flux.adjust!(opt_state, η)
            end
        end
        GC.gc(true)
    end
    return ChaoticNDETools.average_forecast_length(model, valid, 300; λ_max=λ_max), p
end

forecast_length, p = train_node(train, valid, N_epochs, N_weights, N_hidden_layers, activation, τ_max, η)

if length(ARGS) > 1
    println("saving...")
    SlurmHyperopt.save_result(sho, HyperoptResults(pars=pars, res=forecast_length, additonal_res=p), i_job)
end