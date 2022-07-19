# A script for hyperparameter optimiziation for hybrid L63, this takes a long time to execute (I ran it on a HPC)

import Pkg
Pkg.activate("scripts") # change this to "." incase your "scripts" is already your working directory

using Flux, DiffEqFlux, CUDA, OrdinaryDiffEq, BenchmarkTools, JLD2, Plots, Random, Hyperopt

# not registered packages, add them manually (see comment in the Readme.md)
using ChaoticNDETools, NODEData

Random.seed!(1234)

begin
    parse_ARGS(i, ARGS, default) = length(ARGS) >= i ? parse(Int, ARGS[i]) : default

    SAVE_NAME = length(ARGS) >= 1 ? ARGS[1] : "l63-hyperpar.jld2"
    N_epochs = parse_ARGS(2, ARGS, 30)

    N_t = parse_ARGS(3, ARGS, 300) 
    τ_max = parse_ARGS(4, ARGS, 3) 

    N_WEIGHTS = 15
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

const σ_const = σ 
const b_const = b 
const λ_max = 0.9056 # maximum LE of the L63

"""
    train(N_epochs, N_weights, σ, τ_max, η)

Train the hybrid NODE with `N_weights`, activation function `σ`, until integration length `τ_max` with learning rate `η`
"""
function train_node(N_epochs, N_weights, σ, τ_max, η)

    nn = Chain(Dense(3, N_WEIGHTS, σ), Dense(N_WEIGHTS, N_WEIGHTS, σ), Dense(N_WEIGHTS, N_WEIGHTS, σ), Dense(N_WEIGHTS, 1)) |> gpu
    p, re_nn = Flux.destructure(nn)

    function neural_lorenz!(du,u,p,t)
        X,Y,Z = u 
    
        du[1] = -σ_const * X + σ_const * Y
        du[2] = - X*Z + re_nn(p)(u)[1] - Y 
        du[3] = X*Y - b_const*Z
    end

    node_prob = ODEProblem(neural_lorenz!, u0, (Float32(0.),Float32(dt)), p)

    predict(t, u0; reltol=1e-5) = DeviceArray(solve(remake(node_prob; tspan=(t[1],t[end]),u0=u0, p=p), Tsit5(), dt=dt, saveat = t, reltol=reltol))
    loss(t, u0) = sum(abs2, predict(t, view(u0,:,1)) - u0)
    loss(train[1]...)

    N_epochs = ceil(N_epochs)
    opt = Flux.AdamW(η)
    forecast_length=Inf
    for i_τ = 2:τ_max
        println("starting training ]with N_EPOCHS= ",N_epochs, " - N_weights=",N_weights, " - σ=",σ, " - η=",η)
        N_epochs_i = i_τ == 2 ? 2*Int(ceil(N_epochs/τ_max)) : ceil(N_epochs/τ_max) # N_epochs sets the total amount of epochs 
        for i_e = 1:N_epochs_i

            Flux.train!(loss, Flux.params(p), train, opt)
            #plot_node()
            δ = ChaoticNDETools.forecast_δ(Array(predict(valid.t,valid.data[:,1])), valid.data)
            forecast_length = findall(δ .> 0.4)[1][2] * dt * λ_max

            if (i_e % 5) == 0  # reduce the learning rate every 30 epochs
                opt[1].eta /= 2
            end
        end
    end
   
    return ChaoticNDETools.average_forecast_length(predict, valid, λ_max=λ_max)
end

ho = @hyperopt for i=1:50, sampler=RandomSampler(), N_weights = 5:20, σ = [swish], τ_max=2:10, learningrate=[1e-3]
    forecast_length = train_node(resources, N_weights, σ, τ_max, learningrate)
    @show forecast_length
end


println(ho)

printmin(ho)

@save SAVE_NAME hohb 

