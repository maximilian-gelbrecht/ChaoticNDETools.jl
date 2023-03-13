import Pkg
Pkg.activate("scripts") # change this to "." incase your "scripts" is already your working directory

using Flux, DiffEqFlux, CUDA, OrdinaryDiffEq, BenchmarkTools, JLD2, Plots, Random

# not registered packages, add them manually (see comment in the Readme.md)
using ChaoticNDETools, NODEData
Random.seed!(123)
#=
this script can also be called from the command line with extra arguments (e.g. by a batch system such as SLURM), otherwise default values are used.

the extra arguments are
* 1) SAVE_NAME, base name of the saved files
* 2) N_EPOCHS
* 3) N_t, length of training dataset
* 4) τ_max
* 5) RELOAD, reload data
=#
begin
    parse_ARGS(i, ARGS, default) = length(ARGS) >= i ? parse(Int, ARGS[i]) : default

    SAVE_NAME = length(ARGS) >= 2 ? ARGS[1] : "local-test"
    N_epochs = parse_ARGS(2, ARGS, 35)

    N_t = parse_ARGS(3, ARGS, 100) 
    τ_max = parse_ARGS(4, ARGS, 2) 
    RELOAD = parse_ARGS(5, ARGS, 0) 
    RELOAD = RELOAD == 1 ? true : false

    N_WEIGHTS = 10
    dt = 0.1f0
    t_transient = 200f0
    N_t_train = N_t
    N_t_valid = 9*N_t
    N_t = N_t_train + N_t_valid 
end 

begin 
    # system parameters
    #n = 1024
    #L = 290

    n = 128
    L = 36

    # generate KS data and construct the FD Operators
    ∂x = ChaoticNDETools.∂x_PBC(n, Float32(L/(n-1)))
    ∂x² = ChaoticNDETools.∂x²_PBC(n, Float32(L/(n-1)))
    ∂x⁴ = ChaoticNDETools.∂x⁴_PBC(n, Float32(L/(n-1)))
    u0 = DeviceArray(Float32(0.01)*(rand(Float32, n) .- 0.5f0))

    function ks!(du,u,p,t)
        du .= -∂x⁴*u - ∂x²*u - u.*(∂x*u)
    end

    tspan = (0f0, Float32(t_transient + N_t * dt))

    prob = ODEProblem(ks!, u0, tspan)
    sol = solve(prob, Tsit5(), saveat=t_transient:dt:t_transient + N_t * dt)
end 

# we prepare the training and valid data 
begin 
    t_train = t_transient:dt:t_transient+N_t_train*dt
    data_train = DeviceArray(sol(t_train))

    t_valid = t_transient+N_t_train*dt:dt:t_transient+N_t_train*dt+N_t_valid*dt
    data_valid = DeviceArray(sol(t_valid))

    train = NODEDataloader(Float32.(data_train), t_train, 2)
    valid = NODEDataloader(Float32.(data_valid), t_valid, 2)
end

# let's define the NPDE 
nn = Chain(NablaSkipConnection(∂x), NablaSkipConnection(∂x), NablaSkipConnection(∂x), NablaSkipConnection(∂x), x->transpose(x), SkipConnection(Chain(Dense(1, N_WEIGHTS, swish), Dense(N_WEIGHTS, N_WEIGHTS, swish), Dense(N_WEIGHTS, 1)),+), x->reshape(x,:)) |> gpu

p, re = Flux.destructure(nn)

function node_ks(u,p,t)
    -∂x⁴*u - re(p)(u) - u.*(∂x*u)
end

node_prob = ODEProblem(node_ks, u0, (Float32(0.),Float32(dt)), p)
model = ChaoticNDE(node_prob)

nabla_penalty_func_vec(x) =  4f0 .* abs.((2f0.*((x .- 0.5f0))).^6 .- 1f0)
loss(x, y) = sum(abs2, x - y) 
loss(model(train[1]), train[1][2]) 

η = 1f-3
opt = Flux.AdamW(η)
opt_state = Flux.setup(opt, model) 

# setup the training loop 

NN_train = length(train) > 100 ? 100 : length(train)
NN_valid = length(valid) > 100 ? 100 : length(valid)

λmax = 0.07 # maximum LE, computed beforehand with DynamicalSystems.jl
const γ = 1f-5
predictions = zeros(Float32, n, length(valid.t), N_epochs+1)

predictions[:,:,1] = model((valid.t, view(valid[1][2],:,1)))

TRAIN = true
if TRAIN 
    println("starting training...")

    for i_τ = 2:τ_max

        train_τ = NODEDataloader(train, i_τ)

        N_e = N_epochs
    
        valid_error_old = Inf
        valid_error = Inf
        valid_decrease_counter = 0

        for i_e=1:N_e 
            
            Flux.train!(model, train, opt_state) do m, t, x
                result = m((t,x))
                loss(result, x) + sum(nabla_penalty_func_vec(m.p[1:4])) + γ*sum(abs.(p[5:end]))
            end

            if (i_e % 1) == 0
                train_error = mean([loss(model(train[i]),train[i][2]) for i=1:NN_train])
                valid_error = mean([loss(model(valid[i]),valid[i][2]) for i=1:NN_valid])
                predictions[:,:,i_e+1] = model((valid.t, valid[1][2]))
                δ = ChaoticNDETools.forecast_δ(predictions[:,:,i_e], valid.data) 
                forecast_length = findall(δ .> 0.4)[1][2] * dt * λmax

                println("AdamW, i_τ=", i_τ, "- training error =",train_error, "- valid error=", valid_error, " - prediction [λ_max t] =", forecast_length)

                if valid_error_old < valid_error
                    valid_decrease_counter += 1
                end 

                if valid_decrease_counter > 2
                    break
                end
            end
            if (i_e % 3) == 0  # reduce the learning rate every 3 epochs
                η /= 2
                Flux.adjust!(opt_state, η)
            end
           
        end

        global p = cpu(p)
        @save SAVE_NAME p
        global p = gpu(p)
        
        @save "predictions-anim.jld2" predictions

        # this seems to be needed to avoid Out-of-RAM on GPU
        GC.gc(true)
    end 
else   
    println("loading...")
    @load SAVE_NAME p
end



PLOT = false 

if PLOT 
    Plots.pyplot()

    anim = @animate for i ∈ 1:(N_epochs+1)      
        dataplot = cat(train.data, predictions[:,:,i], dims=2)
        plt = heatmap(dataplot, xlabel="Time Steps", ylabel="Grid Points", title=string("Epoch ",i-1), clims=(-3,3))
        plot!(plt, [length(train.t), length(train.t)],[1,n], linewidth=4, label="training data end", c=:black)
    end
    gif(anim, "training-ks.gif",fps=1)
end

