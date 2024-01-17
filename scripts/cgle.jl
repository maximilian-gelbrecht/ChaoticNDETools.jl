import Pkg
Pkg.activate("scripts") # change this to "." incase your "scripts" is already your working directory

using Flux, DiffEqFlux, CUDA, OrdinaryDiffEq, BenchmarkTools, JLD2, Optimisers

# not registered packages, add them manually (see comment in the Readme.md)
using ChaoticNDETools, NODEData, GinzburgLandau

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
    t_transient = 200.
    N_t_train = N_t
    N_t_valid = 8000
    N_t = N_t_train + N_t_valid + 1000
end 

begin 
    # system parameters
    #n = 128
    #L = 192 

    n = 36
    L = 75

    # generate CGLE data and construct the Laplacian 
    Δ = GinzburgLandau.Laplacian2DPeriodic(n, L)
    u0 = reshape(GinzburgLandau.initial_conditions(n),:)

    α = 2.0
    β = -1.0
    p_CGLE = [α, β, Δ]
    tspan = (0., t_transient + N_t * dt)

    prob = ODEProblem(GinzburgLandau.cgle_fd!, u0, tspan, p_CGLE)
    sol = solve(prob, Tsit5(), saveat=t_transient:dt:t_transient + N_t * dt)
end 

# we do the ANN in half-complex format, so we have to do some extra work and seperate real and imaginary parameters
begin 
    t_train = t_transient:dt:t_transient+N_t_train*dt
    data_train = DeviceArray(sol(t_train))

    t_valid = t_transient+N_t_train*dt:dt:t_transient+N_t_train*dt+N_t_valid*dt
    data_valid = DeviceArray(sol(t_valid))

    data_train = permutedims(cat(real.(data_train),imag.(data_train),dims=3),(1,3,2))
    data_valid = permutedims(cat(real.(data_valid),imag.(data_valid),dims=3),(1,3,2))

    train = NODEDataloader(Float32.(data_train), t_train, 2)
    valid = NODEDataloader(Float32.(data_valid), t_valid, 2)
end

# let's define the NPDE, some linear algebra to avoid views (non working?) and mutating (def not working)
const matRe = reshape(DeviceArray(Float32[1,0]),(1,2))
const matIm = reshape(DeviceArray(Float32[0,1]),(1,2))

const matRealPart = DeviceArray(Float32[1,0])
const matImagPart = DeviceArray(Float32[1,0])

nn = Chain(x -> transpose(x), Dense(2, N_WEIGHTS, swish), Dense(N_WEIGHTS, N_WEIGHTS, swish), Dense(N_WEIGHTS, N_WEIGHTS, swish), Dense(N_WEIGHTS, 2), x->transpose(x)) |> gpu
p, re_nn = Optimisers.destructure(nn)

function node_cgle_hc(u, p, t)
    ReU = view(u,:,1)
    ImU = view(u,:,2)
    #ReU = u * matRealPart 
    #ImU = u * matImagPart 
    nn_res = re_nn(p)(u)
    return (Δ*(ReU - α.*ImU))*matRe + (Δ*(ImU + α.*ReU))*matIm + nn_res
end

node_prob = ODEProblem(node_cgle_hc, train[1][1], (Float32(0.),Float32(dt)), p)

predict(t, u0; reltol=1e-5) = DeviceArray(solve(remake(node_prob; tspan=(t[1],t[end]),u0=u0, p=p), Tsit5(), dt=dt, saveat = t, reltol=reltol, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP())))

loss(t, u0) = sum(abs2, predict(t, view(u0,:,:,1)) - u0)
loss(train[1]...)


opt = Optimisers.AdamW(1f-3)
opt_state = Optimisers.setup(opt, model)

# setup the training loop 
λmax = 0.16724655
NN_train = length(train) > 100 ? 100 : length(train)
NN_valid = length(valid) > 100 ? 100 : length(valid)

TRAIN = true
if TRAIN 
    println("starting training...")

    for i_τ = 2:τ_max

        train_τ = NODEDataloader(train, i_τ)

        N_e = N_epochs
        if i_τ == 2 # first iteration gets longer training
            N_e *= 10
        end

        valid_error_old = Inf
        valid_error = Inf
        valid_decrease_counter = 0

        for i_e=1:N_e 
            Flux.train!(loss, Flux.params(p), train_τ, opt)

            if (i_e % 2) == 0
                train_error = mean([loss(train[i]...) for i=1:NN_train])
                valid_error = mean([loss(valid[i]...) for i=1:NN_valid])
                prediction = predict(valid.t, view(valid[1][2],:,:,1))
                δ = ChaoticNDETools.forecast_δ(prediction, valid.data) 
                forecast_length = findall(δ .> 0.4)[1][2] * dt * λmax
                
                println("AdamW, i_τ=", i_τ, " training error =",train_error, " valid error=", valid_error, "prediction [λ_max t] =", forecast_length)

                if valid_error_old < valid_error
                    valid_decrease_counter += 1
                end 

                if valid_decrease_counter > 2
                    break
                end
            end

            if (i_e % 10) == 0  # reduce the learning rate every 30 epochs
                opt[1].eta /= 2
            end
        end

        global p = cpu(p)
        @save SAVE_NAME p
        global p = gpu(p)
                    
        # this seems to be needed to avoid Out-of-RAM on GPU
        GC.gc(true)
    end 
else   
    println("loading...")
    @load SAVE_NAME p
end


