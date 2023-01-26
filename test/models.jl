using OrdinaryDiffEq, DiffEqSensitivity, Flux

# test with a Lotka Volterra system, adjusted from scripts/lv.jl 
# we just test if everything compiles and runs without errors 

begin
    N_t = 200
    τ_max = 2
    N_WEIGHTS = 5
    dt = 0.1
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
    tspan = (0.,5.)
    
    x0 = [0.44249296, 4.6280594]
    
    prob = ODEProblem(lotka_volterra, x0, tspan, p) 
    sol = solve(prob, Tsit5(), saveat=dt)
end 

t = Float32.(0:dt:5.)
train = [(t, Float32.(Array(sol(t))))]

nn = Chain(Dense(2, N_WEIGHTS, swish), Dense(N_WEIGHTS, N_WEIGHTS, swish), Dense(N_WEIGHTS, 2)) |> gpu
p, re_nn = Flux.destructure(nn)

neural_ode(u, p, t) = re_nn(p)(u)
basic_tgrad(u,p,t) = zero(u)
nnf = ODEFunction{false}(neural_ode, tgrad=basic_tgrad)

node_prob = ODEProblem(nnf, x0, (Float32(0.),Float32(dt)), p)

model = ChaoticNDE(node_prob)
model(train[1])

loss(x, y) = sum(abs2, x - y)
loss(model(train[1]), train[1][2]) 

# check if the gradient works 
g = gradient(model) do m
    result = m(train[1])
    loss(result, train[1][2])
end
pgrad = g[1].p

# do a check that the gradient is nonzero, noninf and nonnothing
@test sum(isnan.(pgrad)) == 0
@test sum(isinf.(pgrad)) == 0 
@test sum(isnothing.(pgrad)) == 0 

# check if new parameters are set correctly 
p_new = rand(Float32, size(p))
ChaoticNDETools.set_params!(model, p_new)

@test all(model.p .== p_new)

