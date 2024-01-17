# test that the layers find a first derivative
using Flux, Optimisers

x=0:π/100:5π
t=0:π/10:2π

data = sin.(x)
data∂ = cos.(x)

training_set = [(data, data∂) for i ∈ eachindex(t)]

∂x = ChaoticNDETools.∂x_PBC(length(x), x[2]-x[1])

m = Chain(ChaoticNDETools.NablaSkipConnection(∂x),ChaoticNDETools.NablaSkipConnection(∂x),ChaoticNDETools.NablaSkipConnection(∂x))

# do we have three trainable parameters? 

@test length(Flux.params(m)) == 3

# if so, let's train them 
loss(model,x,y) = Flux.mse(model(x),y) 

opt = Optimisers.AdamW()
opt_state = Optimisers.setup(opt, m)
loss(m, training_set[1]...)

for i=1:500
    Flux.train!(loss, m, training_set, opt_state)
end

@test loss(m, training_set[1]...) < 1.

p, __ = Optimisers.destructure(m)

@test 0.9 < sum(p) < 1.1