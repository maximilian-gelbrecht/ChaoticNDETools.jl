import Base.show, Base.summary
using Flux, Distributions, Optimisers

"""
ParSkipConnection(layer, connection)

A version of SkipConnection that is paramaterized with one single parameter, thus the output is `connection(w .* layer(input), input)`
"""
struct ParSkipConnection{T,S,F}
    layers::T
    connection::S #user can pass arbitrary connections here, such as (a,b) -> a + b
    w::F
end

ParSkipConnection(layer, connection; initw=Flux.glorot_uniform) = ParSkipConnection(layer, connection, initw())

Flux.@layer ParSkipConnection

function (skip::ParSkipConnection)(input)
    skip.connection(skip.w .* skip.layers(input), input)
end

function Base.show(io::IO, b::ParSkipConnection)
    print(io, "ParSkipConnection(", b.layers, ", ", b.connection, ", ,w=",b.w,")")
end


"""
  NablaSkipConnection

With the input ``x``, passes ``w * (∇ * x) + (1 - |w|) * x`` , where ``w\\in\\mathbb{R}, w\\in [0,1]``. Here ``\\Nabla`` is a finite difference derivative matrix
"""
struct NablaSkipConnection{F,S}
  w::F
  one::F
  ∇::S
end

function NablaSkipConnection(∇, w::F) where F<:Number
    return NablaSkipConnection(DeviceArray([w]), DeviceArray([F(1)]), ∇)
end

NablaSkipConnection(∇, initw=()->Float32.(rand(Uniform(0.4f0,0.6f0)))) = NablaSkipConnection(∇, initw())

function (skip::NablaSkipConnection)(input)
  skip.w .* (skip.∇ * input) + (skip.one .- abs.(skip.w)) .* input
end

Flux.@layer NablaSkipConnection
Optimisers.trainable(skip::NablaSkipConnection) = (w = skip.w,)

function Base.show(io::IO, b::NablaSkipConnection)
    print(io, "NablaSkipConnection with w=", b.w)
end