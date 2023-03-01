# provides some 1d finite difference matrices for use with the layers
using LinearAlgebra 

"""
    ∂x_PBC(n::Integer, dx::T)

2nd order central finite difference matrix for 1d domains with periodic boundary conditions
"""
function ∂x_PBC(n::Integer, dx::T) where T
    ∂x = (diagm(1=>ones(T, n-1)) + diagm(-1=>-1*ones(T, n-1)))
    ∂x[1,end] = T(-1)
    ∂x[end,1] = T(1)
    ∂x ./= (2*dx)
    #∂x = sparse(∂x)

    return DeviceArray(∂x)
end

"""
    ∂x²_PBC(n::Integer, dx::T)

2nd order central finite difference matrix of the second derivative for 1d domains with periodic boundary conditions
"""
function ∂x²_PBC(n::Integer, dx::T) where T
    ∂x² = diagm(0=>-2*ones(T, n)) + diagm(-1=>ones(T, n-1)) + diagm(1=>ones(T, n-1))
    ∂x²[1,end] = 1
    ∂x²[end,1] = 1
    ∂x² ./= (dx)^2
    #∂x² = sparse(∂x²)

    return DeviceArray(∂x²)
end

"""
    ∂x⁴_PBC(n::Integer, dx::T)

4th derivative finite difference matrix with periodic boundary conditions
"""
function ∂x⁴_PBC(n::Integer, dx::T) where T
    ∂x² = ∂x²_PBC(n, dx)
    return DeviceArray(∂x² * ∂x²)
end
