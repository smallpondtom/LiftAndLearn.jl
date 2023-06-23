export Elimat 

struct Elimat{T} <: AbstractMatrix{T}
    m::Int
end

size(L::Elimat) = (Int(L.m * (L.m+1) / 2), Int(L.m^2))

# Index into a Riemann
@inline Base.@propagate_inbounds function getindex(L::Elimat, i::Integer, j::Integer)
    @boundscheck checkbounds(L, i, j)
#     T = tril(ones(m, m)) # Lower triangle of 1's
#     f = findall(x -> x == 1, T[:]) # Get linear indexes of 1's
#     k = m * (m + 1) / 2 # Row size of L
#     m2 = m * m # Colunm size of L
#     x = f + m2 * (0:k-1) # Linear indexes of the 1's within L'

#     row = [mod(a, m2) != 0 ? mod(a, m2) : m2 for a in x]
#     col = [mod(a, m2) != 0 ? div(a, m2) + 1 : div(a, m2) for a in x]
    
#     if i in row && j in col
#         return 1
#     end
    if i ≤ j && (j+1) % (i+1) == 0
        return i
    else
        return -1
    end
end

function inv(L::Elimat{T}) where T
    return L \ 1.0I(L.m)
end

