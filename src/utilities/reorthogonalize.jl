"""
    reorthogonalize!(U::AbstractArray{T}, W::AbstractArray{T}, tol::Real) where {T<:Number}

Reorthogonalize the left singular vectors of the incremental SVD algorithm using modified Gram-Schmidt.

# Arguments
- `V::AbstractArray{T}`: Left singular vector matrix.
- `M::AbstractArray{T}`: Weight matrix.
- `tol::Real`: Tolerance for reorthogonalization.

# References
- [GanderGK2014] 
  Gander W, Gander MJ, Kwok F. Scientific computing-An introduction using Maple and MATLAB. 
  Springer Science & Business; 2014 Apr 23.
"""
function reorthogonalize!(V::AbstractMatrix{T}, M::AbstractMatrix{T}, tol::Real) where {T<:Number}
    # Dimension
    r = size(V, 2)
    R = zeros(T, r, r)
    if abs(dot(V[:, end], M * V[:, 1])) > tol
        @views for k in 1:r
            for _ = 1:2  # do this twice (from p307 algo 6.11 in [GanderGK2014])
                for i = 1:k-1
                    E = dot(V[:, i], M * V[:, k])
                    V[:, k] .-= E * V[:, i]
                    R[i, k] += E
                end
            end
            R[k, k] = sqrt(dot(V[:, k], M * V[:, k]))
            V[:, k] ./= R[k, k]
        end
    end
end

# Dispatch
function reorthogonalize!(V::AbstractMatrix{T}, tol::Real) where {T<:Number}
    # Dimension
    r = size(V, 2)
    R = zeros(T, r, r)
    if abs(dot(V[:, end], V[:, 1])) > tol
        @views for k in 1:r
            for _ = 1:2  # do this twice (from p307 algo 6.11 in [GanderGK2014])
                for i = 1:k-1
                    E = dot(V[:, i], V[:, k])
                    V[:, k] .-= E * V[:, i]
                    R[i, k] += E
                end
            end
            R[k, k] = sqrt(dot(V[:, k], V[:, k]))
            V[:, k] ./= R[k, k]
        end
    end
end