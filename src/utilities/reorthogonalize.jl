"""
$(SIGNATURES)

Re-orthonormalize the columns of `V` in place with respect to the inner product weighted by `M`,
using modified Gram–Schmidt with reorthogonalization.

This is used to counter the loss of orthogonality of the basis in the incremental SVD (iSVD).
As a cheap check, the procedure is only run if the first and last columns have lost
``\\mathbf{M}``-orthogonality, i.e., if ``|\\mathbf{v}_r^\\top \\mathbf{M} \\mathbf{v}_1| >`` `tol`;
otherwise `V` is left untouched. When it runs, every column is orthogonalized twice against the
preceding columns (Algorithm 6.11 of [GanderGK2014]) and then normalized, so that
``\\mathbf{V}^\\top \\mathbf{M} \\mathbf{V} \\approx \\mathbf{I}`` on exit.

## Arguments
- `V::AbstractMatrix{T}`: ``n \\times r`` basis matrix (e.g., left singular vectors). Overwritten
  with the re-orthonormalized basis.
- `M::AbstractMatrix{T}`: ``n \\times n`` symmetric positive-definite weight matrix defining the inner
  product ``\\langle \\mathbf{x}, \\mathbf{y} \\rangle_M = \\mathbf{x}^\\top \\mathbf{M} \\mathbf{y}``.
- `tol::Real`: threshold on ``|\\mathbf{v}_r^\\top \\mathbf{M} \\mathbf{v}_1|`` that triggers reorthogonalization.

## Returns
- `nothing`; mutates `V` in place.

## Notes
- Only the first/last column pair is checked; loss of orthogonality among other columns alone
  does not trigger the procedure.
- For a single column (``r = 1``) the check compares ``\\mathbf{v}_1^\\top\\mathbf{M}\\mathbf{v}_1`` with
  `tol`, so the column is simply re-normalized.

## References
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

"""
$(SIGNATURES)

Re-orthonormalize the columns of `V` in place (Euclidean inner product) using modified
Gram–Schmidt with reorthogonalization.

Equivalent to [`reorthogonalize!(V, M, tol)`](@ref reorthogonalize!) with ``\\mathbf{M} = \\mathbf{I}``.
The procedure is only run if ``|\\mathbf{v}_r^\\top \\mathbf{v}_1| >`` `tol` (first vs. last column);
otherwise `V` is left untouched. When it runs, ``\\mathbf{V}^\\top \\mathbf{V} \\approx \\mathbf{I}`` on exit
and the column space of `V` is preserved.

## Arguments
- `V::AbstractMatrix{T}`: ``n \\times r`` basis matrix. Overwritten with the re-orthonormalized basis.
- `tol::Real`: threshold on ``|\\mathbf{v}_r^\\top \\mathbf{v}_1|`` that triggers reorthogonalization.

## Returns
- `nothing`; mutates `V` in place.

## Notes
- Only the first/last column pair is checked (see the weighted method).
"""
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