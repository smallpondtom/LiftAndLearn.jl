"""
$(SIGNATURES)

Split a data matrix into consecutive streams of `streamsize` snapshots each.

The snapshot (time) dimension is taken to be the longer dimension of `X`: for a fat matrix
(``n \\times K`` with ``K \\ge n``) the columns are split, for a tall matrix the rows are split.
The last stream may be shorter if `streamsize` does not divide the number of snapshots.

## Arguments
- `X::AbstractArray{<:Number}`: data matrix (e.g., reduced states ``\\hat{\\mathbf{X}}``,
  derivatives, or inputs). A vector is treated as a single column and split along its length.
- `streamsize::Integer`: number of snapshots per stream.

## Returns
- `streams::Vector{<:AbstractArray}`: the streams, in order.

## Examples
```julia
X = rand(3, 10)
streamify(X, 4)   # 3×4, 3×4, 3×2 blocks
```
"""
function streamify(X::AbstractArray{<:Number}, streamsize::Integer)
    m, n = checksize(X)
    if m > n
        return map(Iterators.partition(axes(X,1), streamsize)) do cols
            X[cols, :]
        end
    else
        return map(Iterators.partition(axes(X,2), streamsize)) do cols
            X[:, cols]
        end
    end
end


"""
$(SIGNATURES)

Split a data matrix into consecutive streams with the given (possibly varying) sizes.

As for the `streamsize::Integer` method, the longer dimension of `X` is the snapshot dimension.

## Arguments
- `X::AbstractArray{<:Number}`: data matrix to split.
- `streamsize::Array{<:Integer}`: size of each stream; `sum(streamsize)` should equal the number
  of snapshots (a final stream is truncated at the end of the data).

## Returns
- `streams::Vector{<:AbstractArray}`: the streams, in order.

## Examples
```julia
X = rand(3, 10)
streamify(X, [2, 3, 5])   # 3×2, 3×3, 3×5 blocks
```
"""
function streamify(X::AbstractArray{<:Number}, streamsize::Array{<:Integer})
    m, n = checksize(X)
    cum = cumsum(streamsize)
    if m > n
        return [
            i==1 ? X[1:cum[i],:] : X[1+cum[i-1]:min(cum[i],size(X,1)),:]
            for i in eachindex(cum)
        ]
    else
        return [
            i==1 ? X[:,1:cum[i]] : X[:,1+cum[i-1]:min(cum[i],size(X,2))]
            for i in eachindex(cum)
        ]
    end
end
