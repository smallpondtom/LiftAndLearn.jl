"""
    batchify(X::AbstractArray{<:Number}, batchsize::Integer)

Split an array into batches of the given size.

## Arguments
- `X::AbstractArray{<:Number}`: The array to split.
- `batchsize::Integer`: A single size of the batches.

## Returns
An array of arrays, each containing a batch of the original array.
"""
function batchify(X::AbstractArray{<:Number}, batchsize::Integer)
    m, n = size(X)
    if m > n
        return map(Iterators.partition(axes(X,1), batchsize)) do cols
            X[cols, :]
        end
    else
        return map(Iterators.partition(axes(X,2), batchsize)) do cols
            X[:, cols]
        end
    end
end


"""
    batchify(X::AbstractArray{<:Number}, batchsize::Array{<:Integer})

Split an array into batches of the given sizes.

## Arguments
- `X::AbstractArray{<:Number}`: The array to split.
- `batchsize::Array{<:Integer}`: The sizes of the batches.

## Returns
An array of arrays, each containing a batch of the original array.
"""
function batchify(X::AbstractArray{<:Number}, batchsize::Array{<:Integer})
    m, n = size(X)
    cum = cumsum(batchsize)
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
