"""
    streamify(X::AbstractArray{<:Number}, datasize::Integer)

Split an array into streams of the given size.

## Arguments
- `X::AbstractArray{<:Number}`: The array to split.
- `streamsize::Integer`: A single size of the streams.

## Returns
An array of arrays, each containing a stream of the original array.
"""
function streamify(X::AbstractArray{<:Number}, streamsize::Integer)
    m, n = size(X)
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
    streamify(X::AbstractArray{<:Number}, streamsize::Array{<:Integer})

Split an array into streams of the given sizes.

## Arguments
- `X::AbstractArray{<:Number}`: The array to split.
- `streamsize::Array{<:Integer}`: The sizes of the streams.

## Returns
An array of arrays, each containing a stream of the original array.
"""
function streamify(X::AbstractArray{<:Number}, streamsize::Array{<:Integer})
    m, n = size(X)
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
