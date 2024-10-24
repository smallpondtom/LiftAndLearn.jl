function quad_indices(N, r)
    xsq_idx = [1 + (N + 1) * (n - 1) - n * (n - 1) / 2 for n in 1:N]
    extract_idx = [collect(x:x+(r-j)) for (j, x) in enumerate(xsq_idx[1:r])]
    return Int.(reduce(vcat, extract_idx))
end

function cube_indices(N,r)
    ct = 0
    tmp = []
    for i in 1:N, j in i:N, k in j:N
        ct += 1
        if (i <= r) && (j <= r) && (k <= r)
            push!(tmp, ct)
        end
    end
    return tmp
end

function extract_indices(stream, n, r, system)
    # Start with linear term
    extract_idx = collect(1:r)
    shift = n

    # Quadratic 
    if 2 in system.state
        tmp = quad_indices(n, r)
        extract_idx = vcat(extract_idx, tmp .+ shift)
        shift += n * (n + 1) / 2
    end

    # Cubic
    if 3 in system.state
        tmp = cube_indices(n, r)
        extract_idx = vcat(extract_idx, tmp .+ shift)
        shift += n * (n + 1) * (n + 2) / 6
    end

    # control
    if !iszero(system.control)
        extract_idx = vcat(
            extract_idx, 
            collect(1:stream.dims[:m]) .+ shift
        )
    end

    return Int.(extract_idx)
end