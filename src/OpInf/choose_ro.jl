export choose_ro

"""
    choose_ro(Σ::Vector; en_low=-15) → r_all, en

Choose reduced order (ro) that preserves an acceptable energy.

## Arguments
- `Σ::Vector`: Singular value vector from the SVD of some Hankel Matrix
- `en_low`: minimum size for energy preservation

## Returns
- `r_all`: vector of reduced orders
- `en`: vector of energy values
"""
function choose_ro(Σ::Vector; en_low=-15)
    # Energy loss from truncation
    en = 1 .- sqrt.(cumsum(Σ .^ 2)) / norm(Σ)

    # loop through ROM sizes
    en_vals = map(x -> 10.0^x, -1.0:-1.0:en_low)
    r_all = Vector{Float64}()
    for rr = axes(en_vals, 1)
        # determine # basis functions to retain based on energy lost
        en_thresh = en_vals[rr]
        push!(r_all, findfirst(x -> x < en_thresh, en))
    end

    return Int.(r_all), en
end