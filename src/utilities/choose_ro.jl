export choose_ro

"""
$(SIGNATURES)

Choose reduced orders that retain a prescribed amount of energy of a singular value spectrum.

The truncation energy loss for order ``r`` is
```math
e_r = 1 - \\frac{\\sqrt{\\sum_{i=1}^{r} \\sigma_i^2}}{\\|\\boldsymbol{\\sigma}\\|_2} .
```
For each threshold ``10^{-1}, 10^{-2}, \\ldots, 10^{\\mathrm{en\\_low}}`` the smallest ``r`` with
``e_r`` below the threshold is returned.

## Arguments
- `Σ::Vector`: singular values (e.g., from the SVD of a snapshot or Hankel matrix), sorted in
  descending order.

## Keyword Arguments
- `en_low=-15`: exponent of the smallest energy-loss threshold; thresholds run over
  `10.0 .^ (-1.0:-1.0:en_low)`.

## Returns
- `r_all::Vector{Int}`: the reduced order for each threshold (length `length(-1.0:-1.0:en_low)`).
- `en::Vector`: the energy loss ``e_r`` for every ``r = 1, \\ldots,`` `length(Σ)`.

## Notes
- If some threshold is never reached (e.g., `en_low` is below the round-off level of `en`),
  `findfirst` returns `nothing` and the function throws an error.
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