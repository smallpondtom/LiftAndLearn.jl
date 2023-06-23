#!/bin/bash

# Run the energy-preserving OpInf test 10 times and collect the data

for i in {1..20};
do
    julia --project ./scripts/burgers_ep-OpInf.jl

    filename="epOpInf_data"
    mv examples/data/${filename}.jld2 examples/data/${filename}_$(date +%m%d%Y).jld2
done

