using Test

using BlockDiagonals
using LinearAlgebra
using DataFrames
using PolynomialModelReductionDataset
const Pomoreda = PolynomialModelReductionDataset

using LiftAndLearn
const LnL = LiftAndLearn

function testfile(file, testname=defaultname(file))
    println("running test file $(file)")
    @testset "$testname" begin; include(file); end
    return
end
defaultname(file) = uppercasefirst(replace(splitext(basename(file))[1], '_' => ' '))

@testset "LiftAndLearn" begin

    testfile("utilities/matrix_dimensions.jl")

    testfile("intrusive/pod.jl")
    testfile("LnL/lifting.jl")

    testfile("LnL/opinf.jl")
    testfile("LnL/lnl.jl")
    # testfile("LnL/optimize.jl")
end