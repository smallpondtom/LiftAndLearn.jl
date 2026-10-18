using Test
using LinearAlgebra
using LiftAndLearn
const LnL = LiftAndLearn

include("tools/rk4.jl")
include("tools/models.jl")

function testfile(file, testname=defaultname(file))
    println("running test file $(file)")
    @testset "$testname" begin; include(file); end
    return
end
defaultname(file) = uppercasefirst(replace(splitext(basename(file))[1], '_' => ' '))

@testset "LiftAndLearn" begin
    # Utilities
    testfile("utilities/matrix_dimensions.jl")
    testfile("utilities/analysis.jl")
    testfile("utilities/interp.jl")
    testfile("utilities/spline.jl")
    testfile("utilities/linalg.jl")

    # Operator Inference building blocks
    testfile("opinf/leastsquares.jl")
    testfile("opinf/tikhonov.jl")
    testfile("opinf/time_derivative.jl")
    testfile("opinf/operators.jl")
    testfile("opinf/regressions.jl")

    # Intrusive, OpInf, Lift & Learn, EP-OpInf
    testfile("intrusive/pod.jl")
    testfile("LnL/lifting.jl")
    testfile("LnL/opinf.jl")
    testfile("LnL/lnl.jl")
    testfile("LnL/epopinf.jl")

    # Streaming OpInf
    testfile("streaming/streamify.jl")
    testfile("streaming/twopass.jl")
    testfile("streaming/onepass.jl")
end
