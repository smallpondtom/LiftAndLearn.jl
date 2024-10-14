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
    testfile("utilities/matrix_dimensions.jl")
    testfile("intrusive/pod.jl")
    testfile("LnL/lifting.jl")
    testfile("LnL/opinf.jl")
    testfile("LnL/lnl.jl")
    testfile("LnL/epopinf.jl")
end