using LiftAndLearn
using Test

# DO_EXTENSIVE_TESTS = get(ENV, "CHAOSTOOLS_EXTENSIVE_TESTS", "false") == "true"

function testfile(file, testname=defaultname(file))
    println("running test file $(file)")
    @testset "$testname" begin; include(file); end
    return
end
defaultname(file) = uppercasefirst(replace(splitext(basename(file))[1], '_' => ' '))

@testset "LiftAndLearn" begin

    testfile("utilities/matrices.jl")
    testfile("utilities/integrators.jl")

    testfile("intrusive/pod.jl")
    testfile("LnL/lifting.jl")

    testfile("LnL/opinf.jl")
    testfile("LnL/lnl.jl")
    testfile("LnL/optimize.jl")

    testfile("models/burgers.jl")
    testfile("models/kse.jl")
end