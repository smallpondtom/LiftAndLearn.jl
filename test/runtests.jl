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

    

    # include("timeevolution/orbitdiagram.jl")

    # testfile("chaosdetection/lyapunovs.jl")
    # testfile("chaosdetection/gali.jl")
    # include("chaosdetection/partially_predictable.jl")
    # include("chaosdetection/01test.jl")
    # testfile("chaosdetection/expansionentropy.jl")

    # include("stability/fixedpoints.jl")
    # include("periodicity/periodicorbits.jl")
    # include("periodicity/period.jl")

    # testfile("rareevents/return_time_tests.jl", "Return times")

    # testfile("dimreduction/broomhead_king.jl")
    # TODO: simplify and make faster this:
    # testfile("dimreduction/dyca.jl")

end