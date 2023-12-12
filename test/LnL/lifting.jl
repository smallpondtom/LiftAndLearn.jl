using LiftAndLearn
using Test

const LnL = LiftAndLearn


@testset "Test lifting with simple pendulum" begin
    """
     For a simple pendulum we have 
        x1' = x2
        x2' = -g/l sin(x1)
     The lifted system becomes 
        x1' = x2
        x2' = -g/l x3
        x3' = x2*x4
        x4' = -x2*x3
     when x3 = sin(x1) and x4 = cos(x1)
    """
    g = 1.0
    l = 1.0
    N = 2
    Nl = 4

    # test data
    x = [π π/2 0; 1.0 0 -1.0]
    xsep = [x[1:1,:], x[2:2,:]]
    x_lift = [π π/2 0; 1.0 0.0 -1.0; 0.0 1.0 0.0; -1.0 0.0 1.0]

    # Define the lifting functions
    lifter = LnL.lifting(N, Nl, [x -> sin.(x[1]), x -> cos.(x[1])])
    lift_data = lifter.map(xsep)
    lift_dataNL = lifter.mapNL(xsep)
    @test lift_data ≈ x_lift
    @test lift_dataNL ≈ x_lift[3:4, :]
end


@testset "Test data lifting" begin
   D =  [-0.365212   0.0        0.0
         -0.930924   0.0        0.0
         0.0       -0.585857  -0.810414
         0.0       -0.810414   0.585857]
   X = [1 2 3 4 5 6; 7 8 9 10 11 12; 13 14 15 16 17 18; 19 20 21 22 23 24]  
   W = LnL.liftedBasis(X, 2, 2, [1,2])
   @test round.(W, digits=6) ≈ D
end