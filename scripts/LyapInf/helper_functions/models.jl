# Lotka-Volterra Predator-Prey model
function LotkaVolterra() 
    A = [-2.0 0.0; 0.0 -1.0]
    H = LnL.makeQuadOp(2, [(1,2,1), (1,2,2)], [1.0, 1.0])
    F = LnL.H2F(H)
    return LnL.operators(A=A, F=F, H=H)
end

# Van der Pol oscillator (cubic)
function VanDerPol_cubic(μ::Real)
    A = zeros(2,2)
    A[1,2] = -1.0
    A[2,1] = 1.0
    A[2,2] = -μ
    G = LnL.makeCubicOp(2, [(1,1,2,2)], [μ])
    # G = zeros(2,8)
    # G[2,2] = μ/3
    # G[2,3] = μ/3
    # G[2,5] = μ/3
    E = LnL.G2E(G)
    return LnL.operators(A=A, E=E, G=G)
end

# Van der Pol oscillator (quadratic)
function VanDerPol_quadratic()
    A = zeros(2,2)
    A[1,2] = 1.0
    A[2,1] = -1.0
    A[2,2] = -0.5
    H = LnL.makeQuadOp(2, [(1,2,2), (2,2,2)], [-0.5, -0.1])
    F = LnL.H2F(H)
    return LnL.operators(A=A, F=F, H=H)
end

# Stable 3D 
function Stable3D()
    A = -1.0I(3)
    H = LnL.makeQuadOp(3, [(1,2,2)], [1.0])
    G = LnL.makeCubicOp(3, [(2,3,3,1)], [1.0])
    return LnL.operators(A=A, H=H, G=G, F=LnL.H2F(H), E=LnL.G2E(G))
end


## ODE models
function lin_quad_model!(xdot, x, p, t)
    A, F = p[1], p[2]
    xdot .= A * x + F * (x ⊘ x)
end

function lin_cubic_model!(xdot, x, p, t)
    A, E = p[1], p[2]
    xdot .= A * x + E * ⊘(x,x,x)
end

function lin_quad_cubic_model!(xdot, x, p, t)
    A, F, E = p[1], p[2], p[3]
    xdot .= A * x + F * (x ⊘ x) + E * ⊘(x,x,x)
end