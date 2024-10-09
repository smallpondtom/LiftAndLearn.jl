function quad2d(t, x, u)
    return [-2*x[1] + x[1]*x[2]; -x[2] + x[1]*x[2]] + [0; -0.01] * u[1]
end

function lotka_volterra(t, x, u)
    return [-1.5*x[1] - 1.5*x[2] - 0.5*x[1]^2 - 0.5*x[1]*x[2]; 3*x[1] + x[1]*x[2]] + [0.01; -0.02] * u[1]
end

function nonlinear_pendulum(t, x, u)
    return [x[2]; -sin(x[1])] + [0.01; -0.02] * u[1]
end