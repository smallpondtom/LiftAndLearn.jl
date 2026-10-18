import numpy as np


def get_z_profile(U):
    return U.mean(axis=(0, 1))


def get_utau(U, V, Z, zind=2):
    Uh = (U[:, :, zind] ** 2 + V[:, :, zind] ** 2) ** 0.5
    ubar = Uh.mean()
    return solve_utau(ubar, Z[zind] + Z[1] / 2)


def log_law(u_tau, zf):
    kappa = 0.384
    B = 4.27
    nu = 8e-6
    return u_tau * (np.log(zf * u_tau / nu) / kappa + B)


def log_law_deriv(u_tau, zf):
    kappa = 0.384
    B = 4.27
    nu = 8e-6
    return (1 + np.log(zf * u_tau / nu)) / kappa + B


def solve_utau(u, zf):
    prev = 0.0
    curr = 0.0414
    iters = 0
    while np.abs(curr - prev) > 1e-12 and iters < 25:
        prev = curr
        curr = prev - (log_law(prev, zf) - u) / log_law_deriv(prev, zf)
        iters += 1
    if iters == 25:
        print("ERROR in utau computation")
    return curr


def get_qois(snapshot, Z):
    return {
        "zprof": get_z_profile(snapshot[0]),
        "utau": get_utau(snapshot[0], snapshot[1], Z),
    }