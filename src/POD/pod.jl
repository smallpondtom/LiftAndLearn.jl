export pod


"""
$(SIGNATURES)

Perform intrusive model reduction using Proper Orthogonal Decomposition (POD)

# Arguments
- `op`: operators of the target system (A, B, C, F/H, N, K)
- `Vr`: POD basis
- `options`: options for the operator inference

# Return
- `op_new`: new operator projected onto the basis
"""
function pod(op::operators, Vr::Union{BlockDiagonal, VecOrMat, AbstractArray}, options::AbstractOption)
    Ahat = Vr' * op.A * Vr
    Bhat = Vr' * op.B
    Chat = op.C * Vr
    Khat = Vr' * op.K

    op_new = operators(
        A=Matrix(Ahat), B=Matrix(Bhat[:, :]), C=Matrix(Chat[:, :]), K=Matrix(Khat[:, :])
    )

    n = size(op.A, 1)
    r = size(Vr, 2)

    if options.system.is_quad  # Add the Fhat term here
        if op.F != 0
            Ln = elimat(n)
            Dr = dupmat(r)
            VV = Vr ⊗ Vr
            Fhat = Vr' * op.F * Ln * VV * Dr
            op_new.F = Matrix(Fhat)

            if op.H == 0
                Hhat = F2Hs(Fhat)
                op_new.H = Matrix(Hhat)
            end
        end

        if op.H != 0  # Add the Hhat term here
            Hhat = Vr' * op.H * (Vr ⊗ Vr)
            op_new.H = Matrix(Hhat)

            if op.F == 0
                Fhat = H2F(Hhat)
                op_new.F = Matrix(Fhat)
            end
        end
    end

    # Add the Nhat term here
    if options.system.is_bilin
        sz = size(op.N)
        # if typeof(op.N) == Vector{Matrix}
        if length(sz) == 3
            p = sz[3]
            # Nhat = Vector{Matrix{Float64}}(undef, length(op.N))
            Nhat = Array{Float64}(undef, (r,r,p))
            # i = 0
            # for Ni in op.N  # Assuming that op.N is a vector of matrices
            for i in 1:p
                tmp = Vr' * op.N[:,:,i] * Vr
                # Nhat[i+=1] = Matrix(tmp[:, :])
                Nhat[:,:,i] = Matrix(tmp[:, :])
            end
            op_new.N = Nhat
        else
            Nhat = Vr' * op.N * Vr
            op_new.N = Matrix(Nhat[:, :])
        end
    end

    # Cubic term
    if options.system.is_cubic
        if op.E != 0
            Ln3 = elimat3(n)
            Dr3 = dupmat3(r)
            Ehat = Vr' * op.E * Ln3 * (Vr ⊗ Vr ⊗ Vr) * Dr3
            op_new.E = Matrix(Ehat)

            if op.G == 0
                Ghat = LnL.E2Gs(Ehat)
                op_new.G = Matrix(Ghat)
            end
        end
        if op.G != 0
            Ghat = Vr' * op.G * (Vr ⊗ Vr ⊗ Vr)
            op_new.G = Matrix(Ghat)

            if op.E != 0
                Ehat = LnL.G2E(Ghat)
                op_new.E = Matrix(Ehat)
            end
        end
    end

    return op_new
end
