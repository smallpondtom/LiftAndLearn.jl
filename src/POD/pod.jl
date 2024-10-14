export pod


"""
$(SIGNATURES)

Perform intrusive model reduction using Proper Orthogonal Decomposition (POD).
This implementation is liimted to
- state: up to 4th order
- input: only B matrix
- output: only C and D matrices
- state-input-coupling: bilinear 
- constant term: K matrix

# Arguments
- `op`: operators of the target system 
- `Vr`: POD basis
- `options`: options for the operator inference

# Return
- `op_new`: new operator projected onto the basis
"""
function pod(op::Operators, Vr::AbstractArray, sys_struct::SystemStructure; 
             nonredundant_operators::Bool=true)::Operators
    # New operator
    op_new = Operators()

    # Linear state operator
    state_struct = copy(sys_struct.state)
    if 1 in sys_struct.state
        op_new.A = Vr' * op.A * Vr

        if state_struct == 1
            state_struct = []
        else
            deleteat!(state_struct, findfirst(isequal(1), state_struct))
        end
    end

    # Linear input operator
    if 1 in sys_struct.control
        op_new.B = Vr' * op.B
    end

    # Linear output operator
    if 1 in sys_struct.output
        op_new.C = op.C * Vr
    end

    # Constant operator
    if !iszero(sys_struct.constant)
        op_new.K = Vr' * op.K
    end

    # Define the dimensions
    n, r = size(Vr)

    for i in state_struct
        if nonredundant_operators
            Aku = getfield(op, Symbol("A$(i)u"))
            Ln = elimat(n, i)
            Dr = dupmat(r, i)
            Akuhat = Vr' * Aku *  Ln * (Vr ⊗ i) * Dr
            setfield!(op_new, Symbol("A$(i)u"), Akuhat)
            setfield!(op_new, Symbol("A$(i)"), duplicate_symmetric(Akuhat, i))
        else
            Ak = getfield(op, Symbol("A$(i)"))
            Akhat = Vr' * Ak * (Vr ⊗ i)
            setfield!(op_new, Symbol("A$(i)"), Akhat)
            setfield!(op_new, Symbol("A$(i)u"), eliminate(Akhat, i))
        end
    end

    # # Quadratic term
    # if !iszero(op.A2)
    #     A2hat = Vr' * op.A2 * (Vr ⊗ Vr)
    #     op_new.A2 = A2hat
    #     op_new.A2u = eliminate(A2hat, 2)
    #     op_new.A2t = H2Q(A2hat)
    # else
    #     Ln = elimat(n, 2)
    #     Dr = dupmat(r, 2)
    #     A2uhat = Vr' * op.A2u * Ln * (Vr ⊗ Vr) * Dr
    #     op_new.A2u = A2uhat
    #     op_new.A2 = duplicate(A2uhat, 2)
    #     op_new.A2t = H2Q(op_new.A2)
    # end

    # # Cubic term
    # if !iszero(op.A3)
    #     A3hat = Vr' * op.A3 * (Vr ⊗ Vr ⊗ Vr)
    #     op_new.A3 = A3hat
    #     op_new.A3u = eliminate(A3hat, 3)
    # else
    #     Ln = elimat(n, 3)
    #     Dr = dupmat(r, 3)
    #     A3uhat = Vr' * op.A3u * Ln * (Vr ⊗ Vr ⊗ Vr) * Dr
    #     op_new.A3u = A3uhat
    #     op_new.A3 = duplicate(A3uhat, 3)
    # end

    # # Quartic term
    # if !iszero(op.A4)
    #     A4hat = Vr' * op.A4 * (Vr ⊗ Vr ⊗ Vr ⊗ Vr)
    #     op_new.A4 = A4hat
    #     op_new.A4u = eliminate(A4hat, 4)
    # else
    #     Ln = elimat(n, 4)
    #     Dr = dupmat(r, 4)
    #     A4uhat = Vr' * op.A4u * Ln * (Vr ⊗ Vr ⊗ Vr ⊗ Vr) * Dr
    #     op_new.A4u = A4uhat
    #     op_new.A4 = duplicate(A4uhat, 4)
    # end

    # Bilinear term
    if 1 in sys_struct.coupled_input
        sz = size(op.N)
        if length(sz) == 3
            p = sz[3]
            Nhat = Array{Float64}(undef, (r,r,p))
            for i in 1:p
                tmp = Vr' * op.N[:,:,i] * Vr
                Nhat[:,:,i] = tmp
            end
            op_new.N = Nhat
        else
            op_new.N = Vr' * op.N * Vr
        end
    end

    # if options.system.is_quad  # Add the Fhat term here
    #     if op.F != 0
    #         Ln = elimat(n)
    #         Dr = dupmat(r)
    #         VV = Vr ⊗ Vr
    #         Fhat = Vr' * op.F * Ln * VV * Dr
    #         op_new.F = Matrix(Fhat)

    #         if op.H == 0
    #             Hhat = F2Hs(Fhat)
    #             op_new.H = Matrix(Hhat)
    #         end
    #     end

    #     if op.H != 0  # Add the Hhat term here
    #         Hhat = Vr' * op.H * (Vr ⊗ Vr)
    #         op_new.H = Matrix(Hhat)

    #         if op.F == 0
    #             Fhat = H2F(Hhat)
    #             op_new.F = Matrix(Fhat)
    #         end
    #     end
    # end

    # # Add the Nhat term here
    # if options.system.is_bilin
    #     sz = size(op.N)
    #     # if typeof(op.N) == Vector{Matrix}
    #     if length(sz) == 3
    #         p = sz[3]
    #         # Nhat = Vector{Matrix{Float64}}(undef, length(op.N))
    #         Nhat = Array{Float64}(undef, (r,r,p))
    #         # i = 0
    #         # for Ni in op.N  # Assuming that op.N is a vector of matrices
    #         for i in 1:p
    #             tmp = Vr' * op.N[:,:,i] * Vr
    #             # Nhat[i+=1] = Matrix(tmp[:, :])
    #             Nhat[:,:,i] = Matrix(tmp[:, :])
    #         end
    #         op_new.N = Nhat
    #     else
    #         Nhat = Vr' * op.N * Vr
    #         op_new.N = Matrix(Nhat[:, :])
    #     end
    # end

    # Cubic term
    # if options.system.is_cubic
    #     if op.E != 0
    #         Ln3 = elimat3(n)
    #         Dr3 = dupmat3(r)
    #         Ehat = Vr' * op.E * Ln3 * (Vr ⊗ Vr ⊗ Vr) * Dr3
    #         op_new.E = Matrix(Ehat)

    #         if op.G == 0
    #             Ghat = E2Gs(Ehat)
    #             op_new.G = Matrix(Ghat)
    #         end
    #     end
    #     if op.G != 0
    #         Ghat = Vr' * op.G * (Vr ⊗ Vr ⊗ Vr)
    #         op_new.G = Matrix(Ghat)

    #         if op.E != 0
    #             Ehat = G2E(Ghat)
    #             op_new.E = Matrix(Ehat)
    #         end
    #     end
    # end

    return op_new
end
