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

    return op_new
end
