export pod


"""
$(SIGNATURES)

Perform intrusive model reduction by Galerkin projection of the full-order operators onto
a Proper Orthogonal Decomposition (POD) basis.

The reduced operators are ``\\hat{\\mathbf{A}} = \\mathbf{V}_r^\\top\\mathbf{A}\\mathbf{V}_r``,
``\\hat{\\mathbf{B}} = \\mathbf{V}_r^\\top\\mathbf{B}``, ``\\hat{\\mathbf{C}} = \\mathbf{C}\\mathbf{V}_r``,
``\\hat{\\mathbf{K}} = \\mathbf{V}_r^\\top\\mathbf{K}``,
``\\hat{\\mathbf{A}}_k = \\mathbf{V}_r^\\top\\mathbf{A}_k(\\mathbf{V}_r\\otimes\\cdots\\otimes\\mathbf{V}_r)``
for the polynomial terms and ``\\hat{\\mathbf{N}} = \\mathbf{V}_r^\\top\\mathbf{N}\\mathbf{V}_r``
for the bilinear term. Only the terms listed in `sys_struct` are projected.

## Arguments
- `op::Operators`: full-order operators of the target system.
- `Vr::AbstractArray`: POD basis of size ``n \\times r``.
- `sys_struct::SystemStructure`: polynomial structure of the system (which operators to project).

## Keyword Arguments
- `nonredundant_operators::Bool=true`: project the non-redundant polynomial operators
  (`A2u`, `A3u`, ...) of `op`; if `false`, project the redundant ones (`A2`, `A3`, ...).
  Both forms are set in the result.

## Returns
- `op_new::Operators`: the reduced operators.

## Notes
- Limited to state terms up to 4th order, the ``\\mathbf{B}`` input matrix, the
  ``\\mathbf{C}`` output matrix, bilinear state-input coupling and a constant term
  ``\\mathbf{K}``. For multiple inputs `op.N` must be an ``n \\times n \\times m`` array.
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
