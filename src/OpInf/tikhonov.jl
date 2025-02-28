"""
    tikhonov(b::AbstractArray, A::AbstractArray, Γ::AbstractMatrix, tol::Real;
        tol_flag::Bool=false, use_gpu::Bool=false, use_backslash::Bool=true)

Tikhonov regression to solve the operator inference problem.

## Features
- This function solves the regression problem using the Tikhonov regularization method.
- The function uses the LinearSolve.jl package to solve the regression problem.
- The function uses a manual SVD-based truncation if the singular values are below the tolerance `tol`.
  To enable this feature, set `tol_flag=true`. This is generally not recommended.
- If the direct solve fails due to memory issues, it switches to the iterative approach (Krylov GMRES).
- The function also supports GPU acceleration using CUDA.jl (Windows/Linux) or Metal.jl (Apple M series).
  To enable GPU acceleration, set `use_gpu=true`.
- The function also supports using the backslash operator for the regression solve (default: true).
  To enable this feature, set `use_backslash=true`.

## Arguments
- `b::AbstractArray`: right hand side of the regression problem 
- `A::AbstractArray`: left hand side of the regression problem 
- `Γ::AbstractMatrix`: Tikhonov matrix 
- `tol::Real`: tolerance for the singular values 
- `tol_flag::Bool`: flag for the tolerance (not recommended)
- `use_gpu::Bool`: flag for GPU acceleration
- `use_backslash::Bool`: flag for using the backslash operator

## Returns
- regression solution
"""
function tikhonov(b::AbstractArray, A::AbstractArray, Γ::AbstractMatrix, tol::Real;
                   tol_flag::Bool=false, use_gpu::Bool=false, use_backslash::Bool=true)
    # If the tolerance flag is set, perform SVD-based truncation where the singular 
    # values are below the tolerance level are truncated to zero and the rest are 
    # filled with zeros. This is a manual way to truncate the singular values.
    let 
        if tol_flag
            try
                # Define the key quantities
                M = A' * A + Γ     # (desired norm: || Γ^(1/2)*O ||_2)
                bhat = A' * b

                # Perform SVD-based truncation if singular values are below tol
                M_svd = svd(M)
                sing_idx = findfirst(M_svd.S .< tol)
                if sing_idx !== nothing
                    @warn "Rank deficient, rank = $(sing_idx), tol = $(M_svd.S[sing_idx])."
                    invSV = [1 ./ M_svd.S[1:sing_idx-1]; zeros(length(M_svd.S[sing_idx:end]))]
                    pinvM = M_svd.Vt' * Diagonal(invSV) * M_svd.U'
                    return pinvM * bhat
                else
                    @info "No singular values below the threshold. Fall back to standard solve."
                end
            catch e
                if isa(e, OutOfMemoryError)
                    @warn string("OutOfMemory encountered when `with_tol=true`. Switching to backslash methods vector. ", 
                                 "Truncation based on singular values will no longer be performed.")
                else
                    rethrow(e)
                end
            end 
        end
    end

    # Tikhonov regularization with augumented matrix
    # [A; Γ^(1/2)] * O = [b; 0]
    # O = ([A; Γ^(1/2)]^⊤ [A; Γ^(1/2)])^(-1) [A; Γ^(1/2)] [b; 0]
    Γsq = sqrt.(Γ)
    if use_gpu  # GPU
        if Sys.isapple()
            @info "GPU computation requested on macOS. Using Metal.jl."
            # Safety net: Ensure that a Metal device (expected for M series) is available
            metal_devs = Metal.devices()
            has_compatible = any(dev -> occursin("Apple M", string(dev)), metal_devs)
            @assert has_compatible "Metal.jl is only available for Apple M series GPUs."
            # Construct the augmented matrix with the Tikhonov matrix
            Atilde = Metal.CuArray(vcat(A, Γsq))
            btilde = Metal.CuArray(vcat(b, zeros(size(Γsq, 1), size(b, 2))))
        else
            if CUDA.has_cuda()
                @info "GPU computation requested. Using CUDA.jl."
                # Construct the augmented matrix with the Tikhonov matrix
                Atilde = CUDA.CuArray(vcat(A, Γsq))
                btilde = CUDA.CuArray(vcat(b, zeros(size(Γsq, 1), size(b, 2))))
            else
                @warn "CUDA GPU not available on this machine. Falling back to CPU"
                use_gpu = false
            end
        end
    else  # CPU
        Atilde = vcat(A, Γsq)
        btilde = vcat(b, zeros(size(Γsq, 1), size(b, 2)))
    end

    # Solve using the backslash operator
    # Works for CUDA/Metal as well
    if use_backslash || use_gpu
        try 
            if use_gpu
                O = Atilde \ btilde  # Operator matrix solution
                return Array(O)
            else
                return Atilde \ btilde  # Operator matrix solution
            end
        catch e 
            if isa(e, OutOfMemoryError)
                @warn "OutOfMemory with backslash least squares solve. Switching to LinearSolve.jl approach."
            elseif isa(e,  SparseArrays.CHOLMOD.CHOLMODException)
                @warn "Sparse array CHOLMOD encountered out of memory. Switching to LinearSolve.jl approach."
            else
                rethrow(e)
            end
            @assert !use_gpu "Disable `use_gpu` to switch to LinearSolve.jl approach."
        end
    end

    # Solve using LinearSolve.jl 
    O = similar(A, size(A, 2), size(b, 2))
    try
        # Try solving the least squares problem directly:
        # Finds O such that D*O ≈ Rt.
        # ATTENTION: LinearSolve.jl works for only vector right-hand side
        # so we need to solve for each column of Rt separately in a loop.
        ls = nothing
        for i in axes(btilde, 2)  
            if i == 1
                prob = LinearSolve.LinearProblem(Atilde, view(btilde, :, i))
                ls = LinearSolve.init(prob)
            else # reuse the linear problem
                ls.b .= view(btilde, :, i)
            end
            sol = LinearSolve.solve(ls)
            O[:,i] .= sol.u
        end
    catch e
        if isa(e, OutOfMemoryError) || isa(e,  SparseArrays.CHOLMOD.CHOLMODException)
            @warn "OutOfMemory in direct least squares solve. Switching to memory-efficient vector version."
            # Solve normal equations: (D' * D + Γ) x = D' * Rt.
            n = size(A, 2)
            op = let  # construct a linear operator to reduce memory usage
                f = (u,p,t) -> Atilde' * (Atilde * u)
                f = (du,u,p,t) -> (mul!(du,Atilde,u); du .= Atilde' * du)
                SciMLOperators.FunctionOperator(f, spzeros(n), spzeros(n))
            end
            ls2 = nothing
            for i in axes(btilde, 2)
                if i == 1
                    prob2 = LinearSolve.LinearProblem(op, view(btilde, :, i))
                    ls2 = LinearSolve.init(prob2)
                else
                    ls2.b .= view(btilde, :, i)
                end
                sol2 = LinearSolve.solve(ls2, LinearSolve.KrylovJL_GMRES())
                O[:,i] .= sol2.u
            end
        else
            rethrow(e)
        end
    end
    return O  # Operator matrix solution
end


"""
    tikhonov_matrix!(Γ::AbstractArray, dims::Dict, options::AbstractOption)

Construct the Tikhonov matrix

## Arguments
- `Γ::AbstractArray`: Tikhonov matrix (pass by reference)
- `options::AbstractOption`: options for the operator inference set by the user

## Returns
- `Γ`: Tikhonov matrix (pass by reference)
"""
function tikhonov_matrix!(Γ::AbstractArray, dims::AbstractArray, operator_symbols::AbstractArray, 
                         λ::TikhonovParameter)
    si = 1
    for (d, symbol) in zip(dims, operator_symbols)
        symbol_str  = string(symbol)
        if (length(symbol_str) >= 2) && ('A' in symbol_str)
            Γ[si:si+d-1] .= getproperty(λ, Symbol(symbol_str[1:2]))
        else
            Γ[si:si+d-1] .= getproperty(λ, symbol)
        end
        si += d
    end
end


"""
This is the old version of the tikhonov function. It is kept here for reference.
Will be archived in the future.
"""
# function tikhonov(b::AbstractArray, A::AbstractArray, Γ::AbstractMatrix, tol::Real; flag::Bool=false)
#     if flag
#         # Ag = A' * A + Γ' * Γ  # This is if || Γ*O ||_F is desired
#         Ag = A' * A + Γ         # This is if || Γ^{1/2}*O ||_2 is desired
#         Ag_svd = svd(Ag)
#         sing_idx = findfirst(Ag_svd.S .< tol)

#         # If singular values are nearly singular, truncate at a certain threshold
#         # and fill in the rest with zeros
#         if sing_idx !== nothing
#             @warn "Rank difficient, rank = $(sing_idx), tol = $(Ag_svd.S[sing_idx]).\n"
#             foo = [1 ./ Ag_svd.S[1:sing_idx-1]; zeros(length(Ag_svd.S[sing_idx:end]))]
#             bar = Ag_svd.Vt' * Diagonal(foo) * Ag_svd.U'
#             return bar * (A' * b)
#         else
#             @info "No singular values below the threshold. Fall back to standard solve."
#         end
#     end

#     Γsq = sqrt.(Γ)
#     Atilde = vcat(A, Γsq)
#     btilde = vcat(b, zeros(size(Γsq, 1), size(b, 2)))
#     return Atilde \ btilde
#     # return (A' * A + Γ) \ (A' * b)    # This is if || Γ^{1/2}*O ||_2 is desired

#     # return (A' * A + Γ' * Γ) \ (A' * b)  # This is if || Γ*O ||_F is desired
# end