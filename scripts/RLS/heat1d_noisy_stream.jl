# ##############################
# ## Streaming-OpInf with noise
# ##############################
# # Construct batches of the training data
# batchsize = 500
# M = size(X,2)÷batchsize
# Xhat_batch = batchify(Vr' * X, batchsize)
# U_batch = batchify(U, batchsize)
# Y_batch = batchify(Y', batchsize)
# R_batch = batchify(R', batchsize)

# # Define the noise with zero mean and variance
# σ = 1e-2
# Random.seed!(1234)
# WN(v,sz) = sqrt(v) * Random.randn(sz...)
# covmat(e) = cov(e * e')

# ## Initialize the stream
# stream = LnL.Streaming_InferOp(options)
# noise = WN(σ, size(R_batch[1]))
# noise_output = WN(σ, size(Y_batch[1]))
# # WARNING: Don't forget to add noise to your data matrix
# D_k = stream.init!(stream, Xhat_batch[1], U_batch[1], Y_batch[1] + noise_output, R_batch[1] + noise, 
#                     covmat(noise), covmat(noise_output))

# ## Stream all at once
# # WARNING: Don't forget to add noise to your data matrix
# noise = [WN(σ, size(R_batch[i])) for i in 2:M]
# noise_output = [WN(σ, size(Y_batch[i])) for i in 2:M]
# R_batch_noisy = [R_batch[i] + noise[i-1] for i in 2:M]
# Y_batch_noisy = [Y_batch[i] + noise_output[i-1] for i in 2:M]

# stream.stream!(stream, Xhat_batch[2:end], U_batch[2:end], R_batch_noisy, [covmat(n) for n in noise])
# stream.stream_output!(stream, Xhat_batch[2:end], Y_batch_noisy, [covmat(n) for n in noise_output])

# ##
# op_stream = stream.unpack_operators(stream)

# ##################
# ## Error Analysis
# ##################
# rel_state_err = zeros(r, 3)
# rel_output_err = zeros(r, 3)

# for i = 1:r
#     Vrr = Vr[:, 1:i]

#     # Integrate the intrusive model
#     Xint = LnL.backwardEuler(op_int.A[1:i, 1:i], op_int.B[1:i, :], Uf, heat1d.t, Vrr' * heat1d.IC)
#     Yint = op_int.C[1:1, 1:i] * Xint

#     # Integrate the inferred model
#     Xinf = LnL.backwardEuler(op_inf.A[1:i, 1:i], op_inf.B[1:i, :], Uf, heat1d.t, Vrr' * heat1d.IC)
#     Yinf = op_inf.C[1:1, 1:i] * Xinf

#     # Integrate the streaming inferred model
#     Xstr = LnL.backwardEuler(op_stream.A[1:i, 1:i], op_stream.B[1:i, :], Uf, heat1d.t, Vrr' * heat1d.IC)
#     Ystr = op_stream.C[1:1, 1:i] * Xstr

#     # Compute relative state errors
#     rel_state_err[i, 1] = LnL.compStateError(Xf, Xint, Vrr)
#     rel_state_err[i, 2] = LnL.compStateError(Xf, Xinf, Vrr)
#     rel_state_err[i, 3] = LnL.compStateError(Xf, Xstr, Vrr)

#     # Compute relative output errors
#     rel_output_err[i, 1] = LnL.compOutputError(Yf, Yint)
#     rel_output_err[i, 2] = LnL.compOutputError(Yf, Yinf)
#     rel_output_err[i, 3] = LnL.compOutputError(Yf, Ystr)
# end

# ## Plot
# fig5 = Figure(fontsize=20, size=(1200,600))
# ax1 = Axis(fig5[1, 1], xlabel="r", ylabel="Relative Error", title="Relative State Error with Noise", yscale=log10)
# scatterlines!(ax1, 1:r, rel_state_err[:, 1], label="Intrusive")
# scatterlines!(ax1, 1:r, rel_state_err[:, 2], label="OpInf")
# scatterlines!(ax1, 1:r, rel_state_err[:, 3], label="Streaming-OpInf")
# ax2 = Axis(fig5[1, 2], xlabel="r", ylabel="Relative Error", title="Relative Output Error with Noise", yscale=log10)
# l1 = scatterlines!(ax2, 1:r, rel_output_err[:, 1], label="Intrusive")
# l2 = scatterlines!(ax2, 1:r, rel_output_err[:, 2], label="OpInf")
# l3 = scatterlines!(ax2, 1:r, rel_output_err[:, 3], label="Streaming-OpInf")
# Legend(fig5[2, 1:2], [l1, l2, l3], ["Intrusive", "OpInf", "Streaming-OpInf"],
#         orientation=:horizontal, halign=:center, tellwidth=false, tellheight=true)
# display(fig5)


# ###########################
# ## Error per stream update
# ###########################
# # Initialize the stream
# M = size(X,2)÷batchsize
# r_select = [5, 10, 15]
# rel_state_err_stream = zeros(M, length(r_select))
# rel_output_err_stream = zeros(M, length(r_select))

# stream = LnL.Streaming_InferOp(options)
# for (j,ri) in enumerate(r_select)
#     Qs = covmat(WN(σ, batchsize))
#     Qo = covmat(WN(σ, batchsize))
#     for i in 1:M
#         if i == 1
#             D_k = stream.init!(stream, Xhat_batch[i], U_batch[i], Y_batch[i], R_batch[i], Qs, Qo)
#         else 
#             stream.stream!(stream, Xhat_batch[i], U_batch[i], R_batch[i], Qs)
#             stream.stream_output!(stream, Xhat_batch[i], Y_batch[i], Qo)
#         end
#         op_stream = stream.unpack_operators(stream)

#         # Integrate the streaming inferred model
#         Xstr = LnL.backwardEuler(op_stream.A[1:ri, 1:ri], op_stream.B[1:ri, :], Uf, heat1d.t, Vr[:,1:ri]' * heat1d.IC)
#         Ystr = op_stream.C[1:1, 1:ri] * Xstr

#         # Compute relative state errors
#         rel_state_err_stream[i,j] = LnL.compStateError(Xf, Xstr, Vr[:,1:ri])

#         # Compute relative output errors
#         rel_output_err_stream[i,j] = LnL.compOutputError(Yf, Ystr)
#     end
# end
# ## Plot
# fig6 = Figure(fontsize=20, size=(1200,600))
# ax1 = Axis(fig6[1, 1], xlabel="Stream Update", ylabel="Relative Error", title="Relative State Error", yscale=log10, xticks=1:M)
# ax2 = Axis(fig6[1, 2], xlabel="Stream Update", ylabel="Relative Error", title="Relative Output Error", yscale=log10, xticks=1:M)
# lines_ = []
# labels_ = []
# for (j,ri) in enumerate(r_select)
#     scatterlines!(ax1, 1:M, rel_state_err_stream[:,j])
#     l = scatterlines!(ax2, 1:M, rel_output_err_stream[:,j])
#     push!(lines_, l)
#     push!(labels_, "r = $ri")
# end
# Legend(fig6[2, 1:2], lines_, labels_, orientation=:horizontal, halign=:center, tellwidth=false, tellheight=true)
#
