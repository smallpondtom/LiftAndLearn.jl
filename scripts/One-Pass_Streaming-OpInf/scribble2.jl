using LinearAlgebra

##

m, n = 4, 7
X = rand(m,n)
dt = 1e-2 
Xdot = (X[:,2:end] - X[:,1:end-1]) / dt

##
# Xcopy = copy(X)
# X = X[:,2:end]

##
U, S, V = svd(X)
r = 2
Ur = U[:,1:r]
Sr = S[1:r]
Vr = V[:,1:r]

##
E = zeros(n, n-1)
for i in 1:n, j in 1:n-1
    if i == j 
        E[i, j] = -1.0 / dt
    elseif i == (j + 1)
        E[i, j] = 1.0 / dt
    end
end
Xdot2 = Xcopy * E

##
Xhatdot = Ur' * Xdot

# ##
# dP = Ur' * Xdot * Vr
# F = zeros(r,r)
# for i in 1:r
#     for j in 1:r
#         if i == j
#             F[i,j] = 0.0
#         else
#             F[i,j] = 1 / (Sr[j]^2 - Sr[i]^2)
#         end
#     end
# end
# dΩudt = F .* (dP * Diagonal(Sr) + Diagonal(Sr) * dP') 
# dSdt = I(r) .* dP
# dΩvdt = F .* (Diagonal(Sr) * dP + dP' * Diagonal(Sr)) 

# ##
# dDdt = dΩudt * Diagonal(Sr) + dSdt + Diagonal(Sr) * dΩvdt

# ##
# foo = dDdt * Vr' 

##
foo = Diagonal(Sr) * Vr' * E

##
println(norm(foo - Xhatdot) / norm(Xhatdot))

