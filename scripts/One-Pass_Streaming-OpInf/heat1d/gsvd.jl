##
using LinearAlgebra

## A * B^{-1}
m, n = 8, 10
A = rand(m,n)
B = rand(n,n)
C = vcat(A,B)
k = rank(C)

P, _, _ = svd(C)
P1 = P[:, 1:k]
P11 = P1[1:m, :]
P21 = P1[m+1:end, :]

U, Sa, W = svd(P11)
V, Sb = qr(P21 * W)
Sb = diag(Sb)
id = Sa.^2 + Sb.^2
println("Check if Sa^2 + Sb^2 = I: ", norm(Diagonal(id) - 1.0I))

ABinv = A / B 
Utrue, Strue, Vtrue = svd(ABinv)

ABinv_approx = U * (Diagonal(Sb) \ Diagonal(Sa)) * V'

err = norm(ABinv - ABinv_approx) / norm(ABinv)
println("Relative error: ", err)


## A^{-1} * B
m, n = 8, 10
A = rand(m,m)
B = rand(m,n)
C = hcat(A,B)
k = rank(C)

_, _, Q = svd(C)
Q1 = Q[:, 1:k]
Q11 = Q1[1:m, :]
Q21 = Q1[m+1:end, :]

if m <= n
    U, Sa, W = svd(Q11)
    V, Sb = qr(Q21 * W)
    Sb = diag(Sb)
else
    V, Sb, W = svd(Q21)
    U, Sa = qr(Q11 * W)
    U = U[:, 1:n]
    Sa = diag(Sa)
end
id = Sa.^2 + Sb.^2
println("Check if Sa^2 + Sb^2 = I: ", norm(Diagonal(id) - 1.0I))

AinvB = A \ B
Utrue, Strue, Vtrue = svd(AinvB)

AinvB_approx = U * (Diagonal(Sa) \ Diagonal(Sb)) * V'

err = norm(AinvB - AinvB_approx) / norm(AinvB)
println("Relative error: ", err)

## A^{-1} * B  (The better one)
m, n = 18, 19
A = rand(m,m)
B = rand(m,n)
C = vcat(A',B')
k = rank(C)

P, _, _ = svd(C)
P1 = P[:, 1:k]
P11 = P1[1:m, :]
P21 = P1[m+1:end, :]

V, Sb, W = svd(P21)
U, Sa = qr(P11 * W)
Sa = diag(Sa)
id = Sa.^2 + Sb.^2
println("Check if Sa^2 + Sb^2 = I: ", norm(Diagonal(id) - 1.0I))

AinvB = A \ B
Utrue, Strue, Vtrue = svd(AinvB)

AinvB_approx = U * (Diagonal(Sa) \ Diagonal(Sb)) * V'

err = norm(AinvB - AinvB_approx) / norm(AinvB)
println("Relative error: ", err)
