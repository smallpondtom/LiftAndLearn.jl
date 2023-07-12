import numpy as np
import matplotlib.pyplot as plt

# problem parameters
L  = 100
nx = 1024
dx = L/nx

dt = 0.05
T = 1000
nt = np.round(T/dt).astype(int)

# meshes
t = np.arange(stop=T+dt,step=dt)    # time mesh
x = np.arange(stop=L,step=dx)       # space mesh
k = np.arange(-nx/2,nx/2,1)         # wave numbers

# define operators in Fourier space on linear and nonlinear terms
A = (2*np.pi*k/L)**2 - (2*np.pi*k/L)**4
F = -0.5*(1j*2*np.pi*k/L)

# define KSE integrator
def kse(u0):
    u  = np.zeros((nx,nt+1))
    uh = np.zeros((nx,nt+1),dtype=complex)

    u[:,0]  = u0
    uh[:,0] = 1/nx*np.fft.fftshift(np.fft.fft(u[:,0]))

    for i in range(nt):
        u2h = 1/nx*np.fft.fftshift(np.fft.fft(u[:,i]**2))
        if i == 0:
            uh[:,i+1] =1/(1-0.5*dt*A)*((1+0.5*dt*A)*uh[:,i] + 1.0*dt*F*u2h)
        else:
            uh[:,i+1] =1/(1-0.5*dt*A)*((1+0.5*dt*A)*uh[:,i] + 1.5*dt*F*u2h-0.5*dt*F*u2hlast)
        u[:,i+1]  = np.real(nx*np.fft.ifft(np.fft.ifftshift(uh[:,i+1])))
        u2hlast = u2h.copy()
    return u

u0 = np.cos((2*np.pi*x)/L) + 0.1*np.cos((4*np.pi*x)/L)
u = kse(u0)

fig,ax = plt.subplots(figsize=(4,3))
xx,tt=np.meshgrid(x,t)
cs = ax.contourf(xx,tt,u.T,cmap="BrBG")
fig.colorbar(cs)
plt.savefig('kurasiva/xt.pdf')
plt.close("all")