#!/usr/bin/env python
# coding: utf-8

"""
L3 2025 Molecular dynamics -- proposed to the students
N-particle system with Jennard-Jones interactions
velocity Verlet algorithm
No boundaries
"""

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# In[2]:

DIM=2
Natom = 64
d0 = 1.2                              # initial distance between atoms
latticeSide= int(Natom**.5+.99)       # unit lattice Z^2
L = (latticeSide+1)*d0                # Box is [-L,L]^DIM; used in plot
Ratom = 0.3                           # radius of atom in the animated plot
vini = 0.5                            # initial velocity magnitude
h = 0.01                              # time step
itmax= 1001                           # total iterations
fastSteps= 5

# In[3]:

def forces(fr, posi):          
    fr[:,:] = 0.0
    for i in range( 0, Natom-1 ):
        for j in range(i + 1, Natom):
            dr = posi[i]-posi[j]           # relative position of particles
            r2 = dr.dot(dr)
            invr6  =  r2**-3               
            fij = (48./r2*(invr6 - 0.5)*invr6)*dr # LJ force
            fr[i] += fij
            fr[j] -= fij
    return fr

# In[4]:

### energies ###
def Ekinetic(vel):
    # code to be completed: calculate the total kinetik energy
    return 0.

def Epotential(posi):
    # code to be completed: calculate the total potential energy
    return 0.
### energies end ###

# In[8]:

def veloverlet(h, nsteps):
    global vel, posi, fr
    for t in range(nsteps):
        vel += 0.5*h*fr
        posi += h*vel                     # Velocity Verlet algorithm 
        fr = forces(fr, posi)
        vel += 0.5*h*fr

# In[9]:

def animate(i):
    global Ekpm
    if i: veloverlet(h, fastSteps)
    for a in range(0,Natom):
        atom[a].set_data(posi[a])   # plot atoms at new positions
    currtime = i*fastSteps*h
    ttime.append(currtime)
    ep= Epotential(posi)/Natom
    ek= Ekinetic(vel)/Natom
    Ekpm= np.append(Ekpm, [[ek,ep,ek+ep]], axis=0)
    for k in range(3):
        Ecurve[k].set_data(ttime, Ekpm[:,k])
    if currtime%2<h :
        print(f"{i}\tt= {currtime},  \tEkpm= {Ekpm[-1]}" )
    return atom + Ecurve

# In[10]:

fr = np.zeros((Natom,DIM))            # force on each atom
posi= np.zeros((Natom,DIM))           # positions

# initial positions on lattice
for i in range (Natom):
    for k in range(DIM):
        posi[i,k]= (i//latticeSide**k) % latticeSide
        posi[i,k]= (posi[i,k] - (latticeSide-1)*0.5 )*d0

# initial velocities, random from normal distribution
vel = vini*np.random.standard_normal((Natom,DIM))

ttime=[]
Ekpm= np.empty((0,3))   # Kinetic, potential and mechanical energies

xmax=1.2*L  # graph window boundary
plt.style.use('dark_background')
fig = plt.figure( figsize=(8,9))
fig.canvas.set_window_title('Lennard-Jones')
gs = plt.GridSpec(3, 1)
ax = fig.add_subplot(gs[:2, 0], xlim=(-xmax,xmax), ylim=(-xmax,xmax), aspect="equal")
atom= plt.plot(np.empty((0,Natom)), np.empty((0,Natom)), marker='o', ms=Ratom*300/xmax) # markersize in pts

axE = fig.add_subplot(gs[-1, 0], xlim=(0,itmax*h),ylim=(-3,1))
Ecurve = axE.plot(ttime, Ekpm)
plt.xlabel("t")
plt.ylabel("energy")
plt.legend(["Ec","Ep","Em"])
plt.tight_layout()

fr = forces(fr, posi)

ani = animation.FuncAnimation(fig, animate, frames=itmax//fastSteps+1, blit=True, interval=1, repeat=False)  # interval: delay in ms; default=200
#writervideo = animation.PillowWriter(fps=60)
#ani.save('output.gif', writer=writervideo)

plt.show()

print(f"end\tt= {ttime[-1]},  \tEkpm= {Ekpm[-1]}")
