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
from matplotlib.animation import FuncAnimation
from time import process_time

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

fig,ax = plt.subplots(1,2,figsize=(10,6))

xmax=1.2*L  # graph window boundary
ttime=[]
def animate(i):
    global Ekpm
    ax[0].clear()
    ax[1].clear()
    currtime = i*fastSteps*h
    ttime.append(currtime)
    veloverlet(h,fastSteps)
    for a in range(Natom):
        dot = plt.Circle(posi[a],radius= Ratom*20/xmax)
        ax[0].add_patch(dot)

    ax[0].set_xlim(-xmax,xmax)
    ax[0].set_ylim(-xmax,xmax)
    ax[0].set_aspect('equal')
    
    ep= Epotential(posi)/Natom
    ek= Ekinetic(vel)/Natom
    Ekpm= np.append(Ekpm, [[ek,ep,ek+ep]], axis=0)
    
    for k in range(3):
        ax[1].plot(ttime,Ekpm[:,k],lw=0.9)
        ax[1].legend(["Ec","Ep","Em"])
        
    ax[1].set_xlim(0,itmax*h)
    ax[1].set_ylim(-3,1)
    
    if currtime%2<h :
        print(f"{i}\tt= {currtime},  \tEkpm= {Ekpm[-1]}", end='' )
        print(f",  \tprocess_time: {process_time()-start_pstime:.6}" )
    
    
    
        
        

# In[10]:

fr = np.zeros((Natom,DIM))            # force on each atom
posi= np.zeros((Natom,DIM))           # positions

# initial positions on lattice
for i in range (Natom):
    for k in range(DIM):
        posi[i,k]= (i//latticeSide**k) % latticeSide
        posi[i,k]= (posi[i,k] - (latticeSide-1)*0.5 )*d0
# Set the origin at the center of mass
posi -= posi.sum(axis=0)/Natom

# initial velocities, random from normal distribution
vel = vini*np.random.standard_normal((Natom,DIM))
vel -= vel.sum(axis=0)/Natom     # fix the center of mass


Ekpm= np.empty((0,3))   # Kinetic, potential and mechanical energies
start_pstime = process_time()


ani =FuncAnimation(fig, animate, frames=itmax//fastSteps+1, blit=False, interval=1.1, repeat=False)  # interval: delay in ms; default=200
plt.tight_layout()
plt.show()


print(f"end\tt= {ttime[-1]},  \tEkpm= {Ekpm[-1]}", end='')
print(f",  \tprocess_time: {process_time()-start_pstime:.6}" )
print(L)

for tt in [0,-1]:
    print(ttime[tt], "\tenergies:", Ekpm[tt])

# # %%
