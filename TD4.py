# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 20:10:19 2025

@author: taher
"""

"""
L3 2025 Molecular dynamics -- version for lecturers; DO NOT DISTRIBUTE
N-particle system with Jennard-Jones interactions
velocity Verlet algorithm
No boundaries
"""
"""
Benchmarks
N=100, t=10, fastSteps= 5, cutoff, ps_time= 5.1, 5.1, 5.3
N=100, t=10, fastSteps= 5, no cutoff, ps_time= 8.3, 8.4
"""

import numpy as np
import matplotlib.pyplot as plt
from time import process_time

# In[2]:

DIM = 2
Natom = 64
d0 = 1.2                              # initial distance between atoms
latticeSide = int(Natom**.5 + .99)    # unit lattice Z^2
L = (latticeSide + 1) * d0            # Box is [-L, L]^DIM; used in plot
Ratom = 0.3                           # radius of atom in the animated plot
r2cut = 3.**2                         # interaction cutoff distance
vini = 0.5                            # initial velocity magnitude
h = 0.01                              # time step
itmax = 10001                          # total iterations
fastSteps = 5
gamma = 0.1                           # Friction coefficient
temp = 1.0                            # Target temperature

# In[3]:
def forces(fr, posi):          
    fr[:,:] = 0.0
    for i in range(0, Natom - 1):
        for j in range(i + 1, Natom):
            dr = posi[i] - posi[j]           # relative position of particles
            r2 = dr.dot(dr)
            if r2 > r2cut: continue
            invr6 = r2**-3               
            fij = (48. / r2 * (invr6 - 0.5) * invr6) * dr  # LJ force
            fr[i] += fij
            fr[j] -= fij
    return fr

# In[4]:
def Ekinetic(vel):
    Ek = np.sum(vel**2) * 0.5
    return Ek

vshift = (r2cut**-3 - 1.0) * r2cut**-3
def Epotential(posi):
    Ep = 0.0
    for i in range(0, Natom - 1):
        for j in range(i + 1, Natom):
            dr = posi[i] - posi[j]
            r2 = dr.dot(dr)
            if r2 > r2cut: continue
            invr6 = r2**-3               
            Ep += (invr6 - 1.0) * invr6
    return Ep * 4.

# In[8]:
def langevin_thermostat(vel, gamma, temp, h):
    kB = 1.0  # Boltzmann constant (normalized to 1 for simplicity)
    sigma = np.sqrt(2 * gamma * kB * temp *h)  # Noise strength
    noise = sigma * np.random.standard_normal(vel.shape)  # Gaussian noise
    vel = vel * (1 - gamma * h) + noise  # Langevin dynamics
    return vel

# In[9]:
def veloverlet(h, nsteps, gamma=0.1, temp=1.0):
    global vel, posi, fr
    for t in range(nsteps):
        vel += 0.5 * h * fr  # Half-step velocity update
        posi += h * vel      # Update positions
        fr = forces(fr, posi)  # Update forces
        vel += 0.5 * h * fr  # Full-step velocity update
        vel = langevin_thermostat(vel, gamma, temp, h)  # Apply Langevin thermostat

# In[10]: Initialization
fr = np.zeros((Natom, DIM))            # Force on each atom
posi = np.zeros((Natom, DIM))          # Positions
vel = vini * np.random.standard_normal((Natom, DIM))  # Initial velocities
ttime = []                             # Time tracker
energies = []                          # Energies tracker
temperatures = []                      # Temperature tracker

# Initial positions on lattice
for i in range(Natom):
    for k in range(DIM):
        posi[i, k] = (i // latticeSide**k) % latticeSide
        posi[i, k] = (posi[i, k] - (latticeSide - 1) * 0.5) * d0
# Set the origin at the center of mass
posi -= posi.sum(axis=0) / Natom

# Fix the center of mass velocity to zero
vel -= vel.sum(axis=0) / Natom

# Compute initial forces
fr = forces(fr, posi)

# In[11]: Simulation Loop
start_pstime = process_time()
for i in range(itmax // fastSteps + 1):
    veloverlet(h, fastSteps, gamma, temp)
    
    # Compute energies and temperature
    ek = Ekinetic(vel) / Natom
    ep = Epotential(posi) / Natom
    em = ek + ep
    energies.append((ek, ep, em))
    temp_inst = np.mean(np.sum(vel**2, axis=1)) / DIM
    temperatures.append(temp_inst)
    
    ttime.append(i * fastSteps * h)

print(f"Simulation completed in {process_time() - start_pstime:.2f} seconds")

# In[12]: Plotting
energies = np.array(energies)
plt.figure(figsize=(12, 6))

# Plot energies
plt.subplot(1, 2, 1)
plt.plot(ttime, energies[:, 0], label='Kinetic Energy')
plt.plot(ttime, energies[:, 1], label='Potential Energy')
plt.plot(ttime, energies[:, 2], label='Total Energy')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('Energy vs Time')
plt.legend()

# Plot temperature
plt.subplot(1, 2, 2)
plt.plot(ttime, temperatures, label='Temperature', color='orange')
plt.axhline(temp, color='red', linestyle='--', label='Target Temperature')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Temperature vs Time')
plt.legend()

plt.tight_layout()
plt.show()
