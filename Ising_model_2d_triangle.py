import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
#import numba

# kB Temperature
kT = 3.0
# Beta
beta = 1/kT
# epsilon
eps = 1.0
# expernal field
h = 0.0

# row and column of the lattice
r, c = 6, 4

# Total number of spins
N = r*c
print("%d x %d lattice with %d spins" % (r, c, N))

# Initial condition: half of the spins are up and half are down
spins = np.ones((r, c), dtype=int)
for index in np.random.choice(N, N//2, replace=False):
    spins[index//c, index%c] = -1

def find_neighbors(i, j, r, c):
    #! two types of neighbors: convex and concave
    return [((i-1)%r, (j-1)%c),
            ((i-1)%r, j),
            (i, (j-1)%c),
            (i, (j+1)%c),
            ((i+1)%r, (j-1)%c),
            ((i+1)%r, j)] if j % 2 == 1 \
        else [((i-1)%r, j),
                ((i-1)%r, (j+1)%c),
                (i, (j-1)%c),
                (i, (j+1)%c),
                ((i+1)%r, j),
                ((i+1)%r, (j+1)%c)]

def sum_neighbors(spins, i, j, r, c):
    return sum(spins[x, y] for x, y in find_neighbors(i, j, r, c))

def get_energy(spins,r,c):
    temp = 0
    for i in range(r):
        for j in range(c):
            temp += eps*spins[i, j]*sum_neighbors(spins, i, j, r, c)/2 + h*spins[i, j]
    return temp

def monte_carlo(spins, energy, mag):
    spinup = list(zip(*np.where(spins == 1)))
    xup, yup = spinup[np.random.randint(len(spinup))]
    spindown = list(zip(*np.where(spins == -1)))
    xdown, ydown = spindown[np.random.randint(len(spindown))]
    spins, energy, mag = flip(xup,yup,xdown,ydown,spins,energy,mag)
    return spins, energy, mag

def flip(xup,yup,xdown,ydown,spins,energy,mag):
    #print((xup, yup), (xdown, ydown))
    #print(spins[xup, yup], spins[xdown, ydown])
    deltaE = -2*eps*(spins[xup, yup]*sum_neighbors(spins, xup, yup, r, c) \
                    + spins[xdown, ydown]*sum_neighbors(spins, xdown, ydown, r, c)) \
            -2*h*(spins[xup, yup] + spins[xdown, ydown])
    #print(deltaE)
    if (xdown, ydown) in find_neighbors(xup, yup, r, c):
        deltaE += 4*eps*spins[xup, yup]*spins[xdown, ydown]
    #print(deltaE)
    if np.random.rand() < np.exp(-beta*deltaE):
        spins[xup, yup] *= -1
        spins[xdown, ydown] *= -1
        energy += deltaE
        mag += spins[xup, yup] + spins[xdown, ydown]
    return spins, energy, mag

nsteps = 5000
nstat = 20

t_energy = np.zeros(nsteps)
t_mag = np.zeros(nsteps)

energy = get_energy(spins,r,c)
mag = np.sum(spins,)

imflag = 1
if imflag:
    fig, ax = plt.subplots(1,1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-1.0, c)
    ax.set_ylim(-1.0, r)
    plt.setp(ax, xticks=[], yticks=[])
    patches = []
    for i, x in enumerate(range(c)):
        for j, y in enumerate(range(r)):
            point = [i-0.5, j] if j % 2 == 0 else [i, j]
            hexagon = RegularPolygon(point, numVertices=6, radius=0.6, edgecolor='k')
            patches.append(hexagon)
    collection = PatchCollection(patches, cmap='plasma')
    ax.add_collection(collection)
    collection.set_array(spins[::-1].T.ravel())
    plt.draw()

print ()
print ("MC simulation")
print ("kT = %.2f h = %.2f J = %.2f" %(kT,h,eps))
print ("row = %d by col = %d" %(r,c))
print ("Nsteps = %d and Nstat = %d" %(nsteps,nstat))
print
print ("Step \t E \t <E> \t M \t <M>")
print ("%d \t %.4f \t %.4f \t %.4f \t %.4f" %(0, energy/N, energy/N, mag/N, mag/N))

for step in range(nsteps):
    spins, energy, mag = monte_carlo(spins, energy, mag)
    t_energy[step] = energy
    t_mag[step] = mag
    if step % nstat == 0 and step > 0:
        if imflag:
            collection.set_array(spins[::-1].T.ravel())
            fig.canvas.draw()
            plt.pause(0.01)
            #plt.savefig("ising_%d.png" % step)
        print("%d \t %.4f \t %.4f \t %.4f \t %.4f" % (step, energy/N, np.mean(t_energy[:step])/N, t_mag[step]/N, np.mean(t_mag[:step])/N))






