import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
import argparse

parser = argparse.ArgumentParser(description='2D Ising model in triangle lattice with equal number of up and down spins')
parser.add_argument('-im', action='store_true', help='Show the animation of the spins')
parser.add_argument('-calc_C_OH', action='store_true', help='Calculate the number of OH bonds')
parser.add_argument('-T', type=float, default=273, help='Temperature in K')
parser.add_argument('-r', type=int, default=20, help='Number of rows in the lattice')
parser.add_argument('-c', type=int, default=20, help='Number of columns in the lattice')
parser.add_argument('-nsteps', type=int, default=50000, help='Number of Monte Carlo steps')
parser.add_argument('-nstat', type=int, default=100, help='Number of steps showing the statistics')
parser.add_argument('-hext', type=float, default=0.0, help='External magnetic field')
parser.add_argument('-eps', type=float, default=1.14e-20/4, help='Interaction energy in J')
parser.add_argument('-run_nums', type=int, default=1, help='Number of runs')
args = parser.parse_args()

T = args.T
kb = 1.38064852e-23
# kB Temperature
kT = kb*T
# Beta
beta = 1/kT
# epsilon
eps = args.eps
# expernal field
h = args.hext

# row and column of the lattice
r, c = args.r, args.c

# Total number of spins
N = r*c

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
            ((i+1)%r, j)] if i % 2 == 1 \
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
    deltaE = -2*eps*(spins[xup, yup]*sum_neighbors(spins, xup, yup, r, c) \
                    + spins[xdown, ydown]*sum_neighbors(spins, xdown, ydown, r, c)) \
            -2*h*(spins[xup, yup] + spins[xdown, ydown])
    if (xdown, ydown) in find_neighbors(xup, yup, r, c):
        deltaE += 4*eps*spins[xup, yup]*spins[xdown, ydown]
    if np.random.rand() < np.exp(-beta*deltaE):
        spins[xup, yup] *= -1
        spins[xdown, ydown] *= -1
        energy += deltaE
        mag += spins[xup, yup] + spins[xdown, ydown]
    return spins, energy, mag

def hexagon_plot(c, r):
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
    return fig, ax, collection

nsteps = args.nsteps
nstat = args.nstat

t_energy = np.zeros(nsteps)
t_mag = np.zeros(nsteps)

energy = get_energy(spins,r,c)
mag = np.sum(spins,)

imflag = args.im
if imflag:
    fig, ax, collection = hexagon_plot(c, r)
    collection.set_array(spins[::-1].T.ravel())
    plt.draw()

print ()
print ("MC simulation")
print ("kT = %.2E h = %.2E J = %.2E" %(kT,h,eps))
print ("row = %d by col = %d" %(r,c))
print ("Nsteps = %d and Nstat = %d" %(nsteps,nstat))
print
print ("Step \t E \t <E> \t M \t <M>")
print ("%d \t %.4E \t %.4E \t %.4E \t %.4E" %(0, energy/N, energy/N, mag/N, mag/N))

def calc_C_OH(spins, r, c):
    # the number of spins up surrounding each spins up
    C_OH = 0
    for x, y in zip(*np.where(spins == 1)):
        for nx, ny in find_neighbors(x, y, r, c):
            if spins[nx, ny] == 1:
                C_OH += 1
    return C_OH/(r*c/2)

run_nums = args.run_nums
calc_C_OH_flag = args.calc_C_OH
C_OH_average = 0
for run in range(run_nums):
    print("Run %d" % (run+1))
    # MC simulation
    for step in range(nsteps):
        spins, energy, mag = monte_carlo(spins, energy, mag)
        t_energy[step] = energy
        t_mag[step] = mag
        if step % nstat == 0 and step > 0:
            if imflag: # update the plot
                collection.set_array(spins[::-1].T.ravel())
                fig.canvas.draw()
                plt.pause(0.01)
            print("%d \t %.4E \t %.4E \t %.4E \t %.4E" % (step, energy/N, np.mean(t_energy[:step])/N, t_mag[step]/N, np.mean(t_mag[:step])/N))
    else:
        if not imflag: # if not showing the animation, plot the final state
            fig, ax, collection = hexagon_plot(c, r)
            collection.set_array(spins[::-1].T.ravel())
        # final plot and save the figure
        plt.savefig("ising_%dK_run%d.png" % (T, run+1), dpi=300, bbox_inches='tight')
        plt.show()
        
    if calc_C_OH_flag:
        C_OH = calc_C_OH(spins, r, c)
        print("C_OH = %.3f" % C_OH)
        print()
        C_OH_average += C_OH/run_nums
if calc_C_OH_flag:
    print("C_OH_ave = %.3f" % C_OH_average)




