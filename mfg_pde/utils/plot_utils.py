import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def myplot3d(X,Y,Z, title="Surface Plot"):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel('x')
    ax.set_ylabel('time')
    ax.set_title(title)
    ax.view_init(40, -135)
    plt.show()

def plot_convergence(iterations_run, l2disturel_u, l2disturel_m, solver_name="Solver"): # Modified to pass relative errors directly
    iterSpace = np.arange(1, iterations_run + 1)

    plt.figure()
    plt.semilogy(iterSpace, l2disturel_u) # Plot relative error for U
    plt.xlabel('Iteration')
    plt.ylabel('$||u_{new}-u_{old}||_{rel}$')
    plt.title(f'Convergence of U ({solver_name})')
    plt.grid(True)
    # Removed: plt.savefig(f'{prefix}_{solver_name}_conv_u.pdf')
    plt.show()

    plt.figure()
    plt.semilogy(iterSpace, l2disturel_m) # Plot relative error for M
    plt.xlabel('Iteration')
    plt.ylabel('$||m_{new}-m_{old}||_{rel}$')
    plt.title(f'Convergence of M ({solver_name})')
    plt.grid(True)
    # Removed: plt.savefig(f'{prefix}_{solver_name}_conv_m.pdf')
    plt.show()

def plot_results(problem, u, m, solver_name="Solver", prefix=None): # Modified to not require prefix
    # Subsample for plotting if desired
    kx = 2 # Example subsampling
    kt = 5 # Example subsampling
    xSpacecut = problem.xSpace[::kx] #
    tSpacecut = problem.tSpace[::kt] #
    ucut = u[::kt, ::kx] #
    mcut = m[::kt, ::kx] #

    # Plot U
    myplot3d(xSpacecut, tSpacecut, ucut, title=f'Evolution of U ({solver_name})') #
    # Removed: plt.savefig(f'{prefix}_{solver_name}_u.pdf')

    # Plot M
    myplot3d(xSpacecut, tSpacecut, mcut, title=f'Evolution of M ({solver_name})') #
    # Removed: plt.savefig(f'{prefix}_{solver_name}_m.pdf')

    # Add other plots as needed (final distributions, mid-time slices, mass conservation)
    plt.figure()
    plt.plot(problem.xSpace, m[-1,:]) #
    plt.xlabel('x')
    plt.ylabel('m(T,x)')
    plt.title(f'Final Density m(T,x) ({solver_name})') #
    plt.grid(True)
    # Removed: plt.savefig(f'{prefix}_{solver_name}_m_final.pdf')
    plt.show()

    plt.figure()
    mtot = np.sum(m * problem.Dx, axis=1) #
    plt.plot(problem.tSpace, mtot) #
    plt.xlabel('t')
    plt.ylabel('Total Mass $\\int m(t)$')
    plt.title(f'Total Mass ({solver_name})') #
    plt.ylim(min(0.9, np.min(mtot)*0.98 if mtot.size > 0 else 0.9), max(1.1, np.max(mtot)*1.02 if mtot.size > 0 else 1.1)) # Adjusted ylim for better view, added checks for empty mtot
    plt.grid(True)
    # Removed: plt.savefig(f'{prefix}_{solver_name}_mass.pdf')
    plt.show()