'''
    TRANSPORT EQUATION
    It describes the transport of a quantity u in a one-dimensional space.
    It  expresses how the temporal change of u is influenced by its movement 
    through space at a constant velocity b. 

    The equation is du/dt + b*(du/dx) = f(u,t), u=u(x,t)

    This file is a serial code for this equation that uses a numeric method
    to solve it: partial derivatives have been approximated 
    using the finite differences forward.

    ***************************************************************************

    CASE STUDY: We are analyzing as a real case the heating process 
    of a one-dimensional rod, where the radius is considered negligible, 
    and the length is arbitrary. 
    The thermal diffusivity alpha is set to 1 and is constant. 
    Boundary conditions are imposed to keep the two ends of the rod 
    at a constant temperature.

'''

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import cm 

# ---
# Non-homogeneous case

# Defining a Heat Function that represents the term function
def heat_function( space : np.ndarray,
                   time : np.ndarray ) -> np.ndarray:
    '''
    Using a fundamental solution of the heat equation.
    '''
    return ((np.pi*4*time)**(-0.5/2)) ** np.exp((-space**2)/(4*time))

# Defining a function to visualize results
def visualizations_u(x : np.ndarray,
                     t : np.ndarray,
                     u : np.ndarray) -> None:
    '''
    Function used to implement the visualizations. 
    '''
    # Visualization of the quantity u 
    # plt.contour(x, t, u, cmap='Spectral')
    # plt.imshow( u, 
    #            cmap='Spectral', 
    #            origin='lower',
    #            aspect='auto',
    #            vmin=u.max(),
    #            vmax=u.min())
    
    X, T = np.meshgrid(x,t)

    plt.style.use('_mpl-gallery')
    
    # 3D IMPLEMENTATION
    fig1, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, T, u, vmin=u.min(), cmap=cm.Reds)

    ax.set_xlabel( 'Length x' )
    ax.set_ylabel( 'Time t' )
    ax.set_zlabel( ' Heat u ')
    plt.suptitle( 'Transport Equation solution' )
    plt.show()

    # LEVELS IMPLEMENTATION
    levels = np.linspace(u.min(), u.max(), 7)
    fig2, ax2 = plt.subplots()
    ax2.contourf(X, T, u, levels=levels)
    
    ax2.set_xlabel( 'Length x' )
    ax2.set_ylabel( 'Time t' )
    plt.suptitle( 'Transport Equation solution' )
    plt.show()

    return None


if __name__ == "__main__":

    print("******** SERIAL TRANSPORT EQUATION ********")
    print(datetime.now())
    start_time = datetime.now()

    # Defining variables of the equation
    L = 20        # Lenght of the rod - [0,L] x-domain
    T = 30        # Time interval - [0,T] t-domain
    N_x = 160000  # Points' number of x-domain
    N_t = 1000    # Points' number of t-domain
    b = 0.0030    # Speed of transport

    # Spatial and temporal spacing
    dx = L / (N_x - 1)
    dt = T / (N_t - 1)

    # Inizializing the space and time vectors - creates two vectors
    # containing the number desidered of values in a given interval
    x = np.linspace(0, L, N_x)
    t = np.linspace(0, T, N_t)

    # Creation of bidimensional array containing the information about 
    # the quantity u - initialized at 0
    u = np.zeros( (N_t, N_x) )

    # Inserting the initial condition - Cauchy condition
    # At time 0, the rod is completely cold
    u[0, :] = 0

    # Inserting boundary conditions - Dirichlet conditions
    # The ends of the rod are at a constant temperature: cold
    u[:, 0] = 0
    u[:, N_x-1] = 0

    # Numeric schema finite differences forward
    for n in range(1, N_t-1):
        for i in range(1, N_x-1):
            u[n+1, i] = u[n, i] \
            - b * (dt/dx) * (u[n, i] - u[n, (i-1) % N_x]) \
            + dt * heat_function(x[i], t[n])
    
    print(datetime.now()) 
    end_time = datetime.now()
    interval = end_time - start_time
    print(f"Solution generate in {interval}")
    # ---
    # Visualization of results
    visualizations_u(x,t,u)