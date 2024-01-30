'''
    TRANSPORT EQUATION
    It describes the transport of a quantity u in a one-dimensional space.
    It  expresses how the temporal change of u is influenced by its movement 
    through space at a constant velocity b. 

    The equation is du/dt + b*(du/dx) = f(x,t), u=u(x,t)

    This file is a parallel code for the serial code of the implementation
    of this equation. In this parallelized version, the domain is decomposed 
    by the master process and then distributed to the worker/tasks processes. 

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
from mpi4py import MPI
from matplotlib import cm

# ---
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
    plt.suptitle( 'Transport Equation solution - Parallel' )
    plt.show()

    # LEVELS IMPLEMENTATION
    levels = np.linspace(u.min(), u.max(), 7)
    fig2, ax2 = plt.subplots()
    ax2.contourf(X, T, u, levels=levels)
    
    ax2.set_xlabel( 'Length x' )
    ax2.set_ylabel( 'Time t' )
    plt.suptitle( 'Transport Equation solution - Parallel' )
    plt.show()

    return None

if __name__ == "__main__":

    # Defining variables of the equation
    L = int(20)       # Lenght of the rod - [0,L] x-domain
    T = int(30)       # Time interval - [0,T] t-domain
    N_x = int(160000) # Points' number of x-domain
    N_t = int(1000)   # Points' number of t-domain
    b = float(0.0030) # Speed of transport
    # Spatial and temporal spacing
    dx = L / (N_x - 1)
    dt = T / (N_t - 1)


    ''' Initialization MPI '''

    array_size = N_x
    master = 0

    # Checking if MPI is already initialized
    if not MPI.Is_initialized():
        MPI.Init()
    
    comm = MPI.COMM_WORLD
    num_tasks = comm.Get_size()
    task_id = comm.Get_rank()
    print(f"MPI Task {task_id} has started...")
    slaves = num_tasks - 1
    chuncksize = int(array_size / slaves)
    tag1 = 1 # Tag used to send the offset values of each task
    tag2 = 2 # Tag used to send the decomposition of the array + neighbours
    tag3 = 3 # Tag used to send message between tasks

    # ---
    # MASTER CODE - domain's decomposition and distribution of work to processes 
    if task_id == master:

        print("***** PARALLEL TRANSPORT EQUATION *****")

        # Checking if the slaves is a divider of the array_size - quit if not
        if array_size % slaves != 0:
            print(f"Quitting. Number of MPI slaves must be a multiple of {array_size}")
            comm.Abort(0)

        print(f"Starting parallelization with {slaves} tasks.")

        # Initializing bidimensional array containing the respective values of u
    
        # Inizializing the space and time vectors - creates two vectors
        # containing the number desidered of values in a given interval
        x = np.linspace(0, L, N_x)
        t = np.linspace(0, T, N_t)

        # Creation of bidimensional array containing the information about 
        # the quantity u - initialized at 0
        u = np.zeros( (N_t, N_x), dtype=np.float64, order='C' )

        # Inserting the initial condition - Cauchy condition
        # At time 0, the rod is completely cold
        u[0, :] = 0

        # Inserting boundary conditions - Dirichlet conditions
        # The ends of the rod are at a constant temperature: cold
        u[:, 0] = 0
        u[:, N_x-1] = 0

        print("Array initialized...")
        start_time = MPI.Wtime()

        # Sending each task its portion of the array
        offset = 0 
        for id in range(1, slaves+1):

            # Telling each slaves who its neighours are - communication needed
            # Inserting value -1 if there is no neighbour
            
            # Left-task - first task does not have a left-neighbour
            left = np.array([id - 1], dtype=np.int64, order='C') if id != 1 else np.array([-1], dtype=np.int64, order='C')

            # Right-task - last task does not have a right-neighbour
            right = np.array([id + 1], dtype=np.int64, order='C') if id != slaves else np.array([-1], dtype=np.int64, order='C')

            offset = np.array([offset], dtype=np.int64, order='C')
            # Sending startup information to each slave
            comm.Send(offset, dest=id, tag=tag1)
            comm.Send(left, dest=id, tag=tag2)
            comm.Send(right, dest=id, tag=tag2)
            
            offset = offset.item()
            for row in range(1, N_t):
                row_to_send = u[row, offset:offset+chuncksize]
                comm.Send(row_to_send, dest=id, tag=tag2)

            print(f"Sent to task {id} offset {offset}")
            print(f"left={left} - right={right}")
            
            offset += chuncksize
        

        # Waiting to receive results from each task
        for id in range(1, slaves+1):
            offset = np.array([0], dtype=np.int64, order='C')
            comm.Recv(offset, source=id, tag=tag1)
            offset = offset.item()
            for row in range(1, N_t):
                row_recv_slave = np.array([0]*chuncksize, dtype=np.float64, order='C')
                comm.Recv(row_recv_slave, source=id, tag=tag2)
                u[row, offset:offset+chuncksize] = row_recv_slave
        
        # Final outputs and visualization of results
        print("Parallelization done and generating rappresentations...")
        end_time = MPI.Wtime()
        elapsed_time = end_time - start_time
        print(elapsed_time)
        
        # visualizations_u(x,t,u)

        if not MPI.Is_finalized():
            MPI.Finalize()

    # END MASTER CODE
        
    # SLAVES CODE
    if task_id != master:

        print(f"Task {task_id} working...")
        # Receiving information sent by master
        offset = np.array([0], dtype=np.int64, order='C')
        left = np.array([0], dtype=np.int64, order='C')
        right = np.array([0], dtype=np.int64, order='C')
        comm.Recv(offset, source=master, tag=tag1)
        comm.Recv(left, source=master, tag=tag2)
        comm.Recv(right, source=master, tag=tag2)
        print(f"Received: {offset} - {left} - {right}")
        
        # Extracting the numeric value
        offset = offset.item()
        left = left.item() 
        right = right.item() 

        # Initializing everything - including borders conditions
        u = np.zeros( (N_t, N_x), dtype=np.float64, order='C' )
        
        # Receiving portion of the domain
        for row in range(1, N_t):
            row_recv = np.array([0]*chuncksize, dtype=np.float64, order='C')
            comm.Recv(row_recv, source=master, tag=tag2)
            u[row, offset:offset+chuncksize] = row_recv
        
        print("Received rows...")

        # Checking if task operates over borders elements. 
        # If so, changing the variables: first and last columns must be
        # equal to zero. It can cause raised value due to the calculation
        # of solution.
        start = offset if offset != 0 else 1
        end = min( offset+chuncksize, N_x ) - 1
        print(f"task={task_id} - start={start} - end={end}")

        x = np.linspace(0, L, N_x)
        t = np.linspace(0, T, N_t)

        # Begin doing work. Must communicate border information with neighbours.
        # If it is first or last task, it has only one neighbour.
        print(f"Task {task_id} received work. Begin work...")
        for n in range(1, N_t-1):

            # Sending information to neighbours - checking if any
            if right != -1:
                el_send = u[n, end]
                comm.Send(el_send, dest=right, tag=tag3)
            
            if left != -1:
                el_recv = np.array([0], dtype=np.float64, order='C')
                comm.Recv(el_recv, source=left, tag=tag3)
                u[n, start-1] = el_recv

            for i in range(start, end+1):
                
                u[n+1, i] = u[n, i] \
                - b * (dt/dx) * (u[n, i]- u[n, (i-1) % N_x]) \
                + dt * heat_function( x[i], t[n] )
        
        # Sendig to master the work done
        offset = np.array([offset], dtype=np.int64, order='C')
        comm.Send(offset, dest=master, tag=tag1)
        for row in range(1, N_t):
            row_to_master = u[row, start:end+1]
            comm.Send(row_to_master, dest=master, tag=tag2)

        if not MPI.Is_finalized():
            MPI.Finalize()

    # END SLAVES' WORK