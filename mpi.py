from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as pl
from room import room

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def receive_print(data):
    print("P[", rank, "] received data =", data)


def send_print(data):
    print(("P[", rank, "] sent data =", data))


def solve(nx, ny, theta=0.7, maxiter=100, tol=1e-10, plot=True):
    # nx & ny = number of interior unknowns per unit length
    temp_wall = 15
    temp_heater = 40

    dx, dy = 1/(nx + 1), 1/(ny + 1)

    if rank == 0:
        # Room 1
        room0 = room(nx + 1, ny, dx, dy)
        room0.add_boundary('D', 'left', temp_heater)
        room0.add_boundary('D', 'top', temp_wall)
        room0.add_boundary('D', 'bottom', temp_wall)

        # Send initial guess of room 1 to room 2
        temp_gamma_0_initial = np.ones(ny)*temp_wall

        comm.send(temp_gamma_0_initial, dest=1)
        send_print(temp_gamma_0_initial)

        # Solving iteration
        for k in range(maxiter):
            # Receive flux from room 2
            flux_gamma_0 = comm.recv(source=1)
            receive_print(flux_gamma_0)

            # Find room temperature
            flux_gamma_0 = flux_gamma_0[:ny]
            room0.add_boundary('N', 'right', -flux_gamma_0)
            room0.solve()
            temp_gamma_0_new = room0.get_solution(where='right')

            # Send the new boundary temperature to room 2
            comm.send(temp_gamma_0_new, dest=1)
            send_print(temp_gamma_0_new)

    if rank == 1:
        # Room 2
        room1 = room(nx, 2*ny+1, dx, dy)
        room1.add_boundary('D', 'top', temp_heater)
        room1.add_boundary('D', 'bottom', temp_window)

        updates1, updates2 = [], []
        for k in range(maxiter):
            temp_gamma_0 = comm.recv(source=0)
            temp_gamma_2 = comm.recv(source=2)

            room1.add_boundary('D', 'left',
                               np.pad(temp_gamma_0, (0, ny + 1), mode='constant', constant_values=temp_wall))  # pad up values to fit whole wall
            room1.add_boundary('D', 'right',
                               np.pad(temp_gamma_2, (ny + 1, 0), mode='constant', constant_values=temp_wall))  # pad up values to fit whole wall

            # step 2: solve room2
            room1.solve()

            # step 3: get heat fluxes
            flux_gamma_0 = room1.get_flux('left')
            flux_gamma_2 = room1.get_flux('right')

            # Send the heatflux to other two rooms
            comm.send(flux_gamma_0, dest=0)
            send_print(flux_gamma_0)
            comm.send(flux_gamma_2, dest=2)
            send_print(flux_gamma_2)

            temp_gamma_0_new = comm.recv(source=0)
            temp_gamma_2_new = comm.recv(source=2)

            # step8: relaxation and update solutions
            temp_gamma_1_old = (1-theta)*temp_gamma_1_old + \
                theta*temp_gamma_0_new
            temp_gamma_2_old = (1-theta)*temp_gamma_2_old + \
                theta*temp_gamma_2_new

            # step9: compute updates
            # TODO: possibly use different norm, e.g., discrete L2 norm?
            updates1.append(np.linalg.norm(
                temp_gamma_1_old - temp_gamma_0_new, 2))
            updates2.append(np.linalg.norm(
                temp_gamma_2_old - temp_gamma_2_new, 2))

            # step10: check if iteration has converged via updates
            if (updates1[-1] < tol) and (updates2[-1] < tol):
                break

    if rank == 2:
        # Room 3

        room2 = room(nx + 1, ny, dx, dy)
        room2.add_boundary('D', 'right', temp_heater)
        room2.add_boundary('D', 'top', temp_wall)
        room2.add_boundary('D', 'bottom', temp_wall)

        # Send initial guess of room 3 to room 2
        temp_gamma_2_initial = np.ones(ny)*temp_wall

        comm.send(temp_gamma_2_initial, dest=1)
        send_print(temp_gamma_2_initial)

        # Solving iteration
        for k in range(maxiter):
            # Receive flux from room 2
            flux_gamma_2 = comm.recv(source=1)
            receive_print(flux_gamma_2)

            # Find room temperature
            flux_gamma_2 = flux_gamma_2[-ny:]
            room2.add_boundary('N', 'left', -flux_gamma_2)
            room2.solve()
            temp_gamma_2_new = room2.get_solution(where='right')

            # Send the new boundary temperature to room 2
            comm.send(temp_gamma_2_new, dest=1)
            send_print(temp_gamma_2_new)
