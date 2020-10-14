#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 12:00:38 2020
"""

import numpy as np
import matplotlib.pyplot as plt
from room import room
from mpi4py import MPI 


comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

if size < 3:
    raise ValueError('invalid number of processors, needs to be 3')
    
    
def DN_iteration(nx, ny, theta = 0.7, maxiter = 100, tol = 1e-10, pltot = True):
    ## nx & ny = number of interior unknowns per unit length
    temp_wall = 15
    temp_heater = 40
    temp_window = 5

    dx, dy = 1/(nx + 1), 1/(ny + 1)
    
    ## set up rooms and their static boundary conditions
    if rank == 0:
        room1 = room(nx + 1, ny , dx, dy)
        room1.add_boundary(temp_heater, 'D', 'left')   
        room1.add_boundary(temp_wall, 'D', 'top')   
        room1.add_boundary(temp_wall, 'D', 'bottom')
        room1.f_checkpoint()
        room1.set_boundary_type('N', 'right')
    
    if rank == 1:
        room2 = room(nx, 2*ny+1, dx, dy)
        room2.add_boundary(temp_heater,'D', 'top')
        room2.add_boundary(temp_window,'D', 'bottom')
        room2.f_checkpoint()
        ## set initial condition for boundaries in room2
        temp_gamma_1_old = np.ones(2*ny+1)*temp_wall
        temp_gamma_2_old = np.ones(2*ny+1)*temp_wall
        
    if rank == 2:    
        room3 = room(nx + 1, ny, dx, dy)
        room3.add_boundary(temp_heater, 'D', 'right')    
        room3.add_boundary(temp_wall, 'D', 'top')    
        room3.add_boundary(temp_wall, 'D', 'bottom')
        room3.f_checkpoint()
        room3.set_boundary_type('N', 'left')
    

    
    updates1, updates2 = [], []
    for k in range(maxiter):
        if rank == 1:
            ## step1: add dirichlet boundaries for room2
            room2.f_reset()
            room2.add_boundary(temp_gamma_1_old, 'D', 'left')
            room2.add_boundary(temp_gamma_2_old, 'D', 'right')
        
            ## step 2: solve room2
            room2.solve()

            ## step 3: get heat fluxes
            flux_gamma_1 = room2.get_flux(temp_gamma_1_old, 'left')
            flux_gamma_2 = room2.get_flux(temp_gamma_2_old, 'right')
        
            ## step 4: cut to length
            flux_gamma_1 = -flux_gamma_1[:ny]
            flux_gamma_2 = -flux_gamma_2[-ny:]
            
            comm.send(flux_gamma_1, dest=0)
            comm.send(flux_gamma_2, dest=2)

        
        ## step5 + 6: add fluxes as boundaries to room1 and room3 and solve + get new boundary values
        if rank == 0:
            room1.f_reset()
            flux_gamma_1 = comm.recv(source=1)
            room1.add_boundary(-flux_gamma_1,'N', 'right')
            temp_gamma_1_new = room1.solve(where = 'right')
            comm.send(temp_gamma_1_new, dest=1)
            
        if rank == 2:
            room3.f_reset()
            flux_gamma_2 = comm.recv(source=1)
            room3.add_boundary(-flux_gamma_2,'N', 'left')
            temp_gamma_2_new = room3.solve(where = 'left')
            comm.send(temp_gamma_2_new, dest=1)

        
        if rank == 1:
            ## step7: relaxation and update solutions
            temp_gamma_1_new = comm.recv(source=0)
            temp_gamma_2_new = comm.recv(source=2)

            L = np.concatenate((temp_gamma_1_old[:ny+1], temp_gamma_1_new))
            temp_gamma_1_new = (1-theta)*temp_gamma_1_old + theta*L
            R=np.concatenate((temp_gamma_2_new,temp_gamma_2_old[ny:]))
            temp_gamma_2_new = (1-theta)*temp_gamma_2_old + theta*R
        
            ## step8: compute updates
            ## TODO: possibly use different norm, e.g., discrete L2 norm?
            updates1.append(np.linalg.norm(temp_gamma_1_old - temp_gamma_1_new, 2))
            updates2.append(np.linalg.norm(temp_gamma_2_old - temp_gamma_2_new, 2))
        
            ## step9: check if iteration has converged via updates
            converged = (updates1[-1] < tol) and (updates2[-1] < tol)
            comm.send(converged, dest=0)
            comm.send(converged, dest=2)
            if converged:
                break
            else:
                temp_gamma_1_old = temp_gamma_1_new
                temp_gamma_2_old = temp_gamma_2_new
    
        if rank == 0 or rank == 2:
            converged = comm.recv(source=1)
            if converged:
                break

    if pltot:
        if rank == 0:
            comm.send(room1.get_solution(), dest = 1)
        if rank == 2:
            comm.send(room3.get_solution(), dest = 1)
        if rank == 1:      
            ## currently assuming
            if dx != dy:
                raise ValueError('pltotting currently not impltemented for dx != dy')
            A = np.zeros((3*(nx+1) + 1, 2*(nx+1) + 1))
            n = nx + 1
        
            # Standard walls 
            A[:n+1, 0] = temp_wall
            A[:n+1, n] = temp_wall
            A[n, n:] = temp_wall
            A[2*n:, -1] = temp_wall
            A[2*n:,n] = temp_wall
            A[2*n, :n] = temp_wall
            # Window front
            A[n+1:2*n, 0] = temp_window
            # Heaters
            A[0, 1:n] = temp_heater
            A[-1, n+1:-1] = temp_heater
            A[n+1:2*n, -1] = temp_heater
            ## rooms
            # TODO: make sure these are in the correct shapes
            A[1:n+1, 1:n] = comm.recv(source=0)
            A[n+1:2*n, 1:-1] = np.flip(room2.get_solution(),1)
            A[2*n:-1, n+1:-1] = comm.recv(source=2)
        
            ## boundaries
            A[n, 1:n] = temp_gamma_1_new[n:] # left
            A[2*n, n+1:-1] = temp_gamma_2_new[:n-1] # right
            
            plt.subplots(figsize = (8, 4))
            plt.pcolor(A.transpose(), cmap = 'RdBu_r', vmin = 5, vmax = 40)
            plt.title('Heat distribution')
            plt.colorbar()
            plt.axis([0, 3/dx+1, 0, 2/dx+1])
            plt.xticks([])
            plt.yticks([])
            plt.savefig('room_mpi.png')
            
            plt.figure()
            plt.semilogy(range(k+1), updates1, label = 'updates gamma1')
            plt.semilogy(range(k+1), updates2, label = 'updates gamma2')
            plt.xlabel('iteration')
            plt.ylabel('update')
            plt.grid(True, 'major')
            plt.legend()
            plt.savefig('updates.png')


if __name__ == "__main__":
    plt.close("all")
    
    DN_iteration(10, 10)