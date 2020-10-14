#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 12:00:38 2020
"""

import numpy as np
import matplotlib.pyplot as plt
from room import room

def DN_iteration(nx, ny, theta = 0.7, maxiter = 100, tol = 1e-10, pltot = True):
    ## nx & ny = number of interior unknowns per unit length
    temp_wall = 15
    temp_heater = 40
    temp_window = 5
    
    dx, dy = 1/(nx + 1), 1/(ny + 1)
    
    ## set up rooms and their static boundary conditions
    room1 = room(nx + 1, ny , dx, dy)
    room1.add_boundary(temp_heater, 'D', 'left')   
    room1.add_boundary(temp_wall, 'D', 'top')   
    room1.add_boundary(temp_wall, 'D', 'bottom')
    room1.f_checkpoint()
    room1.set_boundary_type('N', 'right')
    
    room2 = room(nx, 2*ny+1, dx, dy)
    room2.add_boundary(temp_heater,'D', 'top')
    room2.add_boundary(temp_window,'D', 'bottom')
    room2.f_checkpoint()
    
    room3 = room(nx + 1, ny, dx, dy)
    room3.add_boundary(temp_heater, 'D', 'right')    
    room3.add_boundary(temp_wall, 'D', 'top')    
    room3.add_boundary(temp_wall, 'D', 'bottom')
    room3.f_checkpoint()
    room3.set_boundary_type('N', 'left')
    
    ## set initial condition for boundaries in room2
    temp_gamma_1_old = np.ones(2*ny+1)*temp_wall
    temp_gamma_2_old = np.ones(2*ny+1)*temp_wall
    
    updates1, updates2 = [], []
    for k in range(maxiter):
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
        
        ## step5: add fluxes as boundaries to room1 and room3
        room1.f_reset()
        room1.add_boundary(-flux_gamma_1,'N', 'right')
        
        room3.f_reset()
        room3.add_boundary(-flux_gamma_2,'N', 'left')
        
        ## step6: solve room1 and room3 and get new boundary values
        temp_gamma_1_new = room1.solve(where = 'right')
        temp_gamma_2_new = room3.solve(where = 'left')
        
        ## step7: relaxation and update solutions
        L = np.concatenate((temp_gamma_1_old[:ny+1], temp_gamma_1_new))
        temp_gamma_1_new = (1-theta)*temp_gamma_1_old + theta*L
        R=np.concatenate((temp_gamma_2_new,temp_gamma_2_old[ny:]))
        temp_gamma_2_new = (1-theta)*temp_gamma_2_old + theta*R
        
        ## step8: compute updates
        ## TODO: possibly use different norm, e.g., discrete L2 norm?
        updates1.append(np.linalg.norm(temp_gamma_1_old - temp_gamma_1_new, 2))
        updates2.append(np.linalg.norm(temp_gamma_2_old - temp_gamma_2_new, 2))
        
        ## step9: check if iteration has converged via updates
        if (updates1[-1] < tol) and (updates2[-1] < tol):
            break
        else:
            temp_gamma_1_old = temp_gamma_1_new
            temp_gamma_2_old = temp_gamma_2_new
        
    if pltot: ## currently assuming
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
        A[1:n+1, 1:n] = room1.get_solution()
        A[n+1:2*n, 1:-1] = np.flip(room2.get_solution(),1)
        A[2*n:-1, n+1:-1] = room3.get_solution()
        
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
     
    return updates1, updates2, k+1

if __name__ == "__main__":
    plt.close("all")
    
    up1, up2, iters = DN_iteration(10, 10)
    
    plt.figure()
    plt.semilogy(range(iters), up1, label = 'updates gamma1')
    plt.semilogy(range(iters), up2, label = 'updates gamma2')
    plt.xlabel('iteration')
    plt.ylabel('update')
    plt.grid(True, 'major')
    plt.legend()