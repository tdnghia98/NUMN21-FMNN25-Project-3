#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 12:00:38 2020
"""

import numpy as np
import matplotlib.pyplot as pl
from room import room

def DN_iteration(nx, ny, theta = 0.7, maxiter = 100, tol = 1e-10, plot = True):
    ## nx & ny = number of interior unknowns per unit length
    temp_wall = 15
    temp_heater = 40
    temp_window = 5
    
    dx, dy = 1/(nx + 1), 1/(ny + 1)
    
    ## set up rooms and their static boundary conditions
    room1 = room(nx + 1, ny , dx, dy)
    room1.add_boundary('D', 'left', temp_heater)    
    room1.add_boundary('D', 'top', temp_wall)    
    room1.add_boundary('D', 'bottom', temp_wall)
    
    room2 = room(nx, 2*ny+1, dx, dy)
    room2.add_boundary('D', 'top', temp_heater)
    room2.add_boundary('D', 'bottom', temp_window)
    
    room3 = room(nx + 1, ny, dx, dy)
    room3.add_boundary('D', 'right', temp_heater)    
    room3.add_boundary('D', 'top', temp_wall)    
    room3.add_boundary('D', 'bottom', temp_wall)
    
    
    ## set initial condition for boundaries in room2
    temp_gamma_1_old = np.ones(ny)*temp_wall
    temp_gamma_2_old = np.ones(ny)*temp_wall
    
    
    updates1, updates2 = [], []
    for k in range(maxiter):
        ## step1: add dirichlet boundaries for room2
        room2.add_boundary('D', 'left',
                           np.pad(temp_gamma_1_old, (0, ny + 1), mode = 'constant', constant_values = temp_wall)) ## pad up values to fit whole wall
        room2.add_boundary('D', 'right',
                           np.pad(temp_gamma_2_old, (ny + 1, 0), mode = 'constant', constant_values = temp_wall)) ## pad up values to fit whole wall
        
        ## step 2: solve room2
        room2.solve()
        
        ## step 3: get heat fluxes
        flux_gamma_1 = room2.get_flux('left')
        flux_gamma_2 = room2.get_flux('right')
        
        ## step 4: cut to length
        flux_gamma_1 = flux_gamma_1[:ny]
        flux_gamma_2 = flux_gamma_2[-ny:]
        
        ## step5: add fluxes as boundaries to room1 and room3
        room1.add_boundary('N', 'right', -flux_gamma_1)
        room3.add_boundary('N', 'left', -flux_gamma_2)
        
        ## step6: solve room1 and room3
        room1.solve()
        room3.solve()
        
        ## step7: get new boundary values
        temp_gamma_1_new = room1.get_solution(where = 'right')
        temp_gamma_2_new = room3.get_solution(where = 'left')
        
        ## step8: relaxation and update solutions
        temp_gamma_1_old = (1-theta)*temp_gamma_1_old + theta*temp_gamma_1_new
        temp_gamma_2_old = (1-theta)*temp_gamma_2_old + theta*temp_gamma_2_new
        
        ## step9: compute updates
        ## TODO: possibly use different norm, e.g., discrete L2 norm?
        updates1.append(np.linalg.norm(temp_gamma_1_old - temp_gamma_1_new, 2))
        updates2.append(np.linalg.norm(temp_gamma_2_old - temp_gamma_2_new, 2))
        
        ## step10: check if iteration has converged via updates
        if (updates1[-1] < tol) and (updates2[-1] < tol):
            break
        
    if plot:
        ## TODO: get solutions, pad them accordingly with wall values and put them into a plot
        u1 = room1.get_solution()
        u2 = room2.get_solution()
        u3 = room3.get_solution()
        
        pl.figure()
        ## TODO plotting
        
    return updates1, updates2, k+1

if __name__ == "__main__":
    pl.close("all")
    
    up1, up2, iters = DN_iteration(20, 20)
    
    pl.figure()
    pl.semilogy(range(iters), up1, label = 'updates gamma1')
    pl.semilogy(range(iters), up2, label = 'updates gamma2')
    pl.xlabel('iteration')
    pl.ylabel('update')
    pl.grid(True, 'major')
    pl.legend()
    
    
    