#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 12:00:38 2020
"""

import numpy as np
import matplotlib.pyplot as plt
from room import room

def DN_iteration(nx, ny, theta = 0.7, maxiter = 100, tol = 1e-10, plot = True):
    ## nx & ny = number of interior unknowns per unit length
    temp_wall = 15
    temp_heater = 40
    temp_window = 5
    
    dx, dy = 1/(nx + 1), 1/(ny + 1)
    
    ## set up rooms and their static boundary conditions
    room1 = room(nx + 1, ny , dx, dy)
    room1.add_boundary(temp_heater,'D', 'left')   
    room1.add_boundary(temp_wall,'D', 'top')   
    room1.add_boundary(temp_wall,'D', 'bottom')
    
    
    room2 = room(nx, 2*ny+1, dx, dy)
    room2.add_boundary(temp_heater,'D', 'top')
    room2.add_boundary(temp_window,'D', 'bottom')
    
    room3 = room(nx + 1, ny, dx, dy)
    room3.add_boundary(temp_heater, 'D', 'right')    
    room3.add_boundary(temp_wall, 'D', 'top')    
    room3.add_boundary(temp_wall, 'D', 'bottom')
    
    
    ## set initial condition for boundaries in room2
    temp_gamma_1_old = np.ones(2*ny+1)*temp_wall
    temp_gamma_2_old = np.ones(2*ny+1)*temp_wall
    
    
    updates1, updates2 = [], []
    for k in range(maxiter):
        print(temp_gamma_2_old)
        ## step1: add dirichlet boundaries for room2
        room2.add_boundary(temp_gamma_1_old,'D', 'left')
        room2.add_boundary(temp_gamma_2_old,'D', 'right')
        
        ## step 2: solve room2
        room2.solve()
        
        ## step 3: get heat fluxes
        flux_gamma_1 = room2.get_flux(temp_gamma_1_old,'left')
        flux_gamma_2 = room2.get_flux(temp_gamma_2_old,'right')
        
        ## step 4: cut to length
        flux_gamma_1 = flux_gamma_1[:ny]
        flux_gamma_2 = flux_gamma_2[-ny:]
        
        ## step5: add fluxes as boundaries to room1 and room3
        room1.add_boundary(-flux_gamma_1,'N', 'right')
        room3.add_boundary(-flux_gamma_2,'N', 'left')
        
        ## step6: solve room1 and room3
        room1.solve()
        room3.solve()
        
        ## step7: get new boundary values
        temp_gamma_1_new = room1.solve(where = 'right')
        temp_gamma_2_new = room3.solve(where = 'left')
        
        ## step8: relaxation and update solutions
        L=np.concatenate((temp_gamma_1_old[:ny+1],temp_gamma_1_new))
        temp_gamma_1_old = (1-theta)*temp_gamma_1_old + theta*L
        R=np.concatenate((temp_gamma_2_new,temp_gamma_2_old[ny:]))
        temp_gamma_2_old = (1-theta)*temp_gamma_2_old + theta*R
        
        ## step9: compute updates
        ## TODO: possibly use different norm, e.g., discrete L2 norm?
        updates1.append(np.linalg.norm(temp_gamma_1_old - L, 2))
        updates2.append(np.linalg.norm(temp_gamma_2_old - R, 2))
        
        ## step10: check if iteration has converged via updates
        if (updates1[-1] < tol) and (updates2[-1] < tol):
            break
        
    if plot:
        ## TODO: get solutions, pad them accordingly with wall values and put them into a plot
        room1.solve()
        room2.solve()
        room3.solve()
        
        plt.figure()
        plt.title("room1")
        room1.plotting()
        plt.figure()
        plt.title("room2")
        room2.plotting()
        plt.figure()
        plt.title("room3")
        room3.plotting()        
    return updates1, updates2, k+1

if __name__ == "__main__":
    plt.close("all")
    
    up1, up2, iters = DN_iteration(20, 20)
    
    plt.figure()
    plt.semilogy(range(iters), up1, label = 'updates gamma1')
    plt.semilogy(range(iters), up2, label = 'updates gamma2')
    plt.xlabel('iteration')
    plt.ylabel('update')
    plt.grid(True, 'major')
    plt.legend()
    
    
    