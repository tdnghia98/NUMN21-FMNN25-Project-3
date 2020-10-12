#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 11:43:06 2020
"""

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags
import matplotlib.pyplot as plt


class room:
    def __init__(self, nx, ny):
        #if not self.ny:
         #   self.ny = self.nx
        self.nx, self.ny = nx, ny # number of interior unknowns in given dimension
        self.dx, self.dy = 1/(nx), 2/(ny) 
        self.dof = nx*ny
        self.A = diags([1/self.dy**2,1/self.dx**2,-2*(self.dx**2+self.dy**2)/(self.dx**2*self.dy**2),1/self.dx**2,1/self.dy**2],[-nx,-1,0,1,nx],shape=(self.dof,self.dof),format="csr")
        self.f = None #solution


    def add_boundary(self, which = 'D', where = 'left', value = None, init_guess_Omega1=25, init_guess_Omega3=10):
        ny = self.ny
        nx = self.nx
        dof = self.dof
        dx = self.dx
        dy = self.dy
        A = self.A
        f = self.f
        ## TODO: differentiate between simple value and actual array for "value"
        ## adding boundary condtions to self.A resp. self.b
        if which == 'D': ## adding Dirichlet boundary            
            if where == 'left':
                left_border=np.arange(0,dof,nx)
                for i in left_border:
                    f[i]=-value/dx**2
                    if i>0:
                        A[i,i-1]-=1/dx**2
            elif where == 'right':
                right_border=np.arange(nx-1,dof,nx)
                for i in right_border:
                    f[i]=-value/dx**2
                    if i<dof-1:
                        A[i,i+1]-=1/dx**2
            elif where == 'top':
                top_border=np.arange(nx)
                for i in top_border:
                    f[i]=-value/dy**2
            elif where == 'bottom':
                bottom_border=np.arange(dof-nx,dof)
                for i in bottom_border:
                    f[i]=-value/dy**2
                    #A[i,i]+=1/dy**2
            else:
                raise KeyError('invalid <where> location specified, resp. not implemented')
        elif which == 'N':
            f=np.zeros(dof)
            if where == 'left':
                neu_left_border=np.arange(0,dof,nx)
                for i in neu_left_border:
                    f[i]=-init_guess_Omega1/dx #flux[i] should go here
                    A[i,i]+=1/dx**2
                    if i>0:
                        A[i,i-1]-=1/dx**2
            elif where == 'right':
                neu_right_border=np.arange(nx-1,dof,nx)
                for i in neu_right_border:
                    f[i]=-init_guess_Omega3/dx #flux[i] should go here
                    A[i,i]+=1/dx**2
                    if i<dof-1:
                        A[i,i+1]-=1/dx**2
            else:
                raise KeyError('invalid <where> location specified, resp. not implemented')
        else:
            raise KeyError('invalid boundary condition type specified')
        return([A, f])
    
    def get_flux(self, where = 'left'):
            ## TODO upper half of the wall is dirichilet conditions
        if where == 'left':
            return np.zeros(self.n_y)
        elif where == 'right':
            return np.zeros(self.n_y)
        else:
            raise KeyError('invalid <where> location specified, resp. not implemented')
        
    def get_solution(self, which = 'D', where = 'full'):
        if where == 'left':
            ## TODO for the mpi problem >> put in dirichlet/neumann condition for the boundary
            return np.zeros(self.n_y)
        elif where == 'right':
            ## TODO for the mpi problem >> put in dirichlet/neumann condition for the boundary
            return np.zeros(self.n_y)
        elif where == 'full' and which == 'D':
            self.f = np.zeros(self.dof)
            for i in ('left', 'right', 'top', 'bottom'):
                temp = self.add_boundary(which = 'D', where = i)
                self.A, self.f = temp[0], temp[1]
            return [self.A,self.f]
        
        elif where == 'full' and which == 'N':
            self.f = np.zeros(self.dof)
            for i in ('left', 'right'):
                temp = self.add_boundary(which = 'N', where = i)
                self.A, self.f = temp[0], temp[1]
            return [self.A,self.f]
        else:
            raise KeyError('invalid <where> location specified, resp. not implemented')
        
    def solve(self, which = 'N'):
        temp2 = self.get_solution(which = which)
        self.u = spsolve(temp2[0], temp2[1])
        self.u = self.u.reshape((self.ny,self.nx))[::-1]
        return self.u
    
    def plotting(self):
        plt.figure()
        plt.imshow(self.solve(which = 'N'), origin='upper', cmap='hot')
        plt.colorbar()
