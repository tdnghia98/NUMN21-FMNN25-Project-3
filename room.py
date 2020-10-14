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
    def __init__(self, nx, ny, dx, dy):
        self.nx, self.ny = nx, ny # number of interior unknowns in given dimension
        self.dx, self.dy = dx, dy  
        self.dof = nx*ny
        self.A = diags([1/self.dy**2,1/self.dx**2,-2*(self.dx**2+self.dy**2)/(self.dx**2*self.dy**2),1/self.dx**2,1/self.dy**2],[-nx,-1,0,1,nx],shape=(self.dof,self.dof),format="csr")
        ## removing extra elements on the side diagonals
        for i in range(1,ny):
            self.A[i*nx -1, i*nx] = 0
            self.A[i*nx, i*nx - 1] = 0
        self.f = np.zeros(self.dof)#boundary conditions
        self.u = None #solution
        
    ## storing the right-hand side based on static boundary conditions
    def f_checkpoint(self):
        self.f_save = np.copy(self.f)
    ## resetting right-hand side
    def f_reset(self):
        self.f = np.copy(self.f_save)
        
    def set_boundary_type(self, which = 'D', where = 'left'):
        nx = self.nx
        dof = self.dof
        dx = self.dx
        
        ## adding boundary condtions to self.A resp. self.f
        if which == 'D': ## adding Dirichlet boundary
            ## Dirichlet boundary is assumed to be the default
            pass
        elif which == 'N':
            if where == 'left':
                neu_left_border = np.arange(0, dof, nx)
                self.A[neu_left_border, neu_left_border] += 1/dx**2
            elif where == 'right':
                neu_right_border = np.arange(nx-1, dof, nx)
                self.A[neu_right_border, neu_right_border] += 1/dx**2
            else:
                raise KeyError('invalid <where> location specified, resp. not implemented')
        else:
            raise KeyError('invalid boundary condition type specified')

    def add_boundary(self, value, which = 'D', where = 'left'):
        nx = self.nx
        dof = self.dof
        dx = self.dx
        dy = self.dy
        
        ## adding boundary condtions to self.A resp. self.f
        if which == 'D': ## adding Dirichlet boundary         
            if where == 'left':
                left_border = np.arange(0, dof, nx)
                self.f[left_border] -= value/dx**2
            elif where == 'right':
                right_border = np.arange(nx-1, dof, nx)
                self.f[right_border] -= value/dx**2
            elif where == 'top':
                top_border = np.arange(dof-nx, dof)
                self.f[top_border] -= value/dy**2
            elif where == 'bottom':
                bottom_border = np.arange(nx)
                self.f[bottom_border] -= value/dy**2
            else:
                raise KeyError('invalid <where> location specified, resp. not implemented')
        elif which == 'N':
            if where == 'left':
                neu_left_border = np.arange(0, dof, nx)
                self.f[neu_left_border] += value/dx
            elif where == 'right':
                neu_right_border = np.arange(nx-1, dof, nx)
                self.f[neu_right_border] += value/dx
            else:
                raise KeyError('invalid <where> location specified, resp. not implemented')
        else:
            raise KeyError('invalid boundary condition type specified')
    
    def get_flux(self, temp_gamma_old, where = 'left'):
        if where == 'left':
            return (temp_gamma_old - self.u[np.arange(0, self.dof, self.nx)])/self.dx
        elif where == 'right':
            return (temp_gamma_old - self.u[np.arange(self.nx-1, self.dof, self.nx)])/self.dx
        else:
            raise KeyError('invalid <where> location specified, resp. not implemented')
        
    def solve(self, where = 'full'):
        self.u = spsolve(self.A, self.f) 
        if where == 'left':
            return self.u[np.arange(0, self.dof, self.nx)]
        elif where == 'right':
            return self.u[np.arange(self.nx-1, self.dof, self.nx)]
        elif where == 'full':
            return self.u
        else:
            raise KeyError('invalid <where> location specified, resp. not implemented')
    
    def get_solution(self, where = 'full'):
        return self.u.reshape(self.ny, self.nx).T
        
if __name__  == '__main__':
    r1 = room(3, 3, 1/3, 1/3)
    print(r1.A.todense())