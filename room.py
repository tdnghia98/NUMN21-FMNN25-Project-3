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
        #if not self.ny:
         #   self.ny = self.nx
        self.nx, self.ny = nx, ny # number of interior unknowns in given dimension
        self.dx, self.dy = dx, dy  
        self.dof = nx*ny
        self.A = diags([1/self.dy**2,1/self.dx**2,-2*(self.dx**2+self.dy**2)/(self.dx**2*self.dy**2),1/self.dx**2,1/self.dy**2],[-nx,-1,0,1,nx],shape=(self.dof,self.dof),format="csr")
        self.f = np.zeros(self.dof)#boundary conditions
        self.u = None #solution

    def add_boundary(self, value, which = 'D', where = 'left'):
        ny = self.ny
        nx = self.nx
        dof = self.dof
        dx = self.dx
        dy = self.dy
        A = self.A
        f = self.f
        
        ## adding boundary condtions to self.A resp. self.f
        if which == 'D': ## adding Dirichlet boundary         
            if where == 'left':
                left_border=np.arange(0,dof,nx)
                f[left_border]=-value/dx**2
                lft=left_border[left_border>0] 
                A[lft,lft-1]-=1/dx**2
                
            elif where == 'right':
                right_border=np.arange(nx-1,dof,nx)
                f[right_border]=value/dx**2
                rt=right_border[right_border<dof-1]
                A[rt,rt+1]-=1/dx**2
 
            elif where == 'top':
                top_border=np.arange(nx)
                f[top_border]=-value/dy**2
                
            elif where == 'bottom':
                bottom_border=np.arange(dof-nx,dof)
                f[bottom_border]=-value/dy**2
            else:
                raise KeyError('invalid <where> location specified, resp. not implemented')
        elif which == 'N':
            f=np.zeros(dof)
            if where == 'left':
                neu_left_border=np.arange(0,dof,nx)
                f[neu_left_border]=-value/dx
                A[neu_left_border,neu_left_border]+=1/dx**2
                lft=neu_left_border[neu_left_border>0]
                A[lft,lft-1]-=1/dx**2
                
            elif where == 'right':
                neu_right_border=np.arange(nx-1,dof,nx)
                f[neu_right_border]=-value/dx
                A[neu_right_border,neu_right_border]+=1/dx**2
                rt=neu_right_border[neu_right_border<dof-1]
                A[rt,rt+1]-=1/dx**2

            else:
                raise KeyError('invalid <where> location specified, resp. not implemented')
        else:
            raise KeyError('invalid boundary condition type specified')
        return([A, f])
    
    
    def get_flux(self, temp_gamma_old, where = 'left'):
        if where == 'left':
            flux=temp_gamma_old-self.u[np.arange(0,self.dof,self.nx)]
            return flux
        elif where == 'right':
            flux=temp_gamma_old-self.u[np.arange(self.nx-1,self.dof,self.nx)]
            return flux
        else:
            raise KeyError('invalid <where> location specified, resp. not implemented')
        
    def solve(self, where = 'full'):
        if where == 'left':
            sol=spsolve(self.A,self.f)
            left=sol[np.arange(0,self.dof,self.nx)]
            ## TODO for the mpi problem >> put in dirichlet/neumann condition for the boundary
            return left
        elif where == 'right':
            sol=spsolve(self.A,self.f)
            rt=sol[np.arange(self.nx-1,self.dof,self.nx)]
            ## TODO for the mpi problem >> put in dirichlet/neumann condition for the boundary
            return rt
        elif where == 'full':
            self.u=spsolve(self.A,self.f)
            return self.u
        else:
            raise KeyError('invalid <where> location specified, resp. not implemented')
        
    
    def plotting(self):
        u = self.u.reshape((self.ny,self.nx))[::-1]
        plt.imshow(u, origin='higher', cmap='hot')
        plt.colorbar()
