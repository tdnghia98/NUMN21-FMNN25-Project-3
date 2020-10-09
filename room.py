#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 11:43:06 2020
"""

import numpy as np
import scipy.sparse as sps

class room:
    def __init__(self, n_x, n_y, dx, dy):
        self.n_x, self.n_y = n_x, n_y # number of interior unknowns in given dimension
        self.dy, self.dy = dx, dy
        
        ## TODO
        self.A = sps.eye(self.n_x * self.n_y)
        self.b = np.zeros(self.n_x * self.n_y)
        
        self.u = None ## solution
        
    def add_boundary(self, which = 'D', where = 'left', value = None):
        ## TODO: differentiate between simple value and actual array for "value"
        ## adding boundary condtions to self.A resp. self.b
        if which == 'D': ## adding Dirichlet boundary
            if where == 'left':
                ## TODO
                pass
            elif where == 'right':
                ## TODO
                pass
            elif where == 'top':
                ## TODO
                pass
            elif where == 'bottom':
                ## TODO
                pass
            else:
                raise KeyError('invalid <where> location specified, resp. not implemented')
        elif which == 'N':
            if where == 'left':
                ## TODO
                pass
            elif where == 'right':
                ## TODO
                pass
            else:
                raise KeyError('invalid <where> location specified, resp. not implemented')
        else:
            raise KeyError('invalid boundary condition type specified')
    
    def get_flux(self, where = 'left'):
        if where == 'left':
            ## TODO
            return np.zeros(self.n_y)
        elif where == 'right':
            ## TODO
            return np.zeros(self.n_y)
        else:
            raise KeyError('invalid <where> location specified, resp. not implemented')
        
    def get_solution(self, where = 'full'):
        if where == 'left':
            ## TODO
            return np.zeros(self.n_y)
        elif where == 'right':
            ## TODO
            return np.zeros(self.n_y)
        elif where == 'full':
            ## TODO re-check this one, make sure dimensions fit
            return self.u.reshape(self.n_x, self.n_y)
        else:
            raise KeyError('invalid <where> location specified, resp. not implemented')
        
    def solve(self):
        self.u = sps.linalg.spsolve(self.A, self.b)