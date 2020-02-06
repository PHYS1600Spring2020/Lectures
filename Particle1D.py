#!/usr/bin/python
from scipy.integrate import odeint
import matplotlib.pyplot as plt # for plotting          
import numpy as np

class Particle (object):

    """Class that describes particle"""
    m = 1.0

    def __init__(self, x0=1.0, v0=0.0,  tf = 10.0, dt = 0.001):
        self.x = x0
        self.v = v0
        self.t = 0.0
        self.tf = tf
        self.dt = dt

        self.tlabel = 'time (s)'
        self.xlabel = 'x (m)'
        self.vlabel = 'v (m/s)'

        npoints = int(tf/dt) # always starting at t = 0.0
        self.npoints = npoints
        self.tarray = np.linspace(0.0, tf,npoints, endpoint = True) # include final timepoint
        self.xv0 = np.array([self.x, self.v]) # NumPy array with initial position and velocity

    def F(self, x, v, t):
        # The force on a free particle is 0
        return array([0.0])

    def Euler_step(self): 
        """
        Take a single time step using Euler method
        """
        
        a = self.F(self.x, self.v, self.t) / self.m
        self.x += self.v * self.dt
        self.v += a * self.dt
        self.t += self.dt

    
    def Euler_trajectory(self):
        """
        Loop over all time steps to construct a trajectory with Euler method
        Will reinitialize euler trajectory everytime this method is called
        """
        
        x_euler = []
        v_euler = []
        
        while(self.t < self.tf-self.dt/2):
            v_euler.append(self.v)
            x_euler.append(self.x)
            self.Euler_step()
        
        self.x_euler = np.array(x_euler)
        self.v_euler = np.array(v_euler)


    def scipy_trajectory(self):
        """calculate trajectory using SciPy ode integrator"""
        self.xv = odeint(self.derivative, self.xv0, self.tarray)

    def derivative(self, xv, t):
        """right hand side of the differential equation"""
        x =xv[0]
        v =xv[1]
        a = self.F(x, v, t) / self.m
        return np.ravel(np.array([v, a]))

    def results(self):
        """
        Print out results in a nice format
        """

        
        print('\n\t Position and Velocity at Final Time:')
        print('Euler:')
        print('t = {} x = {} v = {}'.format(self.t, self.x , self.v))
        
        if hasattr(self, 'xv'):
            print('SciPy ODE Integrator:')
            print('t = {} x = {} v = {}'.format(self.tarray[-1], self.xv[-1, 0], self.xv[-1,1]))

    def plot(self, pt = 'trajectory'):
        """
        Make nice plots of our results
        """

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        
        
        if hasattr(self,'xv'):

            if pt == 'trajectory':
                ax1.plot(self.tarray, self.xv[:, 0], "k", label = 'odeint')
            if pt == 'phase':
                ax1.plot(self.xv[:, 0], self.xv[:, 1], "k",'.', label = 'odeint')
        
        if hasattr(self,'x_euler'):

            if pt == 'trajectory':
                ax1.plot(self.tarray, self.x_euler, "b", label = 'euler')
            if pt == 'phase':
                ax1.plot(self.x_euler, self.v_euler, "b",'.', label = 'euler')
        
       
        if pt == 'trajectory':
            ax1.set_xlabel(self.tlabel)
            ax1.set_ylabel(self.xlabel)
        
        if pt == 'phase':
            ax1.set_xlabel(self.xlabel)
            ax1.set_ylabel(self.vlabel)


class Pendulum(Particle):

    """Subclass of Particle Class that describes a pendulum in a harmonic potential"""
    def __init__(self, l = 9.8, nu = 0, Fd  = 0.0, omega_d = 0.0, m = 1.0, x0 = 0.0 ,v0 = 0.0, tf = 50.0, dt = 0.001):
       
        super().__init__(x0,v0,tf,dt) 
        # for pendulum x = theta [-pi, pi]
        g = 9.8
        omega0 = np.sqrt(g/l)
        
        self.l = l # length
        self.m = m # mass
        self.Fd = Fd # driving force, in units of mg
        self.omega_d = omega_d #driving frequency, in units of omega0
        self.nu = nu # viscous damping 
        self.omega0 = omega0 # natural frequency

        self.tlabel = 'time ($1/\omega_0$)'
        self.xlabel = '$\\theta$ (radians)'
        self.vlabel = '$\omega$ (radians/s)'

    # overload method to wrap x between [-pi,pi]
    def scipy_trajectory(self):
        Particle.scipy_trajectory(self)
        
        x = self.xv[:,0]
        x_new = np.zeros(np.shape(x))
        x_new[0] = x[0]

        # find change in x between each point
        dx = np.diff(x)
        nx = np.shape(x)[0]
        
        for ii in range(1,nx):
            # reconstruct x array, checking for out of range values
            x_new[ii] = x_new[ii-1]+dx[ii-1]
            if x_new[ii] > np.pi:
                x_new[ii] -= 2*np.pi
            
            elif x_new[ii] < -np.pi:
                x_new[ii] += 2*np.pi
        
        self.xv_unwrap = 1.0*self.xv
        self.xv[:,0] = x_new
    
    def F(self, x, v, t):
        g = 9.8 

        F = self.Fd*np.cos(self.omega_d*t) - self.nu*v - g/self.l*np.sin(x)
        
        return F

            
