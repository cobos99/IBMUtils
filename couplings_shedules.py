import numpy as np

def line_and_constant(x0, y0, x1, y1, t):
    t = np.asarray(t)
    to_return = np.zeros_like(t)
    to_return[t <= x0] = y0
    to_return[t >= x1] = y1
    mask = (t > x0) & (t < x1)
    to_return[mask] = y0 + (y1 - y0) * ((t[mask] - x0) / (x1 - x0))
    return to_return

def dt_center_lineconstant_couplings(t0, y0, t1, y1, t, dt):
    return line_and_constant(t0, y0, t1, y1, t-dt/2)

class LinearAndConstantSchedule:
    def __init__(self, x0, y0, x1, y1, dt=0):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.dt = dt

    def __call__(self, t):
        t = np.asarray(t) - self.dt/2
        to_return = np.zeros_like(t)
        to_return[t <= self.x0] = self.y0
        to_return[t >= self.x1] = self.y1
        mask = (t > self.x0) & (t < self.x1)
        to_return[mask] = self.y0 + (self.y1 - self.y0) * ((t[mask] - self.x0) / (self.x1 - self.x0))
        return to_return 

    def set_0(self, x0, y0, x1, y1, dt=0):
        return LinearAndConstantSchedule(x0, y0, x1, y1, dt)
    
class ConstantSchedule:
    def __init__(self, value):
        self.value = value
    
    def __call__(self, t):
        t = np.asarray(t)
        return np.full_like(t, self.value)
    
    def set_0(self, value):
        return ConstantSchedule(value)