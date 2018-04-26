import numpy as np
import scipy.optimize as opt

def main():
    nobservations = 4
    a, b, c = 3.0, 2.0, 1.0
    f, x, y, z = generate_data(nobservations, a, b, c)

    print('Linear results (should be {}, {}, {}):'.format(a, b, c))
    print(linear_invert(f, x, y, z))

    print('Non-linear results (should be {}, {}, {}):'.format(a, b, c))
    print(nonlinear_invert(f, x, y, z))

def generate_data(nobservations, a, b, c, noise_level=0.01):
    x, y, z = np.random.random((3, nobservations))
    noise = noise_level * np.random.normal(0, noise_level, nobservations)
    f = func(x, y, z, a, b, c) + noise
    return f, x, y, z

def func(x, y, z, a, b, c):
    f = (a**2
         + x * b**2
         + y * a * b * np.cos(c)
         + z * a * b * np.sin(c))
    return f

def linear_invert(f, x, y, z):
    G = np.vstack([np.ones_like(x), x, y, z]).T
    m, _, _, _ = np.linalg.lstsq(G, f)

    d, e, f, g = m
    a = np.sqrt(d)
    b = np.sqrt(e)
    c = np.arctan2(g, f) # Note that `c` will be in radians, not degrees
    return a, b, c

def nonlinear_invert(f, x, y, z):
    # "curve_fit" expects the function to take a slightly different form...
    def wrapped_func(observation_points, a, b, c):
        x, y, z = observation_points
        return func(x, y, z, a, b, c)

    xdata = np.vstack([x, y, z])
    model, cov = opt.curve_fit(wrapped_func, xdata, f)
    return model

main()
