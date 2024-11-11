from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.interpolate import BarycentricInterpolator
from scipy.interpolate import interp1d
from scipy.interpolate import splrep, splev


import numpy as np


def compute_cubic_spline(freq, H, sample):
    cubic_spline = CubicSpline(freq, H)
    new_freq = np.linspace(freq[0], freq[-1], sample)
    new_H = cubic_spline(new_freq)
    return new_freq, new_H

def compute_linear_interp(freq, H, sample):
    linear_interp = interp1d(freq, H, kind='linear')
    new_freq = np.linspace(freq[0], freq[-1], sample)
    new_H = linear_interp(new_freq)
    return new_freq, new_H

def compute_polynomial_interp(freq, H, sample):
    poly_interp = BarycentricInterpolator(freq, H)
    new_freq = np.linspace(freq[0], freq[-1], sample)
    new_H = poly_interp(new_freq)
    return new_freq, new_H


def compute_quadratic_spline(freq, H, sample):
    quadratic_spline = interp1d(freq, H, kind='quadratic')
    new_freq = np.linspace(freq[0], freq[-1], sample)
    new_H = quadratic_spline(new_freq)
    return new_freq, new_H

def compute_b_spline(freq, H, sample):
    tck = splrep(freq, H, k=3)  # k=3 pour une spline cubique
    new_freq = np.linspace(freq[0], freq[-1], sample)
    new_H = splev(new_freq, tck)
    return new_freq, new_H