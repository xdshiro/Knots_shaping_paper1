from functions_based import *
import my_functions.plotings as pl
import knots_ML.data_generation as dg

import numpy as np
from scipy.special import assoc_laguerre
import my_functions.functions_general as fg
import math

def u(x, y, z):
	numerator = x ** 2 + y ** 2 + z ** 2 - 1 + 2j * z
	denominator = x ** 2 + y ** 2 + z ** 2 + 1
	return numerator / denominator


def v(x, y, z):
	numerator = 2 * (x + 1j * y)
	denominator = x ** 2 + y ** 2 + z ** 2 + 1
	return numerator / denominator
	
def cos_v(x, y, z, power=1):
	return (v(x, y, z) ** power + np.conj(v(x, y, z)) ** power) / 2

def sin_v(x, y, z, power=1):
	return (v(x, y, z) ** power - np.conj(v(x, y, z)) ** power) / 2j

u(x, y, z) * np.exp(1j * theta) - (
				cos_v(x_new, y_new, z_new, pow_cos) / a_cos_3D + 1j
				* sin_v(x_new, y_new, z_new, pow_sin) / a_sin_3D) * np.exp(1j * angle_3D)