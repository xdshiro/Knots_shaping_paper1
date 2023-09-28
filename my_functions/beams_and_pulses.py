"""
This module includes different optical beam shapes
"""
import numpy as np
from scipy.special import assoc_laguerre
import my_functions.functions_general as fg
import math
np.seterr(divide='ignore', invalid='ignore')

def LG_simple(x, y, z=0, l=1, p=0, width=1, k0=1, x0=0, y0=0, z0=0):
    """
    Classic LG beam
    :param l: azimuthal index
    :param p: radial index
    :param width: beam waste
    :param k0: wave number
    :param x0: center of the beam in x
    :param y0: center of the beam in y
    :return: complex field
    """

    def laguerre_polynomial(x, l, p):
        return assoc_laguerre(x, p, l)

    x = x - x0
    y = y - y0
    z = z - z0
    zR = (k0 * width ** 2)
    # zR = (k0 * width ** 2) / 2

    #
    E = (np.sqrt(math.factorial(p) / (np.pi * math.factorial(np.abs(l) + p)))
         * fg.rho(x, y) ** np.abs(l) * np.exp(1j * l * fg.phi(x, y))
         / (width ** (np.abs(l) + 1) * (1 + 1j * z / zR) ** (np.abs(l) + 1))
         * ((1 - 1j * z / zR) / (1 + 1j * z / zR)) ** p
         * np.exp(-fg.rho(x, y) ** 2 / (2 * width ** 2 * (1 + 1j * z / zR)))
         * laguerre_polynomial(fg.rho(x, y) ** 2 / (width ** 2 * (1 + z ** 2 / zR ** 2)), np.abs(l), p)
         )
    # E = (np.sqrt(2 * math.factorial(p) / (np.pi * math.factorial(np.abs(l) + p)))
    #      * (fg.rho(x, y) * np.sqrt(2)) ** np.abs(l) * np.exp(1j * l * fg.phi(x, y))
    #      / (width ** (np.abs(l) + 1) * (1 + 1j * z / zR) ** (np.abs(l) + 1))
    #      * ((1 - 1j * z / zR) / (1 + 1j * z / zR)) ** p
    #      * np.exp(-fg.rho(x, y) ** 2 / (width ** 2 * (1 + 1j * z / zR)))
    #      * laguerre_polynomial(2 * fg.rho(x, y) ** 2 / (width ** 2 * (1 + z ** 2 / zR ** 2)), np.abs(l), p)
    #      )
    # zR = (k0 * width ** 2) / 2
    # E = (np.sqrt(2 * math.factorial(p) / (np.pi * math.factorial(np.abs(l) + p)))
    #      * (fg.rho(x, y) * np.sqrt(2)) ** np.abs(l) * np.exp(-1j * l * fg.phi(x, y))
    #      / ((width * np.sqrt(1 + z ** 2 / zR ** 2)) ** (np.abs(l) + 1))
    #               * np.exp(-fg.rho(x, y) ** 2 / (width ** 2 * (1 + z ** 2 / zR ** 2)))
    #      * laguerre_polynomial(2 * fg.rho(x, y) ** 2 / (width ** 2 * (1 + z ** 2 / zR ** 2)), np.abs(l), p)
    #      * np.exp(1j * (np.abs(l) + 2 * p + 1) * np.arctan(z / zR))
    #      * np.exp(-1j * k0 * fg.rho(x, y) ** 2 * z / (z ** 2 + zR ** 2) / 2)
    #      )
    return E


def trefoil(*x, w, width=1, k0=1, aCoeff=None, coeffPrint=False, **kwargs):
    H = 1.0
    if aCoeff is not None:
        aSumSqr = 0.1 * np.sqrt(sum([a ** 2 for a in aCoeff]))
        aCoeff /= aSumSqr
    else:
        a00 = 1 * (H ** 6 - H ** 4 * w ** 2 - 2 * H ** 2 * w ** 4 + 6 * w ** 6) / H ** 6
        a01 = (w ** 2 * (1 * H ** 4 + 4 * w ** 2 * H ** 2 - 18 * w ** 4)) / H ** 6
        a02 = (- 2 * w ** 4 * (H ** 2 - 9 * w ** 2)) / H ** 6
        a03 = (-6 * w ** 6) / H ** 6
        a30 = (-8 * np.sqrt(6) * w ** 3) / H ** 3
        aCoeff = [a00, a01, a02, a03, a30]
        aSumSqr = 0.1 * np.sqrt(sum([a ** 2 for a in aCoeff]))
        aCoeff /= aSumSqr
    if coeffPrint:
        print(aCoeff)
        print(f'a00 -> a01 -> a02 ->... -> a0n -> an0:')
        for i, a in enumerate(aCoeff):
            print(f'a{i}: {a:.3f}', end=',\t')
    field = (aCoeff[0] * LG_simple(*x, l=0, p=0, width=width, k0=k0, **kwargs) +
             aCoeff[1] * LG_simple(*x, l=0, p=1, width=width, k0=k0, **kwargs) +
             aCoeff[2] * LG_simple(*x, l=0, p=2, width=width, k0=k0, **kwargs) +
             aCoeff[3] * LG_simple(*x, l=0, p=3, width=width, k0=k0, **kwargs) +
             aCoeff[4] * LG_simple(*x, l=3, p=0, width=width, k0=k0, **kwargs)
             )
    return field


def hopf(*x, w, width=1, k0=1, aCoeff=None, coeffPrint=False, **kwargs):
    if aCoeff is not None or aCoeff is False:
        aSumSqr = 0.1 * np.sqrt(sum([a ** 2 for a in aCoeff]))
        aCoeff /= aSumSqr
    else:
        a00 = 1 - 2 * w ** 2 + 2 * w ** 4
        a01 = 2 * w ** 2 - 4 * w ** 4
        a02 = 2 * w ** 4
        a20 = 4 * np.sqrt(2) * w ** 2
        aCoeff = [a00, a01, a02, a20]
        aSumSqr = 0.1 * np.sqrt(sum([a ** 2 for a in aCoeff]))
        aCoeff /= aSumSqr
    if coeffPrint:
        print(aCoeff)
        print(f'a00 -> a01 -> a02 ->... -> a0n -> an0:')
        for i, a in enumerate(aCoeff):
            print(f'a{i}: {a:.3f}', end=',\t')
        print()
    field = (aCoeff[0] * LG_simple(*x, l=0, p=0, width=width, k0=k0) +
             aCoeff[1] * LG_simple(*x, l=0, p=1, width=width, k0=k0) +
             aCoeff[2] * LG_simple(*x, l=0, p=2, width=width, k0=k0) +
             aCoeff[3] * LG_simple(*x, l=2, p=0, width=width, k0=k0)
             )
    return field


def milnor_Pol_u_v_any(mesh, uOrder, vOrder, H=1):
    """This function create u^a-v^b Milnor polynomial"""
    x, y, z = mesh
    R = fg.rho(x, y)
    f = fg.phi(x, y)
    u = (-H ** 2 + R ** 2 + 2j * z * H + z ** 2) / (H ** 2 + R ** 2 + z ** 2)
    v = (2 * R * H * np.exp(1j * f)) / (H ** 2 + R ** 2 + z ** 2)
    return u ** uOrder - v ** vOrder


def LG_combination(*mesh, coefficients, modes, width=1, **kwargs):
    """
    creating the field of any combination of LG beams
    Sum(Cl1p1 * LG_simple(*mesh, l=l1, p=p1, **kwargs))
    :param mesh: np.meshgrid
    :param coefficients: [Cl1p1, Cl2p2...] ...
    :param modes: [(l1,p1), (l2,p2) ...]
    """
    field = 0
    if isinstance(width, int):
        width = [width] * len(modes)
    for num, coefficient in enumerate(coefficients):
        field += coefficient * LG_simple(*mesh, l=modes[num][0], p=modes[num][1], width=width[num], **kwargs)
    return field

if __name__ == '__main__':
    import my_functions.plotings as pl
    xyzMesh = fg.create_mesh_XYZ(4, 4, 1, zMin=None)
    beam = milnor_Pol_u_v_any(xyzMesh, uOrder=2, vOrder=2, H=1)
    pl.plot_2D(np.abs(beam[:, :, 20]))