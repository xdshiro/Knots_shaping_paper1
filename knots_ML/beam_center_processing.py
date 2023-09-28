"""
Algorithm to find the beam center and beam tilt using the electrical field.
Algorithm is based on the quasi-intrinsic-variance and OAM spectrum calculations.
For references check the following papers:
    1) Yi-Dong Liu,.. "Orbital angular momentum (OAM) spectrum correction in free space optical communication", OE, 2008
        This paper describes the algorithm in details. My implementation is only slightly different.
    2) https://doi.org/10.1103/PhysRevLett.96.113901
        Here you can find more theory on the quasi-intrinsic-variance and intrinsic/extrinsic parts of OAM in general.

The final function is called !beamFullCenter!. Read details there.

"__main__" has some example and mainly used for testing.
"""

import my_functions.functions_general as fg
import my_functions.singularities as sing
import my_functions.plotings as pl
import my_functions.beams_and_pulses as bp
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from data_generation import *


from collections import Counter

x = Counter("cat")
y = Counter("dat")
print(x.subtract(y))
x.update(x)
x = +x
print(x)
exit()

def LG_spectre_coeff(field, l, p, xM=(-1, 1), yM=(-1, 1), width=1., k0=1., mesh=None, functions=bp.LG_simple):
    """
    Function calculates a single coefficient of LG_l_p in the LG spectrum of the field
    :param field: complex electric field
    :param l: azimuthal index of LG beam
    :param p: radial index of LG beam
    :param xM: x boundaries for an LG beam (if Mesh is None)
    :param yM: y boundaries for an LG beam (if Mesh is None)
    :param width: LG beam width
    :param k0: k0 in LG beam but I believe it doesn't affect anything since we are in z=0
    :param mesh: mesh for LG beam. if None, xM and yM are used
    :return: complex weight of LG_l_p in the spectrum
    """
    if mesh is None:
        shape = np.shape(field)
        mesh = fg.create_mesh_XY(xMinMax=xM, yMinMax=yM, xRes=shape[0], yRes=shape[1])
        dS = ((xM[1] - xM[0]) / (shape[0] - 1)) * ((yM[1] - yM[0]) / (shape[1] - 1))
    else:
        xArray, yArray = fg.arrays_from_mesh(mesh)
        dS = (xArray[1] - xArray[0]) * (yArray[1] - yArray[0])
        # print(123, xArray)
    # shape = np.shape(field)
    # xyMesh = fg.create_mesh_XY_old(xMax=xM[1], yMax=yM[1], xRes=shape[0], yRes=shape[1], xMin=xM[0], yMin=yM[0])
    LGlp = functions(*mesh, l=l, p=p, width=width, k0=k0)
    # plt.imshow(LGlp)
    # plt.show()
    # print('hi')

    return np.sum(field * np.conj(LGlp)) * dS


def displacement_lateral(field_func, mesh, r_0, eta, **kwargs):
    """
    function shifts the field (function, not an array) to r_0 under eta angle
    :param field_func: any 2d function fo
    :param mesh: mesh for the final field
    :param r_0: radius-vector in polar coordinates
    :param eta: angle in polar coordinates
    :return: array with a shifted field
    """
    xArray, yArray = fg.arrays_from_mesh(mesh)
    xArray, yArray = xArray - r_0 * np.cos(eta), yArray - r_0 * np.sin(eta)
    mesh_new = np.meshgrid(xArray, yArray, indexing='ij')
    return field_func(*mesh_new, **kwargs)


def displacement_deflection(field_func, mesh, eta, gamma, k=1, **kwargs):
    """
    function tilt the field (function, not an array) using angles eta and gamma
    :param field_func: any 2d function fo
    :param mesh: mesh for the final field
    :param eta: check the first reference for the definition
    :param gamma: check the first reference for the definition
    :return: array with a shifted field
    """
    field_change = np.exp(1j * k * fg.rho(*mesh) * np.sin(gamma) * np.cos(fg.phi(*mesh) - eta))
    return field_func(*mesh, **kwargs) * field_change


def removeTilt(field, mesh, eta, gamma, k=1):
    """
    Function remove the tilt from the field (array) and returns new field
    :param field: initial field
    :param mesh: have to provide the mesh for the initial function
    :param k: wave number
    :return: field array
    """
    field_change = np.exp(1j * k * fg.rho(*mesh) * np.sin(gamma) * np.cos(fg.phi(*mesh) - eta))
    return field * field_change


def removeShift(mesh, x0, y0):
    """
    Function remove the shift from the mesh (array) and returns new mesh
    :param mesh: initial mesh
    :return: mesh array
    """
    xArray, yArray = fg.arrays_from_mesh(mesh)
    xArray, yArray = xArray - x0, yArray - y0
    mesh_new = np.meshgrid(xArray, yArray, indexing='ij')
    return mesh_new


def LG_transition_matrix_real_space(operator, l, p, l0, p0, xM=(-1, 1), yM=(-1, 1), shape=(100, 100),
                                    width=1., k0=1., mesh=None, **kwargs):
    """
    Transition element in equation (10) in the first reference. It works for any operator, not only misalignment
    operation T_LD.
    :param operator: T_LD, T_DD or their combination
    :param xM: x boundaries for an LG beam (if Mesh is None)
    :param yM: y boundaries for an LG beam (if Mesh is None)
    :param shape: size of the new mesh (if Mesh is None)
    :param width: LG beam width
    :param k0: k0 in LG beam but I believe it doesn't affect anything since we are in z=0
    :param mesh: mesh for LG beam. if None, xM and yM are used
    :param kwargs: into the operator
    :return: the value of the transition element
    """
    if mesh is None:
        mesh = fg.create_mesh_XY(xM, yM, xRes=shape[0], yRes=shape[1])
        dS = ((xM[1] - xM[0]) / (shape[0] - 1)) * ((yM[1] - yM[0]) / (shape[1] - 1))
    else:
        xArray, yArray = fg.arrays_from_mesh(mesh)
        dS = (xArray[1] - xArray[0]) * (yArray[1] - yArray[0])
    operatorOnLG = operator(bp.LG_simple,
                            mesh, z=0, l=l0, p=p0, width=width, k0=k0, x0=0, y0=0, z0=0, **kwargs)
    LGlp = bp.LG_simple(*mesh, l=l, p=p, width=width, k0=k0)
    return np.sum(operatorOnLG * np.conj(LGlp)) * dS


def shiftTiltCombined(field_func, mesh, r_0, eta, eta2, gamma, k=1, **kwargs):
    """
    This function combines both shift and tilt. Basically, it's displacement_lateral() + displacement_deflection()
    See more details in these 2 functions
    """
    xArray, yArray = fg.arrays_from_mesh(mesh)
    xArray, yArray = xArray - r_0 * np.cos(eta), yArray - r_0 * np.sin(eta)
    mesh_new = np.meshgrid(xArray, yArray, indexing='ij')
    field_change = np.exp(1j * k * fg.rho(*mesh) * np.sin(gamma) * np.cos(fg.phi(*mesh) - eta2))
    return field_func(*mesh_new, **kwargs) * field_change


def variance_V_helper(Pl, lArray):
    """
    helper for the variance_single_transition: eq (23)
    Not used anywhere else
    """
    sum1, sum2 = 0, 0
    # sumTest = 0
    for p, l in zip(Pl, lArray):
        # sumTest += p
        sum1 += p * l ** 2
        sum2 += p * l
    # print(sumTest)
    return sum1 - sum2 ** 2


def variance_single_transition_combined(field, mesh, r, eta, eta2, gamma,
                                        displacement_function=shiftTiltCombined,
                                        p=(0, 5), l=(0, 4), p0=(0, 5), l0=(-3, 3),
                                        width=1., k0=1.):
    """
    This function is the 'heart' of the algorithm implementation.
    It calculate the final variance (eq. (23) 1st reference).
    (Not sure if it works for only tilt or only shift if you change the the displacement_function.
    It worked in the separated implementations below, should work here. You probably don't need this to separate
    them anyway, it can be used only to increase the speed for some tests)
    :param displacement_function: shiftTiltCombined for search both tilt and shift.
    :param p: lowest and highest values for the whole integration range (see eqn. (10))
    :param l: lowest and highest values for the whole integration range (see eqn. (10))
    :param p0: lowest and highest values for the whole integration range (see eqn. (10))
    :param l0: lowest and highest values for the whole integration range (see eqn. (10))
    :param width: LG width
    :param k0: LG k0, doesn't affect anything, we work in z=0
    :return: the value of the variance
    """
    p1, p2 = p
    l1, l2 = l
    p01, p02 = p0
    l01, l02 = l0
    Pl = np.zeros(l2 - l1 + 1)
    for ind_l, l in enumerate(np.arange(l1, l2 + 1)):
        sum = 0
        for p in np.arange(p1, p2 + 1):
            sum_inner = 0
            for l0 in np.arange(l01, l02 + 1):
                for p0 in np.arange(p01, p02 + 1):
                    element_ = LG_transition_matrix_real_space(displacement_function, l=l, p=p,
                                                               l0=l0, p0=p0, mesh=mesh,
                                                               r_0=r, eta=eta, eta2=eta2, gamma=gamma,
                                                               width=width, k0=k0)
                    value = LG_spectre_coeff(field, l=l0, p=p0, mesh=mesh, width=width, k0=k0)
                    # print(element_, value, (np.abs(element_ * value) ** 2))
                    sum_inner += (element_ * value)
            sum += np.abs(sum_inner) ** 2
        Pl[ind_l] = sum
        # print(Pl[ind_l])
    V = variance_V_helper(Pl, np.arange(l1, l2 + 1))
    return V


def LG_spectrum(beam, l=(-3, 3), p=(0, 5), xM=(-1, 1), yM=(-1, 1), width=1., k0=1., mesh=None, plot=True,
                functions=bp.LG_simple, **kwargs):
    """

    :param beam:
    :param l:
    :param p:
    :param xM:
    :param yM:
    :param width:
    :param k0:
    :param mesh:
    :param plot:
    :return:
    """
    l1, l2 = l
    p1, p2 = p
    spectrum = np.zeros((l2 - l1 + 1, p2 - p1 + 1), dtype=complex)
    # spectrumReal = []
    # modes = []
    for l in np.arange(l1, l2 + 1):
        for p in np.arange(p1, p2 + 1):
            value = LG_spectre_coeff(beam, l=l, p=p, xM=xM, yM=yM, width=width, k0=k0, mesh=mesh,
                                     functions=functions, **kwargs)
            # print(l, p, ': ', value, np.abs(value))
            spectrum[l - l1, p] = value
            # if np.abs(value) > 0.5:
            # spectrumReal.append(value)
            # modes.append((l, p))
    # print(modes)
    if plot:
        import matplotlib.pyplot as plt
        pl.plot_2D(np.abs(spectrum), x=np.arange(l1 - 0.5, l2 + 1 + 0.5), y=np.arange(p1 - 0.5, p2 + 1 + 0.5),
                   interpolation='none', grid=True, xname='l', yname='p', show=False)
        plt.yticks(np.arange(p1, p2 + 1))
        plt.xticks(np.arange(l1, l2 + 1))
        plt.show()
    return spectrum


def beamFullCenter(beam, mesh, stepEG=None, stepXY=None, displacement_function=shiftTiltCombined,
                   p=(0, 5), l=(0, 4), p0=(0, 5), l0=(-3, 3),
                   width=1, k0=1, x=None, y=None, eta2=0., gamma=0., threshold=0.99,
                   print_info=False):
    if stepEG is None:
        stepEG = 0.1, 0.1
    if stepXY is None:
        xArray, yArray = fg.arrays_from_mesh(mesh)
        # print(xArray)
        stepXY = xArray[1] - xArray[0], yArray[1] - yArray[0]
    if x is None:
        xArray, yArray = fg.arrays_from_mesh(mesh)
        x = xArray[len(xArray) // 2]
    if y is None:
        xArray, yArray = fg.arrays_from_mesh(mesh)
        y = yArray[len(yArray) // 2]

    def search(coordinate):
        nonlocal x, y, eta2, gamma, var0
        varIt = var0
        signX = 1
        correctWay = False
        while True:
            if coordinate == 'x':
                x = x + signX * stepXY[0]
            elif coordinate == 'y':
                y = y + signX * stepXY[1]
            elif coordinate == 'eta2':
                eta2 = eta2 + signX * stepEG[0]
            elif coordinate == 'gamma':
                gamma = gamma + signX * stepEG[1]

            var = variance_single_transition_combined(beam, mesh, r=fg.rho(x, y), eta=np.angle(x + 1j * y),
                                                      eta2=eta2, gamma=gamma,
                                                      displacement_function=displacement_function,
                                                      p=p, l=l, p0=p0, l0=l0, width=width, k0=k0)
            if print_info:
                print(f'x={round(x, 3)}, y={round(y, 3)},'
                      f' eta={round(eta2 * 180 / np.pi, 3)}*, gamma={round(gamma * 180 / np.pi, 3)}*, var={var}')
            if var < (varIt * threshold):
                varIt = var
                correctWay = True
            else:
                if coordinate == 'x':
                    x = x - signX * stepXY[0]
                elif coordinate == 'y':
                    y = y - signX * stepXY[1]
                elif coordinate == 'eta2':
                    eta2 = eta2 - signX * stepEG[0]
                elif coordinate == 'gamma':
                    gamma = gamma - signX * stepEG[1]
                if correctWay:
                    break
                else:
                    correctWay = True
                    signX *= -1

        return varIt

    var0 = variance_single_transition_combined(beam, mesh, r=fg.rho(x, y), eta=np.angle(x + 1j * y),
                                               eta2=eta2, gamma=gamma,
                                               displacement_function=displacement_function,
                                               p=p, l=l, p0=p0, l0=l0, width=width, k0=k0)

    while True:
        search(coordinate='x')
        search(coordinate='y')
        search(coordinate='eta2')
        varEG = search(coordinate='gamma')
        if varEG == var0:
            print('FINAL COORDINATES:')
            print(f'x={round(x, 5)}, y={round(y, 5)},'
                  f' eta={round(eta2 * 180 / np.pi, 5)}*,'
                  f' gamma={round(gamma * 180 / np.pi, 5)}*, var={var0}')
            return x, y, eta2, gamma
        else:
            var0 = varEG


def find_width(beam, mesh, widthStep=0.1, l=(-8, 8), p=(0, 9), width=1., k0=1., print_steps=True):
    """
    this function finds the approximate beam waste (any position, any beam=sum(LG))
    :param mesh: mesh is required to have a scale of the beam => correct width scale
    :param widthStep: precision of the beam width search (but it's not real precision, method is not very accurate)
    :param l: we wont to cover all spectrum
    :param p: we wont to cover all spectrum
    :param width: starting beam width (you better have it as accurate as poosible, since the method is bad)
    :param k0: does nothing
    :return: ~beam width
    """
    minSpec = np.sum(np.abs(LG_spectrum(beam, l=l, p=p, mesh=mesh, width=width, k0=k0, plot=False)) ** (1 / 2))
    correctWay = False
    direction = +1
    while True:
        width += direction * widthStep
        spec = np.sum(np.abs(LG_spectrum(beam, l=l, p=p, mesh=mesh, width=width, k0=k0, plot=False)) ** (1 / 2))
        if print_steps:
            print(width, spec)
        if spec < minSpec:
            minSpec = spec
            correctWay = True
        else:
            width -= direction * widthStep
            if correctWay:
                break
            else:
                correctWay = True
                direction *= -1
    return width


def beam_center_coordinates(beam, mesh, stepXY=None, stepEG=None,
                            p=(0, 5), l=(0, 4), p0=(0, 5), l0=(-3, 3),
                            width=1, k0=1, x=None, y=None, eta=0., gamma=0.,
                            shift=True, tilt=True, fast=False):
    if fast or not shift or not tilt:
        print(f'Not the perfect center: shift={shift}, tilt={tilt}, fast={fast}')
        if shift:
            x, y = center_beam_finding(beam, mesh, x=x, y=y, stepXY=stepXY,
                                       p=p, l=l, p0=p0, l0=l0, width=width, k0=k0)
        if tilt:
            eta, gamma = tilt_beam_finding(beam, mesh, eta=eta, gamma=gamma, stepEG=stepEG,
                                           p=p, l=l, p0=p0, l0=l0, width=width, k0=k0)
    else:
        meshNew = mesh
        while True:
            print('New cycle')
            # print(x, y, eta, gamma)
            x, y = center_beam_finding(beam, meshNew, x=x, y=y, stepXY=stepXY,
                                       p=p, l=l, p0=p0, l0=l0, width=width, k0=k0)
            eta, gamma = tilt_beam_finding(beam, mesh, eta=eta, gamma=gamma, stepEG=stepEG,
                                           p=p, l=l, p0=p0, l0=l0, width=width, k0=k0)

            meshNew = removeShift(meshNew, -x, -y)
            # beam = removeTilt(beam, mesh, eta=eta, gamma=-gamma)
            pl.plot_2D(np.angle(beam))
            pl.plot_2D(np.abs(beam))
            print(x, y, eta, gamma)
            if (x, y, eta, gamma) == (0, 0, 0, 0):
                break
            x, y, eta, gamma = [0] * 4
        # xBest, yBest, etaBest, gammaBest = [np.inf] * 4
        # while (xBest, yBest, etaBest, gammaBest) != (x, y, eta, gamma):
        #     print('New cycle')
        #     # print(x, y, eta, gamma)
        #     xBest, yBest, etaBest, gammaBest = (x, y, eta, gamma)
        #     print(xBest, yBest, etaBest, gammaBest)
        #     x, y = center_beam_finding(beam, mesh, x=x, y=y, stepXY=stepXY,
        #                                p=p, l=l, p0=p0, l0=l0, width=width, k0=k0)
        #     eta, gamma = tilt_beam_finding(beam, mesh, eta=eta, gamma=gamma, stepEG=stepEG,
        #                                    p=p, l=l, p0=p0, l0=l0, width=width, k0=k0)
        #
        #     mesh = removeShift(mesh, -x, -y)
        #     beam = removeTilt(beam, mesh, eta=eta, gamma=gamma)
        #     print(x, y, eta, gamma)
        #     x, y, eta, gamma = [0] * 4

    print(f'x={x}, y={y}, eta={eta / np.pi * 180}, gamma={gamma / np.pi * 180}')
    return mesh, beam


# functions below are old and may not be functional. There were used for the algorithm implementation
# for separated shift and tilt, not combined.
# Maybe useful if all the fields have only shift or only tilt
####################################################################################


def variance_single_transition(field, mesh, r, eta, displacement_function=displacement_lateral,
                               p=(0, 5), l=(0, 4), p0=(0, 5), l0=(-3, 3),
                               width=1., k0=1.):
    """
    basically the same as variance_single_transition_combined() but for shift or tilt
    """
    p1, p2 = p
    l1, l2 = l
    p01, p02 = p0
    l01, l02 = l0
    Pl = np.zeros(l2 - l1 + 1)
    for ind_l, l in enumerate(np.arange(l1, l2 + 1)):
        sum = 0
        for p in np.arange(p1, p2 + 1):
            sum_inner = 0
            for l0 in np.arange(l01, l02 + 1):
                for p0 in np.arange(p01, p02 + 1):
                    if displacement_function is displacement_lateral:
                        element_ = LG_transition_matrix_real_space(displacement_function, l=l, p=p,
                                                                   l0=l0, p0=p0,
                                                                   mesh=mesh, r_0=r, eta=eta,
                                                                   width=width, k0=k0)
                    else:
                        element_ = LG_transition_matrix_real_space(displacement_function, l=l, p=p,
                                                                   l0=l0, p0=p0,
                                                                   mesh=mesh, eta=r, gamma=eta,
                                                                   width=width, k0=k0)
                    value = LG_spectre_coeff(field, l=l0, p=p0, mesh=mesh, width=width, k0=k0)
                    # print(element_, value, (np.abs(element_ * value) ** 2))
                    sum_inner += (element_ * value)
            sum += np.abs(sum_inner) ** 2
        Pl[ind_l] = sum
        # print(Pl[ind_l])
    V = variance_V_helper(Pl, np.arange(l1, l2 + 1))
    return V


def variance_map_shift(beam, mesh, displacement_function=displacement_lateral,
                       resolution_V=(4, 4), xBound=(-1, 1), yBound=(-1, 1), width=1., k0=1.,
                       p=(0, 5), l=(0, 4), p0=(0, 5), l0=(-3, 3)):
    V = np.zeros(resolution_V)

    xArray = np.linspace(*xBound, resolution_V[0])
    yArray = np.linspace(*yBound, resolution_V[1])
    for i, x in enumerate(xArray):
        print('Main Coordinate x: ', i)
        for j, y in enumerate(yArray):
            print('y: ', j)
            r = fg.rho(x, y)
            eta = np.angle(x + 1j * y)
            V[i, j] = variance_single_transition(beam, mesh, r, eta,
                                                 displacement_function=displacement_function,
                                                 width=width, k0=k0,
                                                 p=p, l=l, p0=p0, l0=l0)
    return V


def variance_map_tilt(beam, mesh, displacement_function=displacement_deflection,
                      resolution_V=(4, 4), etaBound=(-1, 1), gammaBound=(-1, 1), width=1., k0=1.,
                      p=(0, 5), l=(0, 4), p0=(0, 5), l0=(-3, 3)):
    V = np.zeros(resolution_V)

    etaArray = np.linspace(*etaBound, resolution_V[0])
    gammaArray = np.linspace(*gammaBound, resolution_V[1])
    for i, eta in enumerate(etaArray):
        print('Main Coordinate eta: ', i)
        for j, gamma in enumerate(gammaArray):
            print('gamma: ', j)
            V[i, j] = variance_single_transition(beam, mesh, r=eta, eta=gamma,
                                                 displacement_function=displacement_function,
                                                 width=width, k0=k0,
                                                 p=p, l=l, p0=p0, l0=l0)
    return V


def find_beam_waist(field, mesh=None):
    """
    wrapper for the beam waste finder. More details in knots_ML.center_beam_search
    """
    shape = np.shape(field)
    if mesh is None:
        mesh = fg.create_mesh_XY(xRes=shape[0], yRes=shape[1])
    width = find_width(field, mesh=mesh, width=shape[1] // 4, widthStep=1, print_steps=False)
    return width


def center_beam_finding(beam, mesh, stepXY=None, displacement_function=displacement_lateral,
                        p=(0, 5), l=(0, 4), p0=(0, 5), l0=(-3, 3),
                        width=1, k0=1, x=None, y=None):
    if stepXY is None:
        xArray, yArray = fg.arrays_from_mesh(mesh)
        # print(xArray)
        stepXY = xArray[1] - xArray[0], yArray[1] - yArray[0]
    if x is None:
        xArray, yArray = fg.arrays_from_mesh(mesh)
        x = xArray[len(xArray) // 2]
    if y is None:
        xArray, yArray = fg.arrays_from_mesh(mesh)
        y = yArray[len(yArray) // 2]

    def search(xFlag):
        nonlocal x, y, var0
        varIt = var0
        signX = 1
        correctWay = False
        while True:
            if xFlag:
                x = x + signX * stepXY[0]
            else:
                y = y + signX * stepXY[1]
            var = variance_single_transition(beam, mesh=mesh, displacement_function=displacement_function,
                                             r=fg.rho(x, y), eta=np.angle(x + 1j * y),
                                             p=p, l=l, p0=p0, l0=l0, width=width, k0=k0)
            # print(f'x={x}, y={y}, var={var}')
            if var < varIt:
                varIt = var
                correctWay = True
            else:
                if xFlag:
                    x = x - signX * stepXY[0]
                else:
                    y = y - signX * stepXY[1]
                if correctWay:
                    break
                else:
                    correctWay = True
                    signX *= -1

        return varIt

    var0 = variance_single_transition(beam, mesh=mesh, r=fg.rho(x, y), eta=np.angle(x + 1j * y),
                                      p=p, l=l, p0=p0, l0=l0, width=width, k0=k0)
    # print(f'var0={var0}')

    while True:
        search(xFlag=True)
        varXY = search(xFlag=False)
        if varXY == var0:
            return x, y
        else:
            var0 = varXY


def tilt_beam_finding(beam, mesh, stepEG=None, displacement_function=displacement_deflection,
                      p=(0, 5), l=(0, 4), p0=(0, 5), l0=(-3, 3),
                      width=1, k0=1, eta=0., gamma=0.):
    if stepEG is None:
        stepEG = 0.1, 0.1

    def search(etaFlag):
        nonlocal eta, gamma, var0
        varIt = var0
        signX = 1
        correctWay = False
        while True:
            if etaFlag:
                eta = eta + signX * stepEG[0]
            else:
                gamma = gamma + signX * stepEG[1]
            var = variance_single_transition(beam, mesh, eta, gamma,
                                             displacement_function=displacement_function,
                                             p=p, l=l, p0=p0, l0=l0, width=width, k0=k0)

            print(f'eta={eta}, gamma={gamma}, var={var}')
            if var < varIt:
                varIt = var
                correctWay = True
            else:
                if etaFlag:
                    eta = eta - signX * stepEG[0]
                else:
                    gamma = gamma - signX * stepEG[1]
                if correctWay:
                    break
                else:
                    correctWay = True
                    signX *= -1

        return varIt

    var0 = variance_single_transition(beam, mesh, eta, gamma,
                                      displacement_function=displacement_function,
                                      p=p, l=l, p0=p0, l0=l0, width=width, k0=k0)
    # print(f'var0={var0}')

    while True:
        search(etaFlag=True)
        varEG = search(etaFlag=False)
        if varEG == var0:
            return eta, gamma
        else:
            var0 = varEG


def field_interpolation(field, mesh=None, resolution=(100, 100),
                        xMinMax_frac=(1, 1), yMinMax_frac=(1, 1), fill_value=True):
    """
    Wrapper for the field interpolation fg.interpolation_complex
    :param resolution: new field resolution
    :param xMinMax_frac: new dimension for the field. (x_dim_old * frac)
    :param yMinMax_frac: new dimension for the field. (y_dim_old * frac)
    """
    shape = np.shape(field)
    if mesh is None:
        mesh = fg.create_mesh_XY(xRes=shape[0], yRes=shape[1])
    interpol_field = fg.interpolation_complex(field, mesh=mesh, fill_value=fill_value)
    xMinMax = int(shape[0] // 2 * xMinMax_frac[0]), int(shape[0] // 2 * xMinMax_frac[1])
    yMinMax = int(shape[1] // 2 * yMinMax_frac[0]), int(shape[1] // 2 * yMinMax_frac[1])
    xyMesh_interpol = fg.create_mesh_XY(
        xRes=resolution[0], yRes=resolution[1],
        xMinMax=xMinMax, yMinMax=yMinMax)
    return interpol_field(*xyMesh_interpol), xyMesh_interpol


def normalization_field(field):
    """
    Normalization of the field for the beam center finding
    """
    field_norm = field / np.sqrt(np.sum(np.abs(field) ** 2))
    return field_norm


def read_field_2D_single(path, field=None):
    """
    Function reads .mat 2D array from matlab and convert it into numpy array

    If field is None, it will try to find the field name automatically

    :param path: full path to the file
    :param field: the name of the column with the field you want to read
    """
    field_read = sio.loadmat(path, appendmat=False)
    if field is None:
        for field_check in field_read:
            if len(np.shape(np.array(field_read[field_check]))) == 2:
                field = field_check
                break
    return np.array(field_read[field])


def plot_field(field, save=None):
    """
    Function plots intensity and phase of the field in 1 plot.
    Just a small convenient wrapper
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    image1 = ax1.imshow(np.abs(field))
    ax1.set_title('|E|')
    plt.colorbar(image1, ax=ax1, shrink=0.4, pad=0.02, fraction=0.1)
    image2 = ax2.imshow(np.angle(field), cmap='jet')
    ax2.set_title('Phase(E)')
    plt.colorbar(image2, ax=ax2, shrink=0.4, pad=0.02, fraction=0.1)
    plt.tight_layout()
    if save is not None:
        fig.savefig(save, format='png')
    plt.show()


def main_field_processing(
        path,
        field_name=None,
        plotting=True,
        resolution_iterpol_center=(70, 70),
        stepXY=(3, 3),
        zero_pad=0,
        xMinMax_frac_center=(1, 1),
        yMinMax_frac_center=(1, 1),
        resolution_interpol_working=(150, 150),
        xMinMax_frac_working=(1, 1),
        yMinMax_frac_working=(1, 1),
        resolution_crop=(120, 120),
        moments_init=None,
        moments_center=None,
):
    """
    This function:
     1) reading the field from matlab file
     2) converting it into numpy array
     3) normalizing
     4) finding the beam waste
     5) rescaling the field, using the interpolation, for faster next steps
     6) finding the beam center
     7) rescaling field to the scale we want for 3D calculations
     8) removing the tilt and shift

    Assumption
    ----------
    Beam waist finder only works with a uniform grid (dx = dy)

    :param path: file name
    :param plotting: if we want to see the plots and extra information
    :param resolution_iterpol_center: resolution for the beam center finder
    :param xMinMax_frac_center: rescale ration along X axis for the beam center
    :param yMinMax_frac_center: rescale ration along Y axis for the beam center
    :param resolution_interpol_working: resolution for the final field before the cropping
    :param xMinMax_frac_working: rescale ration along X axis for the beam center
    :param yMinMax_frac_working: rescale ration along X axis for the beam center
    :param resolution_crop: actual final resolution of the field
    :param moments_init: the moments for the LG spectrum
    :param moments_center: the moments for the beam center finder
    :return: 2D complex field
    """
    # beam width search work only with x_res==y_res
    if moments_init is None:
        moments_init = {'p': (0, 6), 'l': (-4, 4)}
    if moments_center is None:
        moments_center = {'p0': (0, 6), 'l0': (-4, 4)}

    # reading file
    field_init_all = read_field_2D_single(path, field=field_name)
    xy_coordinates = []
    for i in range(0, field_init_all.shape[2], 1):
        field_init = field_init_all[:, :, i]
        if plotting or True:
            plot_field(field_init)
        continue
        # normalization
        field_norm = normalization_field(field_init)
        if plotting:
            plot_field(field_norm)

        # creating mesh
        mesh_init = fg.create_mesh_XY(xRes=np.shape(field_norm)[0], yRes=np.shape(field_norm)[1])

        # finding beam waste
        width = float(find_beam_waist(field_norm, mesh=mesh_init))
        if plotting or False:
            print(f'Approximate beam waist: {width}')

        # rescaling field
        field_interpol, mesh_interpol = field_interpolation(
            field_norm, mesh=mesh_init,
            resolution=(resolution_iterpol_center[0] - zero_pad * 2, resolution_iterpol_center[1] - zero_pad * 2),
            xMinMax_frac=xMinMax_frac_center, yMinMax_frac=yMinMax_frac_center
        )

        # padding with 0
        if zero_pad:
            field_interpol = np.pad(field_interpol, zero_pad, 'constant')
            shape = np.shape(field_norm)
            xMinMax = int(shape[0] // 2 * xMinMax_frac_center[0]), int(shape[0] // 2 * xMinMax_frac_center[1])
            yMinMax = int(shape[1] // 2 * yMinMax_frac_center[0]), int(shape[1] // 2 * yMinMax_frac_center[1])
            mesh_interpol = fg.create_mesh_XY(
                xRes=resolution_iterpol_center[0], yRes=resolution_iterpol_center[1],
                xMinMax=xMinMax, yMinMax=yMinMax)

        if plotting or False:
            plot_field(field_interpol)

        # rescaling the beam width
        scaling_factor = 1. / np.shape(field_norm)[0] * np.shape(field_interpol)[0]
        # print(width)
        # width *= scaling_factor
        # print(width, width / scaling_factor)
        ###################################################
        # plotting spec to select moments. .T because Danilo's code saving it like that
        if plotting or False:
            _ = LG_spectrum(field_interpol.T, **moments_init, mesh=mesh_interpol, plot=True, width=width, k0=1)

        # finding the beam center
        ## moments_init.update(moments_center)
        ## moments = moments_init
        # x, y, eta, gamma = beamFullCenter(
        #     field_interpol, mesh_interpol,
        #     stepXY=stepXY, stepEG=(1 / 180 * np.pi, 1 / 180 * np.pi),
        #     x=None, y=None, eta2=0., gamma=0.,
        #     **moments_center, threshold=1, width=width, k0=1, print_info=plotting
        # )
        x, y = center_beam_finding(field_interpol, mesh_interpol,
                                   stepXY=stepXY, displacement_function=displacement_lateral,
                                   **moments_center,
                                   width=width, k0=1, x=None, y=None)
        # xy_coordinates.append((x / scaling_factor, y / scaling_factor))
        # xy_coordinates.append((x, y, eta, gamma))
        xy_coordinates.append((x, y))
        eta = 0
        gamma = 0

        print(f'coordinates: {x, y}; tilt: {eta},{gamma}')
        # print(f'coordinates: {x, y}; tilt: {eta},{gamma}')
        # print(f'coordinates in the initial mesh: {x / scaling_factor, y / scaling_factor};'
        #       f' tilt: {eta},{gamma}')
        # x, y, eta, gamma = 0, 0, 0, 0

        # rescaling field to the scale we want for 3D calculations
        field_interpol2, mesh_interpol2 = field_interpolation(
            field_norm, mesh=mesh_init, resolution=resolution_interpol_working,
            xMinMax_frac=xMinMax_frac_working, yMinMax_frac=yMinMax_frac_working, fill_value=False
        )
        if plotting:
            plot_field(field_interpol2)

        # removing the tilt
        field_untilted = removeTilt(field_interpol2, mesh_interpol2, eta=-eta, gamma=gamma, k=1)
        if plotting:
            plot_field(field_untilted)

        # scaling the beam center
        shape = np.shape(field_untilted)
        scaling_factor2 = 1. / np.shape(field_interpol)[0] * shape[0]
        x = int(x * scaling_factor * scaling_factor2)
        y = int(y * scaling_factor * scaling_factor2)
        # x = int(x / np.shape(field_interpol)[0] * shape[0])
        # y = int(y / np.shape(field_interpol)[1] * shape[1])

        # cropping the beam around the center
        field_cropped = field_untilted[
                        shape[0] // 2 - x - resolution_crop[0] // 2:shape[0] // 2 - x + resolution_crop[0] // 2,
                        shape[1] // 2 - y - resolution_crop[1] // 2:shape[1] // 2 - y + resolution_crop[1] // 2]
        if plotting or True:
            plot_field(field_cropped)

        # selecting the working field and mesh
        mesh = fg.create_mesh_XY(xRes=np.shape(field_cropped)[0], yRes=np.shape(field_cropped)[1])
        field = field_cropped
        # print(f'field finished: {path[-20:]}')
    print(xy_coordinates)
    np.save('coordinates_unmod2.npy', np.array(xy_coordinates))
    # return field, mesh


####################################################################################


if __name__ == '__main__':
    finding_center_danilos_files = True
    if finding_center_danilos_files:
        # test = np.load('coordinates_unmod2.npy')
        # print(test.shape)
        # np.savetxt('centers_unmod2.txt', np.round(test)[1::2])
        # np.savetxt('centers2.txt', np.round(test * 2) / 2)
        # exit()
        # path = f'Uz_trefoilmod_exp.mat'
        path = f'Uz_trefoilunmod_exp.mat'
        main_field_processing(
            path,
            field_name='Uz',
            plotting=False,
            resolution_iterpol_center=(80, 80),
            stepXY=(1, 1),
            zero_pad=0,
            xMinMax_frac_center=(-1., 1.),
            yMinMax_frac_center=(-1., 1.),
            resolution_interpol_working=(80, 80),
            xMinMax_frac_working=(-1, 1),
            yMinMax_frac_working=(-1, 1),
            resolution_crop=(65, 65),
            # moments_init={'p': (0, 10), 'l': (-7, 5)},
            moments_init={'p': (0, 6), 'l': (-5, 3)},
            # moments_center={'p': (0, 10), 'l': (-7, 5)})
            moments_center={'p': (0, 5), 'l': (-4, 2)})

    dots_building_ = False
    resolution_crop = (140, 140)
    empty = np.array([[0, 0, 0]])
    if dots_building_:
        path = f'Uz_trefoilsingleunmod.mat'
        field_init_all = read_field_2D_single(path, field='Uz')
        for i in range(np.shape(field_init_all)[2]):
        # for i in range(2):
            field2D = field_init_all[:, :, i]
            # plt.imshow(np.angle(field2D))
            # plt.show()
            # plt.imshow(np.abs(field2D))
            # plt.show()
            # exit()
            dots_raw, dots_filtered = main_dots_building(
                field2D=field2D,
                plotting=False,
                dz=22,
                steps_both=27,
                resolution_crop=resolution_crop,
                r_crop=140
            )
            empty = np.concatenate((empty, dots_raw), axis=0)

        empty = np.delete(empty, 0, 0)
        np.save('all_dots.npy', empty)
        # exit()
        # file_save_dots_raw = directory_field_saved_dots + 'raw_' + file[:-4] + '.npy'
        # file_save_dots_filtered = directory_field_saved_dots + 'filtered_' + file[:-4] + '.npy'
        dp.plotDots(empty, empty, color='black', show=True, size=15,
                    save=None)
                    # save=directory_field_saved_plots + file[:-4] + '_3D.html')
        # np.save(file_save_dots_raw, dots_raw)
        # np.save(file_save_dots_filtered, dots_filtered)