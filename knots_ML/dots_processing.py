import my_functions.functions_general as fg
import my_functions.singularities as sing
import my_functions.plotings as pl
import my_functions.beams_and_pulses as bp
import numpy as np
import pickle


def buildSaveTrefoil():
    xyzMesh = fg.create_mesh_XYZ(2.1, 2.1, 0.7, 40, 40, 40, zMin=None)
    beam = bp.LG_combination(*xyzMesh,
                             coefficients=[1.71, -5.66, 6.38, -2.30, -4.36],
                             modes=[(0, 0), (0, 1), (0, 2), (0, 3), (3, 0)],
                             width=[1, 1, 1, 1, 1])
    dotsDict, dotsInitial = sing.get_singularities(np.angle(beam), axesAll=True, returnDict=True)
    with open("dots_dict.pkl", 'wb') as f:
        pickle.dump(dotsDict, f)
    with open('dots.npy', 'wb') as f:
        np.save(f, dotsInitial)


def loadDots():
    with open("dots_dict.pkl", 'rb') as f:
        dots_dict = pickle.load(f)
    with open('dots.npy', 'rb') as f:
        dots = np.load(f)
    return dots, dots_dict


def plotDots(dots, dots_bound=None, show=True, color='black', size=15, width=185, fig=None,
             save=None):
    """
    Function plots the array of dots in a beautiful and interactive way in your browser.
    Plots both numpy array and dict
    :param dots: array of dots
    :param dots_bound: which dots to use to get the box frames. By default it uses the dots itself,
        but if you want to make the frames the same with other plots, you can use other dots here, same
        for all plots.
    :param show: plotting in the browser. Can be turned off to add extra plots on top
    :param color: color of the dots
    :param size: size of the dots
    :param width: width of the shell of the dots (for a better visualization)
    :param fig: figure for the combination with other plots
    :return: fig
    """
    colorLine = 'white'
    # colorLine = 'black'
    if isinstance(dots, dict):
        dots = np.array([dot for dot in dots])
    if isinstance(dots_bound, dict):
        dots_bound = np.array([dot for dot in dots_bound])
    if dots_bound is None:
        dots_bound = dots
    if fig is None:
        fig = pl.plot_3D_dots_go(dots, marker={'size': size, 'color': color,
                                               'line': dict(width=width, color=colorLine)})
    else:
        pl.plot_3D_dots_go(dots, fig=fig, marker={'size': size, 'color': color,
                                                  'line': dict(width=width, color=colorLine)})
    pl.box_set_go(fig, mesh=None, autoDots=dots_bound, perBox=0.01)
    if save is not None:
        fig.write_html(save)
    if show:
        fig.show()
    return fig

def plotDots_Hopf(dots, dots_bound=None, show=True, color='black', size=15, width=185, fig=None,
             save=None, lines=True, **kwargs):
    """
    Function plots the array of dots in a beautiful and interactive way in your browser.
    Plots both numpy array and dict
    :param dots: array of dots
    :param dots_bound: which dots to use to get the box frames. By default it uses the dots itself,
        but if you want to make the frames the same with other plots, you can use other dots here, same
        for all plots.
    :param show: plotting in the browser. Can be turned off to add extra plots on top
    :param color: color of the dots
    :param size: size of the dots
    :param width: width of the shell of the dots (for a better visualization)
    :param fig: figure for the combination with other plots
    :return: fig
    """
    colorLine = 'black'
    # colorLine = 'white'
    if isinstance(dots, dict):
        dots = np.array([dot for dot in dots])
    if isinstance(dots_bound, dict):
        dots_bound = np.array([dot for dot in dots_bound])
    if dots_bound is None:
        dots_bound = dots
    if fig is None:
        fig = pl.plot_3D_dots_go(dots, marker={'size': size, 'color': color,
                                               'line': dict(width=width, color=colorLine)}, **kwargs)
    else:
        pl.plot_3D_dots_go(dots, fig=fig, marker={'size': size, 'color': color,
                                                  'line': dict(width=width, color=colorLine)}, **kwargs)
    # pl.box_set_go(fig, mesh=None, autoDots=dots_bound, perBox=0.01, lines=lines, **kwargs)
    pl.box_set_go(fig, mesh=None, autoDots=dots_bound, perBox=0.01, lines=lines)
    if save is not None:
        fig.write_html(save)
    if show:
        fig.show()
    return fig

def neighboursDots(x_c, y_c, z_c, dots_dict):
    """
    Algorithm finds all dots in 3x3x3 cube around the dot, return their count and the dots themselves
    :param x_c: (X, y, z) - dot, which neighbours are explored
    :param y_c: (x, Y, z) - dot, which neighbours are explored
    :param z_c: (x, y, Z) - dot, which neighbours are explored
    :param dots_dict: dictionary with all the dots
    :return: number of dots in the square 3x3x3 with the array of dots in it
    """
    count = 0
    dots = []
    for x in [x_c - 1, x_c, x_c + 1]:
        for y in [y_c - 1, y_c, y_c + 1]:
            for z in [z_c - 1, z_c, z_c + 1]:
                if (x, y, z) in dots_dict:
                    count += 1
                    dots.append((x, y, z))
    return count, dots


def filter_ZeroNeighbours(dots_dict):
    """
    Algorithm finds all dots with only 2 neighbors and checks if the trajectory is not
    too sharp. All the good dots are removed from dots_dict (using the shallow copy)
    :param dots_dict: dictionary with all the dots
    :return: an array of new dots (removed and not) [(1,2,3), (3,4,5)...]
    """
    new_dots = []
    dots_problem = []
    for dot in dots_dict:
        count, dots = neighboursDots(*dot, dots_dict)
        if count == 1:
            # print(f'problem dot {dot}, (removed) (filterZeroNeighbours)')
            dots_problem.append(dot)
    # removing dots which are definitely separated from other algorithms
    # and are definitely good and viable
    for dot in dots_problem:
        dots_dict.pop(dot)
    # for dot in new_dots:
    #     count, dots = neighboursDots(*dot, new_dots)
    #     if count == 3:
    #         dots_dict.pop(dot)
    return np.array(new_dots), np.array(dots_problem)


def filter_OneNeighbours(dots_dict):
    """
    Algorithm finds all dots with only 2 neighbors and checks if the trajectory is not
    too sharp. All the good dots are removed from dots_dict (using the shallow copy)
    :param dots_dict: dictionary with all the dots
    :return: an array of new dots (removed and not) [(1,2,3), (3,4,5)...]
    """
    new_dots = []
    dots_problem = []
    for dot in dots_dict:
        count, dots = neighboursDots(*dot, dots_dict)
        if count == 2:
            # print(f'problem dot {dot}, (removed) (filterOneNeighbours)')
            dots_problem.append(dot)
    # removing dots which are definitely separated from other algorithms
    # and are definitely good and viable
    for dot in dots_problem:
        dots_dict.pop(dot)
    # for dot in new_dots:
    #     count, dots = neighboursDots(*dot, new_dots)
    #     if count == 3:
    #         dots_dict.pop(dot)
    return np.array(new_dots), np.array(dots_problem)


def filter_TwoNeighbours(dots_dict):
    """
    Algorithm finds all dots with only 2 neighbors and checks if the trajectory is not
    too sharp. All the good dots are removed from dots_dict (using the shallow copy)
    :param dots_dict: dictionary with all the dots
    :return: an array of new dots (removed and not) [(1,2,3), (3,4,5)...]
    """
    new_dots = []
    dots_problem = []
    dots_to_remove = []
    for dot in dots_dict:
        count, dots = neighboursDots(*dot, dots_dict)
        if count == 3:
            check = np.sum(dots, axis=0) / 3 - dot
            if np.sum(np.abs(check)) > 1:
                # print(f'problem dot {dot}, trajectory {check} (removed) (filterTwoNeighbours)')
                dots_problem.append(dot)
            else:
                dots_to_remove.append(dot)
                new_dots.append(dot)
    # removing dots which are definitely separated from other algorithms
    # and are definitely good and viable
    for dot in dots_problem:
        dots_dict.pop(dot)
    for dot in dots_to_remove:
        dots_dict.pop(dot)
    # for dot in new_dots:
    #     count, dots = neighboursDots(*dot, new_dots)
    #     if count == 3:
    #         dots_dict.pop(dot)
    return np.array(new_dots), np.array(dots_problem)


def filter_ThreeNeighbours(dots_dict):
    """
    Algorithm finds all dots with only 2 neighbors and checks if the trajectory is not
    too sharp. All the good dots are removed from dots_dict (using the shallow copy)
    :param dots_dict: dictionary with all the dots
    :return: an array of new dots (removed and not) [(1,2,3), (3,4,5)...]
    """
    new_dots = [(0, 0, 0)]
    dots_problem = []
    for dot in dots_dict:
        count, dots = neighboursDots(*dot, dots_dict)
        if count == 4:
            # print(f'{np.sum(dots, axis=0) / 4 - dot}, dot {dot}')
            # check = np.sum(dots, axis=0) / 3 - dot
            # if np.sum(np.abs(check)) > 1:
            #     print(f'problem dot {dot}, trajectory {check} (filter_TwoNeighbours)')
            #     continue
            new_dots.append(dot)
    # надо убедиться, что в паре обе точки не имеют соседей
    new_dots_pairs = set()
    new_dots_pairs_temp = set()
    dots_to_remove = []
    for dot in new_dots:
        count, dots = neighboursDots(*dot, new_dots)
        if count == 2:
            count_dot2, dots3 = neighboursDots(*(dots[dots.index(dot) - 1]), new_dots)
            if count_dot2 == 2:
                new_dots_pairs.add(tuple(np.sum(dots, axis=0) / 2))
                dots_to_remove.append(dot)
                # dots_dict.pop(dot)
                # new_dots.remove(dot)
            elif count_dot2 == 3:
                # print(dot, dots3)
                new_dots_pairs_temp.add(tuple(np.sum(dots3, axis=0) / 3))
                for dot_ in dots3:
                    dots_to_remove.append(dot_)
                # dots_dict.pop(dot)
                # new_dots.remove(dot)
    new_dots = []
    for dot in dots_to_remove:
        try:
            dots_dict.pop(dot)
        except KeyError:
            pass
            # print('filter_ThreeNeighbours has a small problem with deleting items')
    for dot in set.union(new_dots_pairs, new_dots_pairs_temp):
        new_dots.append(dot)
    # return np.array(list(new_dots_pairs))
    # return np.array(list(new_dots_pairs_temp)), np.array(dots_problem)
    return np.array(list(new_dots)), np.array(dots_problem)


def filter_PlaneThreeClusters(dots_dict):
    """
    Algorithm finds all 3 clusters in planes and return an average dot.
    0000
    0**0
    0*00
    0000
    All the good dots are removed from dots_dict (using the shallow copy)
    :param dots_dict: dictionary with all the dots
    :return: an array of new dots (removed and not) [(1,2,3), (3,4,5)...]
    """

    def neighbours_InPlaneFourDots(x_c, y_c, z_c, dots_dict, direction):
        """
        Algorithm finds all dots in 2x2 plane around the dot, return their count and the dots themselves
        00
        *0
        :param x_c: (X, y, z) - dot, which neighbours are explored
        :param y_c: (x, Y, z) - dot, which neighbours are explored
        :param z_c: (x, Y, z) - dot, which neighbours are explored
        :param dots_dict: dictionary with all the dots
        :return: number of dots in the square 3x3x3 with the array of dots in it
        """
        count = 0
        dots = []
        if direction == 'z':
            for x_ in [x_c, x_c + 1]:
                for y_ in [y_c, y_c + 1]:
                    if (x_, y_, z_c) in dots_dict:
                        count += 1
                        dots.append((x_, y_, z_c))
        elif direction == 'y':
            for x_ in [x_c, x_c + 1]:
                for z_ in [z_c, z_c + 1]:
                    if (x_, y_c, z_) in dots_dict:
                        count += 1
                        dots.append((x_, y_c, z_))
        elif direction == 'x':
            for z_ in [z_c, z_c + 1]:
                for y_ in [y_c, y_c + 1]:
                    if (x_c, y_, z_) in dots_dict:
                        count += 1
                        dots.append((x_c, y_, z_))
        return count, dots

    def planeEmptyCheck_helper(x_c, y_c, z_c, dots_dict, direction):
        """
        Algorithm checks if there are any neighbours in 4x4 plane around the dot 2x2 dot
        0000
        0**0
        0**0
        0000 - TRUE
        :param x_c: (X, y, z) - dot, which neighbours are explored
        :param y_c: (x, Y, z) - dot, which neighbours are explored
        :param z_c: (x, Y, z) - dot, which neighbours are explored
        :param dots_dict: dictionary with all the dots
        :return: True if the area is clear
        """

        def boundary_square_44(a, b):
            return [(a - 1, b - 1), (a - 1, b), (a - 1, b + 1), (a - 1, b + 2),
                    (a, b + 2), (a + 1, b + 2), (a + 2, b + 2),
                    (a + 2, b + 1), (a + 2, b), (a + 2, b - 1),
                    (a + 1, b - 1), (a, b - 1)]

        if direction == 'z':
            for x_, y_ in boundary_square_44(x_c, y_c):
                if (x_, y_, z_c) in dots_dict:
                    return False
        elif direction == 'y':
            for x_, z_ in boundary_square_44(x_c, z_c):
                if (x_, y_c, z_) in dots_dict:
                    return False
        elif direction == 'x':
            for z_, y_ in boundary_square_44(z_c, y_c):
                if (x_c, y_, z_) in dots_dict:
                    return False

        return True

    new_dots = []
    dots_problem = []
    dots_to_remove = []
    for dot in dots_dict:
        # z
        count, dots = neighbours_InPlaneFourDots(*dot, dots_dict, direction='z')
        if count == 3:
            if planeEmptyCheck_helper(*dot, dots_dict, direction='z'):
                new_dots.append(tuple(np.sum(dots, axis=0) / 3))
                for dot_ in dots:
                    dots_to_remove.append(dot_)
        # y
        count, dots = neighbours_InPlaneFourDots(*dot, dots_dict, direction='y')
        if count == 3:
            if planeEmptyCheck_helper(*dot, dots_dict, direction='y'):
                new_dots.append(tuple(np.sum(dots, axis=0) / 3))
                for dot_ in dots:
                    dots_to_remove.append(dot_)
        # x
        count, dots = neighbours_InPlaneFourDots(*dot, dots_dict, direction='x')
        if count == 3:
            if planeEmptyCheck_helper(*dot, dots_dict, direction='x'):
                new_dots.append(tuple(np.sum(dots, axis=0) / 3))
                for dot_ in dots:
                    dots_to_remove.append(dot_)
    for dot in set(dots_to_remove):
        dots_dict.pop(dot)
    return np.array(new_dots), np.array(dots_problem)


def filter_PlaneTwoClusters(dots_dict):
    """
    Algorithm finds all 2 horizontal cluster in 3 planes and return average dots.
    Verticals are included since they are horizontal in z-planes.
    0000
    0**0
    0000
    All the good dots are removed from dots_dict (using the shallow copy)
    :param dots_dict: dictionary with all the dots
    :return: an array of new dots (removed and not) [(1,2,3), (3,4,5)...]
    """

    def neighbours_InPlaneTwoDots(x_c, y_c, z_c, dots_dict, direction):
        """
        Algorithm finds all dots in 2x2 plane around the dot, return their count and the dots themselves
        00
        *0
        :param x_c: (X, y, z) - dot, which neighbours are explored
        :param y_c: (x, Y, z) - dot, which neighbours are explored
        :param z_c: (x, Y, z) - dot, which neighbours are explored
        :param dots_dict: dictionary with all the dots
        :return: number of dots in the square 3x3x3 with the array of dots in it
        """
        count = 0
        dots = []
        if direction == 'x':
            for x_ in [x_c, x_c + 1]:
                if (x_, y_c, z_c) in dots_dict:
                    count += 1
                    dots.append((x_, y_c, z_c))
        elif direction == 'y':
            for y_ in [y_c, y_c + 1]:
                if (x_c, y_, z_c) in dots_dict:
                    count += 1
                    dots.append((x_c, y_, z_c))
        elif direction == 'z':
            for z_ in [z_c, z_c + 1]:
                if (x_c, y_c, z_) in dots_dict:
                    count += 1
                    dots.append((x_c, y_c, z_))
        return count, dots

    def planeEmptyCheck_helper_twoDots(x_c, y_c, z_c, dots_dict, direction):
        """
        Algorithm checks if there are any neighbours in 3x4 plane around the dots 2x1
        0000
        0**0
        0000 - TRUE
        :param x_c: (X, y, z) - dot, which neighbours are explored
        :param y_c: (x, Y, z) - dot, which neighbours are explored
        :param z_c: (x, Y, z) - dot, which neighbours are explored
        :param dots_dict: dictionary with all the dots
        :return: True if the area is clear
        """

        def boundary_square_34(a, b):
            return [(a - 1, b - 1), (a - 1, b), (a - 1, b + 1),
                    (a, b + 1), (a + 1, b + 1),
                    (a + 1, b), (a + 1, b - 1),
                    (a, b - 1)]

        if direction == 'x':
            for x_, y_ in boundary_square_34(x_c, y_c):
                if (x_, y_, z_c) in dots_dict:
                    for x__, z__ in boundary_square_34(x_c, z_c):
                        if (x__, y_c, z__) in dots_dict:
                            return False
        elif direction == 'y':
            for y_, x_ in boundary_square_34(y_c, x_c):
                if (x_, y_, z_c) in dots_dict:
                    for y__, z__ in boundary_square_34(y_c, z_c):
                        if (x_c, y__, z__) in dots_dict:
                            return False
        elif direction == 'z':
            for z_, x_ in boundary_square_34(z_c, x_c):
                if (x_, y_c, z_) in dots_dict:
                    for x__, y__ in boundary_square_34(z_c, y_c):
                        if (x__, y__, z_c) in dots_dict:
                            return False

        return True

    new_dots = []
    dots_problem = []
    dots_to_remove = []
    for dot in dots_dict:
        # z
        count, dots = neighbours_InPlaneTwoDots(*dot, dots_dict, direction='z')
        if count == 2:
            if planeEmptyCheck_helper_twoDots(*dot, dots_dict, direction='z'):
                new_dots.append(tuple(np.sum(dots, axis=0) / 2))
                for dot_ in dots:
                    dots_to_remove.append(dot_)
        # y
        count, dots = neighbours_InPlaneTwoDots(*dot, dots_dict, direction='y')
        if count == 2:
            if planeEmptyCheck_helper_twoDots(*dot, dots_dict, direction='y'):
                new_dots.append(tuple(np.sum(dots, axis=0) / 2))
                for dot_ in dots:
                    dots_to_remove.append(dot_)
        # x
        count, dots = neighbours_InPlaneTwoDots(*dot, dots_dict, direction='x')
        if count == 2:
            if planeEmptyCheck_helper_twoDots(*dot, dots_dict, direction='x'):
                new_dots.append(tuple(np.sum(dots, axis=0) / 2))
                for dot_ in dots:
                    dots_to_remove.append(dot_)
    for dot in set(dots_to_remove):
        dots_dict.pop(dot)
    return np.array(new_dots), np.array(dots_problem)


def filter_PlaneTwoClusters_Vertical(dots_dict):
    pass


def globalFilterDots(dots_dict, single_dot=False, print_=False):
    """
    Applaying all the filters to the dictionary of dots
    :param dots_dict: dictionary with all the dots
    :return: [[dots from 1st filter], [dots from 2nd filter], ... [(1,2,3),(2,3,4), ...], ...]
    """
    import copy
    dots_dict_copy = copy.deepcopy(dots_dict)
    dots_final = []
    if print_:
        print('raw: ', len(dots_dict_copy))

    dots_final.append(filter_ZeroNeighbours(dots_dict_copy)[0])
    if print_:
        print('0. zero neighbour: ', len(dots_dict_copy))
    if not single_dot:
        dots_final.append(filter_OneNeighbours(dots_dict_copy)[0])
        if print_:
            print('1. one neighbour: ', len(dots_dict_copy))
        dots_final.append(filter_TwoNeighbours(dots_dict_copy)[0])
        if print_:
            print('2. two neighbours: ', len(dots_dict_copy))
        dots_final.append(filter_ThreeNeighbours(dots_dict_copy)[0])
        if print_:
            print('3. three neighbours: ', len(dots_dict_copy))
        dots_final.append(filter_PlaneThreeClusters(dots_dict_copy)[0])
        if print_:
            print('4. plane 3 dots: ', len(dots_dict_copy))
        dots_final.append(filter_PlaneTwoClusters(dots_dict_copy)[0])
        if print_:
            print('5. plane 2 dots (FINAL LENGTH): ', len(dots_dict_copy))
    return dots_final, dots_dict_copy


def filtered_dots(dots_dict, single_dot=False):
    """
    It's a wrapper for globalFilterDots()
    This function places all the filtered dots into 1 array. It checks, if the arrays are empty as well.
    :param dots_dict: dictionary with all the dots
    """
    dots_final, dots_left = globalFilterDots(dots_dict, single_dot=single_dot)
    dots_final_combined = []
    for dots_arrays in dots_final:
        if len(dots_arrays) != 0:
            for dot in dots_arrays:
                dots_final_combined.append(dot)
    for dot in dots_left:
        dots_final_combined.append(dot)
    dots_final_combined = np.array(dots_final_combined)
    return dots_final_combined


# def one_plane_propagator(fieldPlane, dz, stepsNumber, n0=1, k0=1):  # , shapeWrong=False
#     # if shapeWrong is not False:
#     #     if shapeWrong is True:
#     #         print(f'using the middle plane in one_plane_propagator (shapeWrong = True)')
#     #         fieldPlane = fieldPlane[:, :, np.shape(fieldPlane)[2] // 2]
#     #     else:
#     #         fieldPlane = fieldPlane[:, :, np.shape(fieldPlane)[2] // 2 + shapeWrong]
#
#     fieldPropPLus = fg.propagator_split_step_3D(fieldPlane, dz=dz, zSteps=stepsNumber, n0=n0, k0=k0)
#     return fieldPropPLus

# xyzMesh = fg.create_mesh_XYZ(2.1, 2.1, 0.7, 50, 50, 50, zMin=None)
# w = 1.7
# a00 = 1 - 2 * w ** 2 + 2 * w ** 4
# a01 = 2 * w ** 2 - 4 * w ** 4
# a02 = 2 * w ** 4
# a20 = 4 * np.sqrt(2) * w ** 2
# hopfCoeff = [a00, a01, a02, a20]
#
# xyMesh = fg.create_mesh_XY(xMinMax=(-4.5,4.5), yMinMax=(-4.5, 4.5),xRes=90, yRes=90)
# beam = bp.LG_combination(*xyMesh, z0=0.35,
#                          coefficients=hopfCoeff,
#                          modes=[(0, 0), (0, 1), (0, 2), (2, 0)],
#                          width=[1, 1, 1, 1, 1])
# f = 1 * 0.5
# beam *= np.exp(-1j * 1/f * (xyMesh[0] ** 2 + xyMesh[1] ** 2))
# beam = one_plane_propagator(beam, dz=0.3, stepsNumber=80, n0=1, k0=1)
# pl.plot_2D(np.angle(beam[:, :, 0]))
# # pl.plot_2D(np.angle(beam[:, :, 59]))
# # exit()
# dotsDict, dotsInitial = sing.get_singularities(np.angle(beam), axesAll=True, returnDict=True)
# plotDots(dotsInitial, dotsInitial, color='blue')
# exit()
# beam = bp.LG_combination(*xyzMesh,
#                          coefficients=hopfCoeff,
#                          modes=[(0, 0), (0, 1), (0, 2), (2, 0)],
#                          width=[1, 1, 1, 1, 1])
# dotsDict, dotsInitial = sing.get_singularities(np.angle(beam), axesAll=True, returnDict=True)
# plotDots(dotsInitial, dotsInitial, color='blue')
# exit()

if __name__ == '__main__':
    buildSaveTrefoil()
    dots, dots_dict = loadDots()
    dots_final, dots_left = globalFilterDots(dots_dict)
    dots_final_combined = filtered_dots(dots_dict)
    # print(dots_final_combined)
    # exit()
    # fig = plotDots(np.array(list(dots_left)), dots, color='black', show=False)
    fig = plotDots(dots_final_combined, dots, color='blue', show=False, size=20)
    # fig = plotDots(dots, dots, color='Black', show=False, size=20)
    # plotDots(np.array(list(dots_left)), dots, color='red', show=False, fig=fig)
    # plotDots(dots_final[1], dots, color='blue', show=False, fig=fig)
    # plotDots(dots_final[0], dots, color='blue', show=False, fig=fig)
    # plotDots(dots_final[4], dots, color='blue', show=False, fig=fig)
    fig.show()
    plot_trefoil = False
    if plot_trefoil:

        # pl.plot_2D(np.abs(beam[:,:,40//2]))
        # sing.Jz_calc_no_conj(beam[:,:,40//2])

        # dotsInitial = sing.get_singularities(np.angle(beam), axesAll=True, returnDict=False)
        # fig = pl.plot_3D_dots_go(dots, marker={'size': 15, 'color': 'black', 'line': dict(width=185,
        #                                                                                   color='white')})
        # pl.box_set_go(fig, mesh=None, autoDots=dots, perBox=0.05)
        # fig.show()
        # exit()
        #
        # with open("dots_dict.pkl", 'wb') as f:
        #     pickle.dump(dotsDict, f)
        with open("dots_dict.pkl", 'rb') as f:
            dotsDict = pickle.load(f)
        print(dotsDict)
        exit()
        dotsPerfectOnly = [[0, 0, 0]]
        # np.save('trefoil_50.npy', dotsInitial)
        # np.save('trefoil_50.npy', dotsInitial)
        with open('trefoil_50.npy', 'rb') as f:
            dotsInitial = np.load(f)

        # exit()

        # find_the_group = True
        # if find_the_group

        interpolation_lines = False
        if interpolation_lines:
            dotsInitial = np.array([[41., 57., 92.], [39., 57., 92.4], [43., 57., 91.2], [23., 47., 119.6],
                                    [27., 47., 115.2], [25., 45., 122.], [25., 49., 114.], [29., 49., 109.6],
                                    [29., 47., 114.4], [27., 49., 111.2], [23., 45., 125.6], [31., 49., 106.8],
                                    [25., 47., 117.6], [39., 55., 95.6], [37., 53., 98.4], [35., 55., 96.8],
                                    [33., 53., 116.8], [23., 43., 132.8], [25., 41., 145.2], [25., 43., 133.6],
                                    [29., 51., 106.4], [31., 53., 121.2], [31., 51., 104.8], [41., 55., 93.6],
                                    [33., 51., 103.6], [35., 53., 99.6], [37., 55., 96.4]])

            x_sample = dotsInitial[:, 0]
            y_sample = dotsInitial[:, 1]
            z_sample = dotsInitial[:, 2]
            # import scipy.optimize as optimize
            # params, cov = optimize.curve_fit(fit, np.transpose(XY), Z, guess)
            from scipy import interpolate

            tck = interpolate.splprep([x_sample, y_sample, z_sample], s=2)[0]

            x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
            import matplotlib.pyplot as plt

            fig2 = plt.figure(2)
            ax3d = fig2.add_subplot(111, projection='3d')
            ax3d.plot(x_knots, y_knots, z_knots, 'g-')
            ax3d.plot(x_sample, y_sample, z_sample, 'r*')
            fig2.show()
            plt.show()
            exit()

        straight_lines = False
        if straight_lines:
            for dot, value in dotsDict.items():
                neighbours = 0
                xD, yD, zD = dot
                if 1:
                    if dotsDict.get((xD - 1, yD, zD), 10) + dotsDict.get((xD + 1, yD, zD), 10) < 5:
                        dotsPerfectOnly.append(dot)
                    if dotsDict.get((xD, yD - 1, zD), 10) + dotsDict.get((xD, yD + 1, zD), 10) < 5:
                        dotsPerfectOnly.append(dot)
                    if dotsDict.get((xD, yD, zD - 1), 10) + dotsDict.get((xD, yD, zD + 1), 10) < 5:
                        dotsPerfectOnly.append(dot)
                if 1:
                    if dotsDict.get((xD - 1, yD - 1, zD - 1), 10) + dotsDict.get((xD + 1, yD + 1, zD + 1), 10) < 5:
                        dotsPerfectOnly.append(dot)
                    if dotsDict.get((xD + 1, yD - 1, zD - 1), 10) + dotsDict.get((xD - 1, yD + 1, zD + 1), 10) < 5:
                        dotsPerfectOnly.append(dot)
                    if dotsDict.get((xD - 1, yD + 1, zD - 1), 10) + dotsDict.get((xD + 1, yD - 1, zD + 1), 10) < 5:
                        dotsPerfectOnly.append(dot)
                    if dotsDict.get((xD + 1, yD + 1, zD + 1), 10) + dotsDict.get((xD - 1, yD - 1, zD - 1), 10) < 5:
                        dotsPerfectOnly.append(dot)
                if 1:
                    if dotsDict.get((xD - 1, yD, zD - 1), 10) + dotsDict.get((xD + 1, yD, zD + 1), 10) < 5:
                        dotsPerfectOnly.append(dot)
                    if dotsDict.get((xD + 1, yD, zD - 1), 10) + dotsDict.get((xD - 1, yD, zD + 1), 10) < 5:
                        dotsPerfectOnly.append(dot)
                    if dotsDict.get((xD, yD - 1, zD - 1), 10) + dotsDict.get((xD, yD + 1, zD + 1), 10) < 5:
                        dotsPerfectOnly.append(dot)
                    if dotsDict.get((xD, yD + 1, zD - 1), 10) + dotsDict.get((xD, yD - 1, zD + 1), 10) < 5:
                        dotsPerfectOnly.append(dot)
                    if dotsDict.get((xD - 1, yD + 1, zD), 10) + dotsDict.get((xD + 1, yD - 1, zD), 10) < 5:
                        dotsPerfectOnly.append(dot)
                    if dotsDict.get((xD - 1, yD - 1, zD), 10) + dotsDict.get((xD + 1, yD + 1, zD), 10) < 5:
                        dotsPerfectOnly.append(dot)

            # for x in [xD - 1, xD, xD + 1]:
            #     for y in [yD - 1, yD, yD + 1]:
            #         for z in [zD - 1, zD, zD + 1]:
            #             if dotsDict.get((x, y, z), 0) != 0:
            #                 neighbours += 1
            # if neighbours == 2:
            #     dotsPerfectOnly.append(dot)

        dotsGood = np.array(dotsPerfectOnly)

        # print(len(dotsInitial))
        # dotsCutSeparated = sing.dots_filter(dotsInitial, checkValue=np.sqrt(2)*1.01, checkNumber=1)
        # print(len(dotsInitial))
        # dotsCutBigCluster = sing.dots_filter(dotsCutSeparated, checkValue=4, checkNumber=3)
        # dotsFiltered = sing.dots_dens_reduction(dotsCutBigCluster, checkValue=2.5, checkNumber=3)
        # dots = dotsFiltered
        # pl.plot_2D(np.abs(beam[:, :, zRes//2]))
        # pl.plot_scatter_3D(dots[:, 0], dots[:, 1], dots[:, 2], show=True)
        fig = pl.plot_3D_dots_go(dotsInitial, marker={'size': 15, 'color': 'black', 'line': dict(width=185,
                                                                                                 color='white')})
        pl.plot_3D_dots_go(dotsGood, fig=fig, marker={'size': 15, 'color': 'blue', 'line': dict(width=185,
                                                                                                color='white')})
        pl.box_set_go(fig, mesh=None, autoDots=dotsDict, perBox=0.05)
        fig.show()
