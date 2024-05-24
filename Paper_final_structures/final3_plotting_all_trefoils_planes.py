import matplotlib.pyplot as plt

from functions_based import *
import my_functions.plotings as pl
import knots_ML.data_generation as dg

import numpy as np
from scipy.special import assoc_laguerre
import my_functions.functions_general as fg
import math
import scipy.io


font_numbers = 26
font_label = 26
font_text = 24
fonts = (font_numbers, font_label, font_text)
family = 'Sans-serif'
cmapF = 'hsv'
cmapE = 'hot'

res_x_3D = 551
x_lim_3D = (-6.4, 6.4)
w = 1.3
w_real = 1.6


def plot_field(field, axes=True, titles=('Amplitude', 'Phase'), cmapE='afmhot', cmapF='jet', x_lim=(-1, 1),
               show_axis=True, fonts=fonts):
    if len(np.shape(field)) == 3:
        field2D = field[:, :, np.shape(field)[2] // 2]
    else:
        field2D = field
    show_axis = show_axis
    plt.rc('font', size=fonts[2], family=family)  # controls default text sizes
    plt.rc('axes', titlesize=fonts[1])  # fontsize of the axes title
    plt.rc('axes', labelsize=fonts[1])  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=fonts[0])  # fontsize of the tick labels
    plt.rc('ytick', labelsize=fonts[0])  # fontsize of the tick labels
    plt.rc('legend', fontsize=100)  # legend fontsize
    plt.rc('figure', titlesize=100)  # fontsize of the figure title
    cbar_size = fonts[0]  # colorbar fontsize
    plt.subplots(1, 2, figsize=(11, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(field2D.T), cmap=cmapE, interpolation='nearest', vmin=0,
               extent=[*x_lim, *x_lim])
    plt.xticks(())
    plt.yticks(())
    if show_axis:
        # plt.xticks(np.arange(-4, 5, 2))
        # plt.yticks(np.arange(-4, 5, 2))
        plt.xticks([-3, 0, 3])
        plt.yticks([-3, 0, 3])
        plt.xlabel('$x/w_0$', labelpad=-1)
        plt.ylabel('$y/w_0$', labelpad=-13)
    plt.text(-3.6, 3.1, '$z=0$', color='w')
    cbar1 = plt.colorbar(fraction=0.06, aspect=15, pad=0.02, ticks=[0, 1])
    cbar1.ax.tick_params(labelsize=cbar_size)
    plt.title(titles[0])
    if not axes:
        plt.tick_params(top=False, bottom=False, left=False, right=False,
                        labelleft=False, labelbottom=False)
    plt.subplot(1, 2, 2)
    plt.imshow(np.angle(field2D.T), cmap=cmapF, interpolation='nearest', vmin=-np.pi, vmax=np.pi,
               extent=[*x_lim, *x_lim])  # , cmap='twilight', interpolation='nearest'
    plt.xticks(())
    plt.yticks(())
    if show_axis:
        # plt.xticks(np.arange(-3, 5, 2))
        # plt.yticks(np.arange(-3, 5, 2))
        plt.xticks([-3, 0, 3])
        plt.yticks([-3, 0, 3])
        plt.xlabel('$x/w_0$', labelpad=-1)
        plt.ylabel('$y/w_0$', labelpad=-13)
    plt.text(-3.6, 3.1, '$z=0$')
    # cbar2 = plt.colorbar(fraction=0.05, pad=0.02, ticks=[-np.pi, 0, np.pi])
    cbar2 = plt.colorbar(fraction=0.06, aspect=15, pad=0.02, ticks=[-np.pi, 0, np.pi])
    cbar2.ax.tick_params(labelsize=cbar_size)
    cbar2.ax.set_yticklabels([f'-$\pi$', 0, f'$\pi$'])
    plt.title(titles[1])
    if not axes:
        plt.tick_params(top=False, bottom=False, left=False, right=False,
                        labelleft=False, labelbottom=False)
    plt.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.1)


trefoil_normal = 0
if trefoil_normal:
    file_name = (f'2D_field_trefoil_rotated_{False}_shifted_{False}_'
                 f'xy_res_{res_x_3D}_xy_bord_{x_lim_3D}'
                 f'_w_{w}_w_real_{w_real}.npy')
    print(file_name)
    field = np.load(file_name)
    plot_field(field, titles=('', ''), cmapF=cmapF, cmapE=cmapE,
               axes=True, x_lim=np.array(x_lim_3D) / w_real,
               show_axis=True, fonts=fonts)
    plt.savefig('test.svg', transparent=True)
    plt.show()
    exit()

trefoil_normal_no_ticks = 0
if trefoil_normal_no_ticks:
    file_name = (f'2D_field_trefoil_rotated_{False}_shifted_{False}_'
                 f'xy_res_{res_x_3D}_xy_bord_{x_lim_3D}'
                 f'_w_{w}_w_real_{w_real}.npy')
    print(file_name)
    field = np.load(file_name)
    fonts2 = (font_numbers+4, font_label, font_text+4)
    plot_field(field, titles=('', ''), cmapF=cmapF, cmapE=cmapE,
               axes=True, x_lim=np.array(x_lim_3D) / w_real,
               show_axis=False, fonts=fonts2)
    plt.savefig('test.svg', transparent=True)
    plt.show()

trefoil_rotated_shifted_no_ticks = 1
if trefoil_rotated_shifted_no_ticks:
    file_name = (f'2D_field_trefoil_rotated_{True}_shifted_{True}_'
                 f'xy_res_{res_x_3D}_xy_bord_{x_lim_3D}'
                 f'_w_{w}_w_real_{w_real}.npy')
    print(file_name)
    field = np.load(file_name)
    fonts2 = (font_numbers+4, font_label, font_text+4)
    plot_field(field, titles=('', ''), cmapF=cmapF, cmapE=cmapE,
               axes=True, x_lim=np.array(x_lim_3D) / w_real,
               show_axis=False, fonts=fonts2)
    plt.savefig('test.svg', transparent=True)
    plt.show()

trefoil_rotated_shifted_milnor_no_ticks = 0
if trefoil_rotated_shifted_milnor_no_ticks:
    file_name = (f'2D_field_trefoil_milnor_rotated_{True}_shifted_{True}_'
                 f'xy_res_{res_x_3D}_xy_bord_{tuple(np.array(x_lim_3D) / w_real * w)}'
                 f'_w_{w}_w_real_{w_real}.npy')
    print(file_name)
    field = np.load(file_name)
    fonts2 = (font_numbers+4, font_label, font_text+4)
    plot_field(field, titles=('', ''), cmapF=cmapF, cmapE=cmapE,
               axes=True, x_lim=np.array(x_lim_3D) / w_real,
               show_axis=False, fonts=fonts2)
    plt.savefig('test.svg', transparent=True)
    plt.show()