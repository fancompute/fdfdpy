import numpy as np
import matplotlib.pyplot as plt

def plt_base(field_val, outline_val, cmap, vmin, vmax, label, cbar=True, outline=None, ax=None):
    # Base plotting function for fields

    if ax is None:
        fig, ax = plt.subplots(1, constrained_layout=True)

    h = ax.imshow(field_val.transpose(), cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')

    if cbar:
        plt.colorbar(h, label=label, ax=ax)

    if outline:
        # Do black and white so we can see on both magma and RdBu
        ax.contour(outline_val.transpose(), levels=2, linewidths=1.0, colors='w')
        ax.contour(outline_val.transpose(), levels=2, linewidths=0.5, colors='k')

    ax.set_xticks([])
    ax.set_yticks([])

    return ax

def plt_base_eps(field_val, outline_val, cmap, vmin, vmax, cbar=True, outline=None, ax=None):
    # Base plotting function for permittivity

    if ax is None:
        fig, ax = plt.subplots(1, constrained_layout=True)

    h = ax.imshow(field_val.transpose(), cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')

    if cbar:
        plt.colorbar(h, label='relative permittivity', ax=ax)

    if outline:
        # Do black and white so we can see on both magma and RdBu
        ax.contour(outline_val.transpose(), levels=2, linewidths=1.0, colors='w')
        ax.contour(outline_val.transpose(), levels=2, linewidths=0.5, colors='k')

    ax.set_xticks([])
    ax.set_yticks([])

    return ax
