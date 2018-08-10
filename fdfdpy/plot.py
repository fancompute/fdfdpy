from numpy import pi, real, zeros, abs, exp
from matplotlib.pyplot import subplots, colorbar, close
from matplotlib import animation

def plt_base(field_val, outline_val, cmap, vmin, vmax, label, cbar=True, outline=None, ax=None):
    # Base plotting function for fields

    if ax is None:
        fig, ax = subplots(1, constrained_layout=True)

    h = ax.imshow(field_val.transpose(), cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')

    if cbar:
        colorbar(h, label=label, ax=ax)

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
        fig, ax = subplots(1, constrained_layout=True)

    h = ax.imshow(field_val.transpose(), cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')

    if cbar:
        colorbar(h, label='relative permittivity', ax=ax)

    if outline:
        # Do black and white so we can see on both magma and RdBu
        ax.contour(outline_val.transpose(), levels=2, linewidths=1.0, colors='w')
        ax.contour(outline_val.transpose(), levels=2, linewidths=0.5, colors='k')

    ax.set_xticks([])
    ax.set_yticks([])

    return ax


def plt_base_ani(field_val, cbar=True, Nframes=40, interval=80):

    fig, ax = subplots(1, constrained_layout=True)
    h = ax.imshow(zeros(field_val.shape).transpose(), origin='lower')

    ax.set_xticks([])
    ax.set_yticks([])

    def init():
        vmax=abs(field_val).max()
        h.set_data(zeros(field_val.shape).transpose())
        h.set_cmap('RdBu')
        h.set_clim(vmin=-vmax, vmax=+vmax)

        return (h,)

    def animate(i):
        h.set_data(real(field_val*exp(1j*2*pi*i/(Nframes-1))).transpose())
        return (h,)

    close()
    return animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=Nframes, interval=interval, blit=True)
