from hardware.utils.circs import depth2qb
import matplotlib.pyplot as plt
import numpy as np
import os

def luminance(*rgb):
    """Calcula la luminancia relativa de un color."""
    rgb = list(rgb)
    for i in range(3):
        if rgb[i] <= 0.03928:
            rgb[i] = rgb[i] / 12.92
        else:
            rgb[i] = ((rgb[i] + 0.055) / 1.055) ** 2.4
    return rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722

def contrast_ratio(l1, l2):
    """Calcula la relación de contraste entre dos luminancias."""
    if l1 > l2:
        return (l1 + 0.05) / (l2 + 0.05)
    else:
        return (l2 + 0.05) / (l1 + 0.05)

def best_text_color(*rgb):
    """Determina si el texto blanco o negro se verá mejor sobre un color dado."""
    lum = luminance(*rgb)
    contrast_with_white = contrast_ratio(lum, luminance(1, 1, 1))
    contrast_with_black = contrast_ratio(lum, luminance(0, 0, 0))
    
    if contrast_with_white > contrast_with_black:
        return "white"
    else:
        return "black"
    
def save_site_gauge_observable_matrix(site_gauge_observable_matrix, filepath, header=""):
    if os.path.splitext(filepath)[1] == ".npy":
        np.save(filepath, site_gauge_observable_matrix)
    else:
        np.savetxt(filepath, site_gauge_observable_matrix, header=header)

def load_site_gauge_observable_matrix(filepath):
    if os.path.splitext(filepath)[1] == ".npy":
        return np.load(filepath)
    else:
        return np.loadtxt(filepath)
    
def plot_circuit_arr_depth(circuit_arr, x_ticks=None, x_label=None, filepath=""):
    plt.rc("text", usetex=True)
    plt.rc("font", size=20, family="serif", weight="bold")

    fig, ax = plt.subplots(figsize=[9, 6])
    
    x = np.arange(1, len(circuit_arr)+1)
    y = [depth2qb(circ) for circ in circuit_arr]
    color = plt.get_cmap("Set2")((1 % 8)/8 + 0.01)
    plt.plot(x, y, "o--", markersize=12, markeredgecolor="black", color=color)
    if x_label is None:
        plt.xlabel("Circuit index", labelpad=10)
    else:
        plt.xlabel(x_label, labelpad=10)
    if x_ticks is not None:
        plt.xticks(x_ticks)
    plt.ylabel("Depth")
    plt.grid(color="gray", linestyle="dashdot", linewidth=1.6, zorder=0)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=300, facecolor="none")
    plt.show()
    plt.rcdefaults()

def plot_many_lines(x, ys, x_label="", x_ticks=None, y_label="", y_ticks=None, labels=None, filepath=""):
    x = np.array(x)
    ys = np.array(ys)
    if x.ndim != 1:
        raise ValueError("x must be a 1D array-like")
    if x.shape[0] != ys.shape[1]:
        raise ValueError("x and ys must have compatible shapes")
    if ys.ndim > 2:
        raise ValueError("ys must be a 1D or 2D array-like")
    
    plt.rc("text", usetex=True)
    plt.rc("font", size=18, family="serif", weight="bold")

    fig, ax = plt.subplots(figsize=[9, 6])
    
    nlines = len(ys)
    for i, y in enumerate(ys):
        color = plt.get_cmap("Set2")((i % 8)/8)
        plt.plot(x, y, "o--", markersize=12, linewidth=2, markeredgecolor="black", color=color, label=labels[i] if labels is not None else None)
    plt.xlabel(x_label, labelpad=10)
    plt.ylabel(y_label)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    if labels is not None:
        plt.legend()
    plt.grid(color="gray", linestyle="dashdot", linewidth=1.6, zorder=0)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=300, facecolor="none")
    plt.show()
    plt.rcdefaults()

def matrix(matrix_to_plot, cmap="inferno", xlabel="Row", ylabel="Column", xticks=None, yticks=None, cbar_range=None, clabel=r"$\left|\left\langle u(t_a) \right| \left. u(t_b) \right\rangle\right|^2$", filepath=""):
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", size=18, weight="bold")
    xs = np.arange(matrix_to_plot.shape[1])
    ys = np.arange(matrix_to_plot.shape[0])
    X, Y = np.meshgrid(xs, ys)
    if cbar_range is None:
        vmin, vmax = None, None
    else:
        vmin, vmax = cbar_range
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.pcolormesh(X, Y, matrix_to_plot, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.gca().invert_yaxis()
    plt.colorbar(label=clabel)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=300, facecolor="none")
    plt.show()
    plt.rcdefaults()

def samples_histogram(samples, xlabel="Basis state", ylabel="Counts", filepath=""):
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", size=18)
    
    plt.plot([int(k) for k in samples.keys()], list(samples.values()), "o", markersize=8, markeredgecolor="black", color=plt.get_cmap("Set2")(0))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.rcdefaults()
    if filepath:
        plt.savefig(filepath, dpi=300, facecolor="none")
    plt.show()