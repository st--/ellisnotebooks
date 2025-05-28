# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.3",
#     "numpy==2.2.6",
#     "scipy==1.15.3",
# ]
# ///

import marimo

__generated_with = "0.12.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    return np, plt, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Covariance **matrix** vs covariance **function**""")
    return


@app.cell
def _(np):
    import itertools


    def gaussian(x, x2=None, ell=0.5):
        if x2 is None: x2 = x
        return np.exp(- (x[:, None] - x2[None, :])**2 / (2*ell**2))
    return gaussian, itertools


@app.cell
def _(np):
    x_fine = np.linspace(0, 5, 201)

    idx_base = np.array([10, 40, 50, 70, 130, 150])

    perm1 = np.arange(len(idx_base))
    perm2 = [0, 5, 1, 4, 3, 2]
    perm = perm2

    idx = idx_base[perm]
    xs = x_fine[idx]
    return idx, idx_base, perm, perm1, perm2, x_fine, xs


@app.cell
def _(gaussian, idx, np, x_fine, xs):
    num_points = len(x_fine)
    num_samples = 3
    K0 = gaussian(x_fine)
    L0 = np.linalg.cholesky(K0 + 1e-6*np.eye(num_points))
    np.random.seed(128)
    #np.random.seed(131)
    f0 = L0 @ np.random.randn(num_points, num_samples)

    Kxx = gaussian(xs)
    Lx = np.linalg.cholesky(Kxx)
    f = Lx @ np.random.randn(len(idx), 10)
    return K0, Kxx, L0, Lx, f, f0, num_points, num_samples


@app.cell
def _():
    import enum

    class WhatTicks(enum.Enum):
        range = enum.auto()
        index = enum.auto()
        variable = enum.auto()
        value = enum.auto()
    return WhatTicks, enum


@app.cell
def _(WhatTicks, mo):
    what_ticks_ui = mo.ui.dropdown({c.name: c for c in [WhatTicks(i+1) for i in range(len(WhatTicks))]}, value=WhatTicks(1).name)
    return (what_ticks_ui,)


@app.cell
def _(Kxx, WhatTicks, draw_prep, idx, mo, np, perm, plt, what_ticks_ui, xs):
    def draw1():
        what_ticks = what_ticks_ui.value

        fig, ax, set_ticks = draw_prep()

        ax.imshow(Kxx)

        kwargs = {}
        if what_ticks == WhatTicks.range:
            ticks = np.arange(len(idx)) + 1
            xlabel, ylabel = "$i$", "$j$"
        elif what_ticks == WhatTicks.index:
            ticks = np.array(perm) + 1
            xlabel, ylabel = "$i$", "$j$"
        elif what_ticks == WhatTicks.variable:
            ticks = [f"$x_{i+1}$" for i in perm]
            xlabel = ylabel = ""
            kwargs = dict(color='r')
        elif what_ticks == WhatTicks.value:
            ticks = xs
            xlabel, ylabel = "$x$", "$x'$"
            plt.xticks(rotation=90)

        set_ticks(np.arange(len(idx)), ticks, **kwargs)
        ax.set_xlabel(xlabel, color='r', fontsize=20)
        ax.set_ylabel(ylabel, color='r', fontsize=20)

        plt.tight_layout()
        return fig

    mo.vstack([what_ticks_ui, mo.hstack([draw1()])])
    return (draw1,)


@app.cell
def _(K0, draw_prep, idx, np, plt, x_fine):
    def draw2(case):
        draw_lines = draw_patches = False
        if case == 4:
            draw_ticks_var = False
            draw_lines = False
            draw_patches = False
        elif case == 5:
            draw_ticks_var = False
            draw_lines = True
            draw_patches = False
        elif case == 6:
            draw_ticks_var = True
            draw_lines = True
            draw_patches = False
        elif case == 7:
            draw_ticks_var = True
            draw_lines = True
            draw_patches = True

        fig, ax, set_ticks = draw_prep()

        ax.set_xlim(0, len(K0))
        ax.set_ylim(len(K0), 0)

        ax.imshow(K0)
        set_ticks(np.arange(len(K0))[::20], x_fine[::20])
        plt.xticks(rotation=90)
        xlabel, ylabel = "$x$", "$x'$"
        ax.set_xlabel(xlabel, color='r', fontsize=20)
        ax.set_ylabel(ylabel, color='r', fontsize=20)

        def _draw_ticks_var():
            set_ticks(idx, [f"$x_{i+1}$" for i in range(len(idx))], color='r')

        def _draw_lines():
            for i in idx:
                ax.hlines(i, 0, len(K0), color='r')
                ax.vlines(i, 0, len(K0), color='r')

        def _draw_patches():
            for i in idx:
                for j in idx:
                    c = plt.cm.viridis(K0[i,j])
                    r = plt.Rectangle((i-5, j-5), 10, 10, facecolor=c, edgecolor='w', lw=0.5, zorder=3)
                    ax.add_patch(r)

        return fig

    [draw2(case) for case in range(4, 8)]
    return (draw2,)


@app.cell
def _(f0, idx, np, perm, plt, x_fine, xs):
    def draw_prep():
        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(111)

        plt.tick_params(axis='both', which='major', labelsize=20)
        fig.subplots_adjust(bottom=0.2) # or whatever

        def set_ticks(values, labels, **kwargs):
            ax.set_xticks(values, labels, **kwargs)
            ax.set_yticks(values, labels, **kwargs)

        return fig, ax, set_ticks

    def draw3(case):
        fig, ax, set_ticks = draw_prep()
        is_perm = np.array_equal(perm, np.arange(len(perm)))

        ax.set_ylim(-2.5, 2.5)
        fig.subplots_adjust(left=0.2) # or whatever

        if case == 8:
            ticks = np.array(perm) + 1
            ax.set_xticks(np.arange(len(idx)), ticks)
            ax.plot(f0[idx], 'o--' if is_perm else 'o-')
            xlabel = "$i$"
            ylabel = "$f_i$"

        if case >= 9:
            plt.xticks(rotation=90)
            ax.set_xlim(x_fine.min(), x_fine.max())
            ax.set_xticks(np.linspace(0, 5, 11))
            xlabel = "$x$"
            ylabel = "$f(x)$"
            ax.plot(x_fine, f0)

        if case >= 10:
            for i in range(f0.shape[1]):
                ax.plot(xs, f0[idx, i], 'o--' if is_perm else 'o-', color=f"C{i}", alpha=0.5)

        ax.set_xlabel(xlabel, color='r', fontsize=20)
        ax.set_ylabel(ylabel, color='r', fontsize=20)

        return fig
    return draw3, draw_prep


@app.cell
def _(draw3):
    [draw3(case) for case in range(8, 11)]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
