# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "anthropic==0.52.0",
#     "marimo",
#     "matplotlib==3.10.3",
#     "numpy==2.2.6",
#     "scipy==1.15.3",
#     "sympy==1.14.0",
# ]
# ///

import marimo

__generated_with = "0.13.15"
app = marimo.App(
    width="medium",
    layout_file="layouts/2_multivariate_normal.slides.json",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Multivariate normal distributions

    &copy; 2025 by [ST John](https://github.com/st--)


    We want to put a distribution $p(\mathbf{f}) = p(f_1, f_2, ..., f_N)$ on all the function values corresponding to our $N$ observed data points. For that, we will use the simplest multivariate distribution... the multivariate Gaussian distribution. Let's get a bit more familiar with that!
    """
    )
    return


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
    _var = "f"
    _var = r"\star"
    # _var = "x"

    mo.md(
        r"""
        # First: Gaussian distribution

        $$ \mathrm{N}(VAR | \mu, \sigma^2) $$

        Fully defined by mean $\mu$ and standard deviation $\sigma$ / variance $\sigma^2$

        $$\mathbb{E}[VAR] = \mu \qquad \mathbb{V}[VAR] = \sigma^2$$
        """.replace("VAR", r"{\color{red} VAR}").replace("VAR", _var)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    # Create sliders for mean and standard deviation
    mean_slider = mo.ui.slider(-3.0, 3.0, value=0.0, step=0.1, label="Mean", show_value=True)
    std_slider = mo.ui.slider(0.1, 3.0, value=1.0, step=0.1, label="Standard Deviation", show_value=True)

    # Display the sliders
    ui_1d = mo.hstack([mean_slider, std_slider])
    return mean_slider, std_slider, ui_1d


@app.cell(hide_code=True)
def _(mo, np, plt, stats):
    @mo.cache
    def plot_normal_distribution(mean, std): 
        # Create x values for plotting
        x = np.linspace(-5, 5, 1000)

        # Calculate the PDF values
        pdf = stats.norm.pdf(x, loc=mean, scale=std)

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, pdf, 'b-', lw=2, label=f'Normal PDF (μ={mean:.1f}, σ={std:.1f})')
        ax.fill_between(x, pdf, alpha=0.2)

        # Add vertical line at mean
        ax.axvline(mean, color='r', linestyle='--', alpha=0.5, label='Mean')

        # Add vertical lines at mean ± std
        ax.axvline(mean - std, color='g', linestyle=':', alpha=0.5, label='Mean ± Std Dev')
        ax.axvline(mean + std, color='g', linestyle=':', alpha=0.5)

        # Add labels and title
        ax.set_xlim(x.min(), x.max())
        ax.set_xlabel('x')
        ax.set_ylabel('Probability Density')
        ax.set_title('Normal Distribution PDF')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return plt.gca()
    return (plot_normal_distribution,)


@app.cell(hide_code=True)
def _(mean_slider, mo, plot_normal_distribution, std_slider, ui_1d):
    mo.vstack([
        ui_1d,
        mo.hstack([
            plot_normal_distribution(
                mean_slider.value, std_slider.value    # Get current values from sliders
            ),
        ]),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Multivariate Gaussian distribution

    $$
    \mathrm{N}(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Sigma})
    $$

    where:

    - $\mathbf{x}$ is a $d$-dimensional random vector
    - $\mathbb{E}[\mathbf{x}] = \boldsymbol{\mu}$ is the $d$-dimensional mean vector
    - $\operatorname{cov}[\mathbf{x}] = \boldsymbol{\Sigma}$ is the $d \times d$ covariance matrix
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    a_slider = mo.ui.slider(0.1, 3, step=0.1, value=1, label="a", show_value=True)
    b_slider = mo.ui.slider(0.1, 3, step=0.1, value=1, label="b", show_value=True)

    rho_slider = mo.ui.slider(-0.9, 0.9, step=0.1, value=0, label=r"$\rho$", show_value=True)

    ui_cov2d_diag = mo.ui.array([a_slider, b_slider])
    ui_cov2d_offd = mo.ui.array([rho_slider])
    return ui_cov2d_diag, ui_cov2d_offd


@app.cell
def _(np, ui_cov2d_diag, ui_cov2d_offd):
    a, b = [ui.value for ui in ui_cov2d_diag]

    cov_matrix_diag = np.array([
            [a, 0],
            [0, b]
        ])


    rho, = [ui.value for ui in ui_cov2d_offd]

    cov_matrix_offd = np.array([
            [1, rho],
            [rho, 1]
        ])
    return cov_matrix_diag, cov_matrix_offd


@app.cell(hide_code=True)
def _(mo):
    cov2d_choices = ["diagonal", "off-diagonals"]

    ui_cov2d_chooser = mo.ui.dropdown(
        options=cov2d_choices,
        value=list(cov2d_choices)[0],
        label="# Covariance matrix:"
    )
    return (ui_cov2d_chooser,)


@app.cell(hide_code=True)
def _(ui_cov2d_chooser):
    cov2d_choice = ui_cov2d_chooser.value
    return (cov2d_choice,)


@app.cell(hide_code=True)
def _(cov2d_choice, ui_cov2d_diag, ui_cov2d_offd):
    if cov2d_choice == "diagonal":
        ui_cov2d = ui_cov2d_diag
    elif cov2d_choice == "off-diagonals":
        ui_cov2d = ui_cov2d_offd
    return (ui_cov2d,)


@app.cell(hide_code=True)
def _(cov2d_choice, cov_matrix_diag, cov_matrix_offd):
    if cov2d_choice == "diagonal":
        cov_matrix = cov_matrix_diag
        cov_ui_eqn = r"""$$\boldsymbol{\Sigma} = \begin{bmatrix} {\color{red}a} & 0 \\ 0 & {\color{red}b} \end{bmatrix}$$"""
        cov_elem_eqn = r"""$(\boldsymbol{\Sigma})_{ii} = \operatorname{cov}(x_i, x_i) = \mathbb{V}[x_i] = \mathbb{E}[x_i^2] - \mu_i^2$"""

    elif cov2d_choice == "off-diagonals":
        cov_matrix = cov_matrix_offd
        cov_ui_eqn = r"""$$\boldsymbol{\Sigma} = \begin{bmatrix} 1 & {\color{red}\rho} \\ {\color{red}\rho} & 1 \end{bmatrix}$$"""
        cov_elem_eqn = r"""$(\boldsymbol{\Sigma})_{ij} = \operatorname{cov}(x_i, x_j) = \mathbb{E}[x_i x_j] - \mu_i \mu_j$"""
    return cov_elem_eqn, cov_matrix, cov_ui_eqn


@app.function(hide_code=True)
def matrix_to_latex(mat):
    if len(mat.shape) == 1:
        # turn vector into 1-column matrix
        mat = mat[:, None]

    rows = [
        " & ".join(str(mat[i, j]) for j in range(mat.shape[1]))
        for i in range(mat.shape[0])
    ]
    return r"\begin{bmatrix} " + r" \\ ".join(rows) + r" \end{bmatrix}"


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's look at how the values of the covariance matrix affect the distribution:""")
    return


@app.cell(hide_code=True)
def _(
    cov_elem_eqn,
    cov_matrix,
    cov_ui_eqn,
    mo,
    np,
    plot_2d_gaussian,
    ui_cov2d,
    ui_cov2d_chooser,
):
    def _():
        # Format the covariance matrix for LaTeX
        mean_vector_latex = matrix_to_latex(np.array([0,0]))
        cov_matrix_latex = matrix_to_latex(cov_matrix)

        # Create a markdown object with the multivariate normal distribution formula
        multivariate_normal_latex = rf"""{cov_ui_eqn}

        {cov_elem_eqn}

        $\boldsymbol{{\mu}} = {mean_vector_latex}$
        $\boldsymbol{{\Sigma}} = {cov_matrix_latex}$"""
        print(cov_matrix)

        # Display the markdown
        return mo.md(multivariate_normal_latex)

    mo.vstack([
        ui_cov2d_chooser,
        mo.hstack([mo.hstack(ui_cov2d)]),
        _(),
        mo.hstack([ plot_2d_gaussian(cov_matrix) ]),
    ])
    return


@app.cell
def _(np, plt, stats):
    def stddevs_to_coverage_1d(n_std: float) -> float:
        """ Given number of standard deviations (±n), return coverage under 1D Gaussian, e.g. ±n -> 0.9545 """
        return 2 * stats.norm.cdf(n_std) - 1

    def coverage_to_level(mvn: stats.multivariate_normal, coverage: float) -> float:
        """ For a given mvn distribution, return the contour level containing the given coverage. """
        D2 = stats.chi2(df=mvn.dim).ppf(coverage)  # squared Mahalanobis radius corresponding to coverage
        return mvn.pdf(mvn.mean) * np.exp(-0.5 * D2)

    class Gaussian2DPdfPlotter:
        def __init__(self, mvn: stats.multivariate_normal, xmin=-5, xmax=5, step=0.01):
            self.mvn = mvn

            # Create a grid of points
            self.X, self.Y = np.mgrid[xmin:xmax:step, xmin:xmax:step]

            # Stack the meshgrid points into a 2D array
            pos = np.dstack((self.X, self.Y))

            # Calculate the PDF values
            self.pdf = mvn.pdf(pos)

        def contourf(self, ax, with_colorbar=None, **kwargs):
            contour = ax.contourf(self.X, self.Y, self.pdf, **kwargs)

            if with_colorbar is not None:  # Add a colorbar
                plt.colorbar(contour, ax=ax, **with_colorbar)

        def contour(self, ax, **kwargs):
            ax.contour(self.X, self.Y, self.pdf, **kwargs)
    return Gaussian2DPdfPlotter, coverage_to_level, stddevs_to_coverage_1d


@app.cell
def _(
    Gaussian2DPdfPlotter,
    coverage_to_level,
    mo,
    np,
    plt,
    stats,
    stddevs_to_coverage_1d,
):
    from matplotlib import cm

    @mo.cache
    def plot_2d_gaussian(cov_matrix, mean=None, stddevs=[1, 2, 3], title="2D Gaussian Distribution"):
        """
        Plot the 2D probability density function of a Gaussian distribution.

        Parameters:
        -----------
        cov_matrix : array-like, shape (2, 2)
            The covariance matrix of the Gaussian distribution.
        mean : array-like, shape (2,), optional
            The mean vector of the Gaussian distribution. Default is [0, 0].
        title : str, optional
            The title of the plot.

        Returns:
        --------
        ax : matplotlib.axes.Axes
            The axes object containing the plot.
        """
        if mean is None:
            mean = np.zeros(2)

        mvn = stats.multivariate_normal(mean, cov_matrix)

        pdf_plotter = Gaussian2DPdfPlotter(mvn)

        fig, ax = plt.subplots(figsize=(10, 8))
        pdf_plotter.contourf(ax, cmap='viridis', levels=50, with_colorbar=dict(label='Probability Density'))
        pdf_plotter.contour(ax, colors='white', alpha=0.3, levels=10, linestyles='dashed')

        levels = []
        for n_std in reversed(sorted(stddevs)):  # contour levels must be ascending, so higher stddevs = lower pdf contours need to come first
            coverage = stddevs_to_coverage_1d(n_std)
            contour_level = coverage_to_level(mvn, coverage)
            levels.append(contour_level)
            ax.plot([], [], 'r--', label=f'{coverage:.2%} (~±{n_std}σ)')

        pdf_plotter.contour(ax, levels=levels, linewidths=2, linestyles='dashed', colors='red')

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)

        # Add a point for the mean
        ax.scatter(mean[0], mean[1], color='red', marker='x', s=100, label='Mean')

        ax.legend()

        # Make the plot tight
        plt.tight_layout()

        return plt.gca()
    return (plot_2d_gaussian,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Marginalizing: Projection

    Given a joint distribution $p(\mathbf{x}) = p(x_1, x_2)$, a *marginal* distribution is the distribution of a subset of the random variables, which we get by integrating out the others:

    \begin{align}
    p(x_1) &= \int p(x_1, x_2) \mathrm{d}x_2 \\
    p(x_2) &= \int p(x_1, x_2) \mathrm{d}x_1
    \end{align}

    (i.e., "projection onto the margins")

    For a multivariate Gaussian distribution, all marginal distributions are also Gaussian.
    """
    )
    return


@app.cell
def _(Gaussian2DPdfPlotter, coverage_to_level, stddevs_to_coverage_1d):
    def plot_bivariate_gaussian(ax, xvec_dist, xvec_samples, xmin, xmax, *, alpha=0.5, size=1, levels_coverage=True):
        ax.set_xlim(xmin, xmax); ax.set_ylim(xmin, xmax); ax.grid()

        pdf_plotter = Gaussian2DPdfPlotter(xvec_dist, xmin=xmin, xmax=xmax)
        levels = (
            [coverage_to_level(xvec_dist, stddevs_to_coverage_1d(n)) for n in [3,2,1]]
            if levels_coverage else None
        )
        pdf_plotter.contour(ax, colors='black', levels=levels)

        ax.scatter(xvec_samples[:,0], xvec_samples[:,1], s=size, color='C0', alpha=alpha)

        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    return (plot_bivariate_gaussian,)


@app.cell
def _(np, plot_bivariate_gaussian, plt, stats):
    def _():
        xvec_dist = stats.multivariate_normal(
            mean=np.array([0, 2]),
            cov=np.array([[1, 1],
                         [1, 3]]))

        N = 1000
        xvec_samples = xvec_dist.rvs(N)  # draw samples

        x1_dist = stats.norm(loc=xvec_dist.mean[0], scale=xvec_dist.cov[0,0]**0.5)  # loc=mean, scale=std.dev.=sqrt(var)
        x2_dist = stats.norm(loc=xvec_dist.mean[1], scale=xvec_dist.cov[1,1]**0.5)

        xlims = xmin, xmax = (-6, 6)

        def plot_marginal_distribution(ax, x_dist, x_samples, xlabel="", orientation='vertical'):
            assert orientation in ('vertical', 'horizontal')
            if orientation == 'vertical':
                # ax.set_xlim(*xlims)
                ax.set_xlabel(xlabel)
                ax.set_ylabel("PDF")
            elif orientation == 'horizontal':
                # ax.set_ylim(*xlims)
                ax.set_ylabel(xlabel)
                ax.set_xlabel("PDF")

            # Create grid for evaluating the PDF
            x_values = np.linspace(*xlims, 200)    

            # Exact PDF:
            pdf_values = x_dist.pdf(x_values)

            if orientation == 'vertical':
                _x, _y = x_values, pdf_values
            elif orientation == 'horizontal':
                _x, _y = pdf_values, x_values

            ax.plot(_x, _y, 'k-', linewidth=2, label='exact PDF')

            # Empirical histogram:
            ax.hist(x_samples, bins=30, range=xlims, density=True,
                    orientation=orientation, alpha=0.6, color='C0', 
                    label='empirical distribution')

            ax.legend(loc='lower left')
            ax.grid()

        fig, axes = plt.subplots(2, 2, constrained_layout=True, sharex='col', sharey='row')
        plot_bivariate_gaussian(axes[0, 0], xvec_dist, xvec_samples, xmin, xmax)
        plot_marginal_distribution(axes[1, 0], x1_dist, xvec_samples[:, 0], "$x_1$")
        plot_marginal_distribution(axes[0, 1], x2_dist, xvec_samples[:, 1], "$x_2$", orientation='horizontal')
        axes[1, 1].axis('off')

        return fig

    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Marginalizing in higher dimensions

    Let's convince ourselves that this works the same in higher dimensions.
    """
    )
    return


@app.cell
def _(mo):
    button_redraw = mo.ui.button(label="Regenerate random MVN distribution")
    button_redraw
    return (button_redraw,)


@app.cell
def _(button_redraw, np, stats):
    # Dimension
    D = 7  # 🔧

    button_redraw  # to trigger re-running this cell when the button is pressed

    # Generate a random mean vector
    random_mean_vector = 3*np.random.randn(D)  # 🔧

    def random_covariance_from_wishart(D):
        """ Generate a random covariance matrix by sampling from a Wishart distribution """
        scale_matrix = np.eye(D)  # here: identity matrix  # 🔧
        # Higher df makes the matrix closer to the scale matrix
        df = D + 5  # Using D + some value to ensure positive definiteness  # 🔧
        assert df >= D, "df (degrees of freedom) must be >= D for a valid covariance matrix"
        return stats.wishart.rvs(df=df, scale=scale_matrix)

    random_covariance_matrix = random_covariance_from_wishart(D)

    ### alternative: compute arbitrary matrix, 'square' it, add small value on diagonal
    # 🔧
    # matrix_factor = np.random.randn(D, D)
    # random_covariance_matrix = (matrix_factor @ matrix_factor.T) + 0.1 * np.eye(D)

    assert (np.linalg.eigvals(random_covariance_matrix) > 0).all(), "covariance matrix must be positive-definite"
    return D, random_covariance_matrix, random_mean_vector


@app.cell
def _(mo, plt, random_covariance_matrix, random_mean_vector):
    _fig1 = plt.figure()
    plt.title("Randomly generated mean vector")
    plt.imshow(random_mean_vector[:, None])
    plt.colorbar()

    _fig2 = plt.figure()
    plt.title("Randomly generated covariance matrix")
    plt.imshow(random_covariance_matrix)
    plt.colorbar()

    mo.hstack([_fig1, _fig2])
    return


@app.cell
def _(D, np, plt, random_covariance_matrix, random_mean_vector, stats):
    i = 2  # 🔧
    assert 0 <= i < D

    random_mvn = stats.multivariate_normal(random_mean_vector, random_covariance_matrix)

    _X_samples = random_mvn.rvs(1000)

    plt.title("Marginal distribution")
    X_i = _X_samples[:, i]
    plt.hist(X_i, bins=30, density=True)

    plt.xlabel(f"$x_{i+1}$")

    _mean = random_mvn.mean[i]
    _stddev = random_mvn.cov[i,i]**0.5
    xi_marginal = stats.norm(loc=_mean, scale=_stddev)

    _x_grid = np.linspace(_mean - 3*_stddev, _mean + 3*_stddev, 100)

    plt.xlim(-10,10)
    plt.ylim(0, xi_marginal.pdf(_mean)*1.2)
    plt.plot(_x_grid, xi_marginal.pdf(_x_grid), 'k-')

    plt.gcf()
    return


@app.cell(hide_code=True)
def _(button_redraw):
    button_redraw
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Conditioning: Slice""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Conditioning (Slicing/Filtering)

    $$ p(A | B) = \frac{p(A \cap B)}{p(B)} $$

    The conditional distribution of a random variable, given a value of another random variable that is now held constant at that value, corresponds to "filtering" out only that part of the joint distribution. This is effectively Bayes's theorem.

    We can actually do this empirically, by taking samples from A and discarding those that are not in B!


    ### in Gaussian distribution:

    Here, conditional distributions are again Gaussian as well:

    $$
    p(x_1, x_2) = \mathrm{N}\left(\begin{bmatrix}x_1 \\ x_2\end{bmatrix}
        | \begin{bmatrix}\mu_1 \\ \mu_2\end{bmatrix}, \begin{bmatrix}\Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22}\end{bmatrix} \right)
    $$

    $$
    p(x_1 | x_2 = {\color{orange}\tilde{x}_2}) = \mathrm{N}\big( x_1
    \;|\; \overbrace{\Sigma_{12} \Sigma_{22}^{-1} ({\color{orange}\tilde{x}_2} - \mu_2) + \mu_1},
    \quad \overbrace{\Sigma_{11} - \Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21}} \big)
    $$
    """
    )
    return


@app.cell
def _(np, stats):
    xvec_dist = stats.multivariate_normal(
        mean=np.array([0, 2]),
        cov=np.array([[1, 0.8],
                     [0.8, 1]]))

    _num_samples = 1000
    xvec_samples = xvec_dist.rvs(_num_samples)  # draw samples
    return xvec_dist, xvec_samples


@app.cell
def _(np, stats, xvec_dist, xvec_samples):
    def x1_given_x2__samples(x2_condition, tolerance):
        """ empirical conditioning for arbitrary 2D distribution: """
        x1 = xvec_samples[:, 0]
        x2 = xvec_samples[:, 1]
        mask = np.abs(x2 - x2_condition) < tolerance  # because the random variables are continuous, we cannot use exact equality comparison - then there would be zero samples left!
        return x1[mask]

    def x1_given_x2__dist(x2_condition):
        """ exact conditioning for 2D Gaussian distribution: """
        mean1, mean2 = xvec_dist.mean
        [[Sigma11, Sigma12], [Sigma21, Sigma22]] = xvec_dist.cov

        x1_cond_mean = Sigma12 / Sigma22 * (x2_condition - mean2) + mean1
        x1_cond_var = Sigma11 - Sigma12 / Sigma22 * Sigma21

        return stats.norm(x1_cond_mean, x1_cond_var**0.5)
    return x1_given_x2__dist, x1_given_x2__samples


@app.cell(hide_code=True)
def _(mo):
    ui_x2_cond = mo.ui.slider(start=-5, stop=5, step=0.1, value=0, label="Conditioning on $x_2 =$ ", show_value=True)
    ui_x2_cond
    return (ui_x2_cond,)


@app.cell
def _(
    np,
    plot_bivariate_gaussian,
    plt,
    ui_x2_cond,
    x1_given_x2__dist,
    x1_given_x2__samples,
    xvec_dist,
    xvec_samples,
):
    _tolerance = 0.1
    _x2_condition = ui_x2_cond.value

    def _():
        x1_cond_samples = x1_given_x2__samples(_x2_condition, tolerance=_tolerance)
        x1_cond_dist = x1_given_x2__dist(_x2_condition)

        xmin, xmax = (-6, 6)

        def _plot_condition_line(ax):
            ax.axhline(y=_x2_condition, color='r', linestyle='--', alpha=0.5)  # zero-line for histogram/pdf

        def _plot_condition_filter(ax):
            ymin = _x2_condition - _tolerance
            ymax = _x2_condition + _tolerance
            ax.fill_between([xmin, xmax], [ymin]*2, [ymax]*2, color='r', alpha=0.1)

        def _plot_x1_conditional_distribution_empirical(ax):
            ax.hist(x1_cond_samples, bottom=_x2_condition, bins=40, range=(xmin, xmax), density=True, color='r', alpha=0.3)

        def _plot_x1_conditional_distribution_exact(ax):
            _x_grid = np.linspace(xmin, xmax, 100)
            ax.plot(_x_grid, x1_cond_dist.pdf(_x_grid) + _x2_condition, 'r-')

        ax = plt.subplot()
        plot_bivariate_gaussian(ax, xvec_dist, xvec_samples, xmin, xmax, alpha=0.5)
        _plot_condition_line(ax)
        _plot_condition_filter(ax)
        _plot_x1_conditional_distribution_empirical(ax)
        # _plot_x1_conditional_distribution_exact(ax)  # 🔧

        return plt.gcf()

    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 🔧 Try it yourself

    - vary `_x2_condition` using the slider above
    - play with `_tolerance` (cell right above) vs `_num_samples` (cell further above)
    - compare against the exact conditional distribution that for a Gaussian we can compute in closed form
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Visualizing samples in higher dimensions

    So far, we looked at only bivariate (2D) Gaussian distributions, because then we can plot the full probability distribution. However, in the end we want to put a distribution over many points. We will no longer be able to plot the full $N$-dimensional pdf, but we can consider a different way of visualizing samples from high-dimensional distributions:
    """
    )
    return


@app.cell
def _(mo, np, plot_bivariate_gaussian, plt, stats):
    _cov_matrix = np.array(  # 🔧
        [[1, 0.8],
         [0.8, 1]]
    )

    # _cov_matrix = cov_matrix_offd  # 🔧

    _xv_dist = stats.multivariate_normal(
        mean=np.array([0, 0]),  # 🔧
        cov=_cov_matrix,
    )

    _points = np.array([
        (0, 0),
        # (0.1, 1.1),  # 🔧
        # (1, 1),      # 🔧
        # (-0.1, 0.4), # 🔧
    ])

    # _points = _xv_dist.rvs(10)  # 🔧

    def _plot_points_2d():
        ax = plt.subplot(1, 2, 1)
        plot_bivariate_gaussian(ax, _xv_dist, np.zeros((0,2)), -3, 3, levels_coverage=False)
        ax.plot(_points[None, :, 0], _points[None, :, 1], 'o')

        ax2 = plt.subplot(1, 2, 2, sharey=ax)
        indices = [1, 2]
        ax2.set_xlabel("index $i$")
        ax2.set_xticks(indices)
        ax2.plot(indices, _points.T, 'o-')
        ax2.set_ylabel("$x_i$")

        plt.tight_layout()
        return ax

    mo.vstack([
        mo.md("## 2D"),
        # mo.hstack(ui_cov2d_offd),  # 🔧
        mo.md(r"$$ \boldsymbol{\Sigma} = " + matrix_to_latex(_xv_dist.cov) + "$$"),
        mo.hstack([_plot_points_2d()]),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 🔧 Try it yourself

    - edit `_points` manually to grasp the change of visualization
    - change `_points` to be random draws from the distribution
    - change `_cov_matrix` (either manually, or by pointing it to `cov_matrix_offd` and also uncommenting the line with `ui_cov2d_offd` at the bottom)
    """
    )
    return


@app.cell
def _(mo):
    button_redraw_rvs = mo.ui.button(label="Redraw random samples from MVN distribution")
    return (button_redraw_rvs,)


@app.cell(hide_code=True)
def _(button_redraw_rvs, mo, np, plt, stats):
    def generate_covmatrix(D):
        mat = np.zeros((D, D))
        for i in range(D):
            for j in range(D):
                if i == j:
                    mat[i, j] = 1
                else:
                    mat[i, j] = 0.9 ** np.abs(i - j)  # 🔧
        return mat

    def generate_covmatrix_gaussian(D, ell=1):
        x = np.linspace(0, 5, D)
        return np.exp(- (x[:, None] - x[None, :])**2 / (2*ell**2))

    covariance_matrix = generate_covmatrix(10)  # 🔧

    covariance_matrix = generate_covmatrix_gaussian(10, ell=1)  # 🔧

    # Or use the `random_covariance_matrix` defined in section "Marginalizing in higher dimensions" 
    # covariance_matrix = random_covariance_matrix  # 🔧 (also uncomment the `button_redraw` at the bottom of this cell)

    _dim = len(covariance_matrix)

    ### ⚠️ Note that the covariance matrix must be *symmetric positive definite* (i.e., symmetric & all eigenvalues are positive). If this is not the case, you will get a `np.linalg.LinAlgError` when constructing the multivariate_normal distribution in the next line. Check by e.g. printing `np.linalg.eigvals(_cov_matrix)`. You can fix this by adding some small value to the diagonal:
    # covariance_matrix = covariance_matrix + 1e-6*np.eye(_dim)  # 🔧

    mvn_dist = stats.multivariate_normal(
        # mean=np.zeros(_dim),  # default
        # mean=np.ones(_dim),  # 🔧
        # mean=3*np.random.randn(_dim),  # 🔧
        #  ... or e.g. linear?
        # mean=np.arange(_dim),
        cov=covariance_matrix,
    )

    def _plot_cov():
        fig = plt.figure(); plt.title("covariance matrix")
        plt.imshow(mvn_dist.cov)
        return fig


    button_redraw_rvs  # to rerun the cell when corresponding button gets pressed
    _points = mvn_dist.rvs(10)

    def _rvs_viz(plot_mean=False):
        fig, ax = plt.subplots()
        indices = np.arange(mvn_dist.dim)
        ax.set_xlabel("index $i$")
        ax.set_xticks(indices)
        ax.plot(indices, _points.T, 'o-')
        if plot_mean:
            ax.plot(indices, mvn_dist.mean, 'k--')
        ax.set_ylabel("$x_i$")
        return ax

    mo.vstack([
        mo.md(f"## Higher dimensions: {_dim}D"),
        mo.vstack([
            mo.md(r"$$ \boldsymbol{\Sigma} = " + matrix_to_latex(np.round(mvn_dist.cov, 4)) + "$$"),
            mo.hstack([_plot_cov()]),
        ]),
        mo.hstack([
            _rvs_viz(),  # 🔧 pass  plot_mean=True
        ]),
        button_redraw_rvs,
        # button_redraw,  # 🔧 if you use `covariance_matrix = random_covariance_matrix`
    ])
    return generate_covmatrix_gaussian, mvn_dist


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 🔧 Try it yourself

    - change the parameters and/or the method that generate `covariance_matrix`
    - can you map the values of the covariance matrix to how it affects the distribution of points?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Conditioning in higher dimensions

        We can exactly condition a multivariate Gaussian distribution even in higher dimensions, the equation looks almost the same as in the bivariate case:

        $$
        p(x_1, x_2) = \mathrm{N}\left(\begin{bmatrix}x_1 \\ x_2\end{bmatrix}
            | \begin{bmatrix}\mu_1 \\ \mu_2\end{bmatrix}, \begin{bmatrix}\Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22}\end{bmatrix} \right)
        $$

        $$
        p(x_1 | x_2 = {\color{orange}\tilde{x}_2}) = \mathrm{N}\big( x_1
        \;|\; {\Sigma_{12} \Sigma_{22}^{-1} ({\color{orange}\tilde{x}_2} - \mu_2) + \mu_1},
        \quad {\Sigma_{11} - \Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21}} \big)
        $$
        """
        .replace(r"x_", r"\mathbf{x}_")
        .replace(r"\tilde{x}_", r"\tilde{\mathbf{x}}_")
        .replace(r"\mu_", r"\boldsymbol{\mu}_")
        .replace(r"\Sigma_{11}", r"{\color{red}\Sigma_{11}}")
        .replace(r"\Sigma_{12}", r"{\color{green}\Sigma_{12}}")
        .replace(r"\Sigma_{21}", r"{\color{green}\Sigma_{21}}")
        .replace(r"\Sigma_{22}", r"{\color{blue}\Sigma_{22}}")
        .replace(r"\Sigma_", r"\boldsymbol{\Sigma}_")
    )
    return


@app.cell
def _(np, stats):
    def conditional_multivariate_normal(mean, cov, cond_index, cond_values):
        """
        Compute the conditional distribution of a multivariate normal distribution.
        """
        # Convert inputs to numpy arrays
        mean = np.asarray(mean)
        cov = np.asarray(cov)
        cond_index = np.asarray(cond_index)
        cond_values = np.asarray(cond_values)

        # Get the indices of the variables not being conditioned on
        n = len(mean)
        uncond_index = np.array([i for i in range(n) if i not in cond_index])

        # Partition the mean vector
        mean_1 = mean[uncond_index]
        mean_2 = mean[cond_index]

        # Partition the covariance matrix
        cov_11 = cov[np.ix_(uncond_index, uncond_index)]
        cov_12 = cov[np.ix_(uncond_index, cond_index)]
        cov_21 = cov[np.ix_(cond_index, uncond_index)]
        cov_22 = cov[np.ix_(cond_index, cond_index)]

        # Compute the conditional mean and covariance
        cond_mean = mean_1 + cov_12 @ np.linalg.solve(cov_22, cond_values - mean_2)
        cond_cov = cov_11 - cov_12 @ np.linalg.solve(cov_22, cov_21)

        return cond_mean, cond_cov

    def sample_conditional_multivariate_normal(mvn, cond_index, cond_values, n_samples=1):
        """
        Generate samples from a conditional multivariate normal distribution.

        Parameters:
        -----------
        mean : array-like
            Mean vector of the original multivariate normal distribution
        cov : array-like
            Covariance matrix of the original multivariate normal distribution
        cond_index : array-like
            Indices of the variables being conditioned on
        cond_values : array-like
            Values of the variables being conditioned on
        n_samples : int, optional
            Number of samples to generate (default: 1)

        Returns:
        --------
        samples : ndarray
            Samples from the full distribution with the conditional values fixed
        """
        # Get conditional distribution parameters
        cond_mean, cond_cov = conditional_multivariate_normal(mvn.mean, mvn.cov, cond_index, cond_values)

        # Get the indices of the variables not being conditioned on
        n = mvn.dim
        uncond_index = np.array([i for i in range(n) if i not in cond_index])

        # Sample from the conditional distribution
        uncond_samples = stats.multivariate_normal.rvs(cond_mean, cond_cov, size=n_samples)

        # Ensure uncond_samples is 2D even if n_samples=1
        if n_samples == 1:
            uncond_samples = uncond_samples.reshape(1, -1)

        # Create full samples
        full_samples = np.zeros((n_samples, n))

        # Fill in the conditioned values
        for i, idx in enumerate(cond_index):
            full_samples[:, idx] = cond_values[i]

        # Fill in the unconditioned values
        for i, idx in enumerate(uncond_index):
            full_samples[:, idx] = uncond_samples[:, i]

        return full_samples
    return (sample_conditional_multivariate_normal,)


@app.cell
def _(np):
    def mv_condition_samples(xsamples, cond_index, cond_values, tolerance):
        """ empirical conditioning for arbitrary multivariate distribution: """
        mask = np.ones(xsamples.shape[0], bool)
        for i, x_condition in zip(cond_index, cond_values):
            submask = np.abs(xsamples[:, i] - x_condition) < tolerance
            mask = mask & submask
        return xsamples[mask]
    return (mv_condition_samples,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Let's look at conditioning in higher dimensions.

    - You can uncomment the line that plots the empirical mean and confidence band
    - Change the points on which you condition
    - What is the effect of changing mean and/or covariance matrix (cf previous section)?

    As PART 2, you can use a much higher-dimensional distribution - you might want to comment out the empirical conditioning which will work less and less well the more points you condition on.
    """
    )
    return


@app.cell
def _(
    mv_condition_samples,
    mvn_dist,
    np,
    plt,
    sample_conditional_multivariate_normal,
    stats,
):
    _prior = mvn_dist

    # # 🔧 *PART 2*
    # _dim = 100
    # _prior = stats.multivariate_normal(
    #     # mean=np.zeros(_dim),  # default
    #     # mean=np.ones(_dim),
    #     #  ... or e.g. linear?
    #     # mean=np.linspace(2, -5, _dim),
    #     cov=generate_covmatrix_gaussian(_dim, ell=1) + 1e-6*np.eye(_dim)
    # )

    _conditioning_points = [
        # (index, value) pairs: index must be an integer, value can be a float or integer

        (2, 0.0),
        # (3, 0.5),
        # (9, -1),

        # ### 🔧 for *PART 2*
        # (5, 1),
        # (30, 2),
        # (50, 1.5),
        # (70, -1),
    ]

    if len(_conditioning_points) > 0:
        # turn list-of-tuples into two separate arrays:
        _cond_index, _cond_values = map(np.asarray, zip(*_conditioning_points))
    else:
        # zip() does not work with empty lists, so we manually create empty arrays of the right type:
        _cond_index = np.empty((0,), int)
        _cond_values = np.empty((0,), float)

    _num_samples = 300  # 🔧
    _tolerance = 0.1  # 🔧

    _samples_prior = _prior.rvs(_num_samples)
    _samples_post_empirical = mv_condition_samples(_samples_prior, _cond_index, _cond_values, _tolerance)
    _samples_post_analytical = sample_conditional_multivariate_normal(_prior, _cond_index, _cond_values, 100)

    def _rvs_viz(samples):
        fig, ax = plt.subplots()

        indices = np.arange(samples.shape[1])

        def plot_samples(ax):
            ax.plot(indices, samples.T, 'C0-', alpha=0.3)

        def plot_conditioning_points(ax):
            ax.plot(_cond_index, _cond_values, 'ko')

        def plot_confidence_empirical(ax, n_std=2):
            """ plot mean and quantiles corresponding to +/- n std.dev. """
            mean_minus_n_std_quantile = stats.norm.cdf(-n_std)
            mean_quantile = 0.5
            mean_plus_n_std_quantile = stats.norm.cdf(n_std)

            # assuming the samples are actually from a Gaussian distribution, we can get lower-variance estimators through empirical mean and standard deviation:
            qmean = np.mean(samples, axis=0)
            qlo = qmean - n_std * np.std(samples, axis=0)
            qhi = qmean + n_std * np.std(samples, axis=0)

            # ### alternatively, the following works for _any_ distribution of samples:
            # qlo, qmean, qhi = np.quantile(samples, [mean_minus_n_std_quantile, mean_quantile, mean_plus_n_std_quantile], axis=0)

            ax.fill_between(indices, qlo, qhi, color='k', alpha=0.1)
            ax.plot(indices, qmean, 'k-')
            ax.plot(indices, qlo, 'k--')
            ax.plot(indices, qhi, 'k--')

        plot_samples(ax)
        plot_confidence_empirical(ax)  # 🔧 
        plot_conditioning_points(ax)

        if len(indices) < 20:
            ax.set_xticks(indices)
        ax.set_xlabel("index $i$")
        ax.set_ylabel("$x_i$")
        return ax

    [
        _rvs_viz(_samples_prior),
        _rvs_viz(_samples_post_empirical),  # 🔧 comment out for PART 2
        _rvs_viz(_samples_post_analytical),
    ]
    return


@app.cell
def _(generate_covmatrix_gaussian, np, stats):
    _dim = 100
    dist100 = stats.multivariate_normal(
        # mean=np.zeros(_dim),  # default
        # mean=np.ones(_dim),  # 🔧
        # mean=3*np.random.randn(_dim),  # 🔧
        #  ... or e.g. linear?
        # mean=np.arange(_dim),
        cov=generate_covmatrix_gaussian(_dim, ell=1) + 1e-6*np.eye(_dim)
    )
    return


if __name__ == "__main__":
    app.run()
