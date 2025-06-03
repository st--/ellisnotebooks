# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "anthropic==0.52.1",
#     "marimo",
#     "matplotlib==3.10.3",
#     "numpy==2.2.6",
#     "scipy==1.15.3",
#     "seaborn==0.13.2",
# ]
# ///

import marimo

__generated_with = "0.12.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Linear regression and Gaussian distributions""")
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    import seaborn as snb
    snb.set(font_scale=1.5)
    return np, plt, snb, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Gaussian linear regression""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Linear regression is perhaps the most frequently used technique in applied statistics for modelling the relationship between set of a covariates $\left\lbrace \mathbf{x}_n \right\rbrace_{n=1}^N$ and a response variable $\left\lbrace y_n \right\rbrace_{n=1}^N$. More formally, let $\mathbf{X} \in \mathbb{R}^{N \times D}$ be a design matrix and let  $\mathbf{y} \in \mathbb{R}^N$ be the response variables collected into a single vector, then the linear regression model is given by

        \begin{align}
        \mathbf{y} = \mathbf{X}\mathbf{w} + \mathbf{e},
        \end{align}

        where $\mathbf{w} \in \mathbb{R}^D$ is the regression weights and $\mathbf{e} \in \mathbb{R}^N$ is the observation noise vector.

        Assuming isotropic Gaussian noise and imposing a multivariate Gaussian prior on $\mathbf{w} \sim \mathrm{N}\left(\mathbf{m}, \mathbf{S}\right)$ gives rise to the following joint distribution

        \begin{align}
        p(\mathbf{y}, \mathbf{w}) = p\left(\mathbf{y}|\mathbf{w}\right)p\left(\mathbf{w}\right) = \mathrm{N}\left(\mathbf{y}\big|\mathbf{Xw}, \sigma^2\mathbf{I}\right)\mathrm{N}\left(\mathbf{w}\big|\mathbf{m}, \mathbf{S}\right).
        \end{align}

        To start, we will use the following simple model as running example:

        \begin{align}
        y_n = ax_n + b +  e_n = \underbrace{\begin{bmatrix}x_n&1\end{bmatrix}}_{\text{design matrix }\mathbf{X}} \begin{bmatrix}a\\b\end{bmatrix} + e_n.
        \end{align}

        That is, the parameters are $\mathbf{w} = \left[a, b\right]$, where $a$ and $b$ are the slope and intercept of the line, respectively.
        """
    )
    return


@app.cell
def _(np):
    def design_matrix_linreg(x):
        X = np.column_stack((x, np.ones(len(x))))  # [x_n 1]
        return X
    return (design_matrix_linreg,)


@app.cell
def _(mo):
    mo.md(r"""### Data set""")
    return


@app.cell
def _(np, plt):
    # Generate a small synthetic toy data set

    true_a = 1.7
    true_b = 1.0
    true_sigma = 1

    np.random.seed(18)
    n_example = 20
    example_x = np.random.uniform(-1, 2, size=n_example)
    example_y = (true_a * example_x + true_b + true_sigma * np.random.randn(*example_x.shape))

    def _():
        fig = plt.figure()
        plt.xlim(-3,3)
        plt.ylim(-7,7)
        plt.scatter(example_x, example_y, c='b')
        # plt.scatter(example_x[:2], example_y[:2], c='r')
        return fig

    _()
    return example_x, example_y, n_example, true_a, true_b, true_sigma


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Prior distribution

        Because $f(\mathbf{x}) = \mathbf{x}^T \mathbf{w}$, The distribution $p(\mathbf{w})$ over parameters implies a distribution $p(f)$ over functions:
        """
    )
    return


@app.cell
def _(np, stats, true_sigma):
    # we define the prior distribution p(w) = N(m, S):
    dim = 2
    prior_mean = np.zeros(dim)
    prior_scale = np.ones(dim)
    prior_cov = np.diag(prior_scale**2)
    prior_dist = stats.multivariate_normal(prior_mean, prior_cov)

    # the parameter of the likelihood:
    noise_var = true_sigma**2
    return dim, noise_var, prior_cov, prior_dist, prior_mean, prior_scale


@app.cell
def _(np):
    def compute_f_samples(X, param_samples):
        """ given w=param_samples, compute f = X w """
        return X @ param_samples.T

    def compute_f_marginals(X, param_dist):
        """ given p(w)=param_dist, compute marginal mean and variance of p(f|X) """
        mean_f = X @ param_dist.mean
        var_f = np.diag(X @ param_dist.cov @ X.T)
        return mean_f, var_f
    return compute_f_marginals, compute_f_samples


@app.cell
def _(np):
    param_lims = (-3, 3)

    def plot_contour(ax, func):
        """ visualize a 2D function by plotting contour lines of func() """
        a_grid = b_grid = np.linspace(*param_lims, 100)
        A, B = np.meshgrid(a_grid, b_grid, indexing='ij')
        AB = np.dstack([A, B])
        Z = func(AB)
        ax.contour(A, B, Z)
        ax.set_xlabel('slope')
        ax.set_ylabel('intercept')
    return param_lims, plot_contour


@app.cell
def _(
    compute_f_marginals,
    compute_f_samples,
    design_matrix_linreg,
    np,
    plot_contour,
    plt,
    prior_dist,
):
    def _():
        param_samples_prior = prior_dist.rvs(30)

        x_grid = np.linspace(-4, 4, 100)

        f_samples_prior = compute_f_samples(design_matrix_linreg(x_grid), param_samples_prior)

        fig, axes = plt.subplots(1, 2, figsize=(10,4))
        ax_param, ax_data = axes

        ax_data.set_title("Data space")
        ax_data.set_xlim(x_grid[0], x_grid[-1])
        ax_data.set_xticks([-4,-2,0,2,4])
        ax_data.set_xlabel("x")
        ax_data.set_ylabel("y")

        ax_data.plot(x_grid, f_samples_prior, 'b', alpha=0.5)

        c = 'k'
        mean_f, var_f = compute_f_marginals(design_matrix_linreg(x_grid), prior_dist)
        std_f = np.sqrt(var_f)
        ax_data.plot(x_grid, mean_f, c)
        ax_data.plot(x_grid, mean_f + 2 * std_f, c + '--')
        ax_data.plot(x_grid, mean_f - 2 * std_f, c + '--')

        ax_param.set_title("Parameter space")
        plot_contour(ax_param, prior_dist.pdf)
        ax_param.scatter(param_samples_prior[:,0], param_samples_prior[:,1], alpha=0.5)

        plt.tight_layout()
        return fig

    _()
    return


@app.cell
def _(mo):
    mo.md(r"""### Posterior distribution""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We want to compute the posterior

        $$
        p(\mathbf{w} | \mathbf{y}) = \frac{p(\mathbf{y}|\mathbf{w}) p(\mathbf{w})}{p(\mathbf{y})}
        $$

        The prior is
        $\mathrm{N}\left(\mathbf{w}\big|\mathbf{m}, \mathbf{S}\right)$.

        The likelihood is
        $\mathrm{N}\left(\mathbf{y}\big|\mathbf{Xw}, \sigma^2\mathbf{I}\right)$.

        With respect to the parameters $\mathbf{w}$, the marginal likelihood (aka 'evidence') $p(\mathbf{y})$ is a constant that normalizes the probability distribution.

        How can we find the posterior distribution?
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""First, let's define the data we want to use and the corresponding likelihood function:""")
    return


@app.cell
def _(design_matrix_linreg, example_x, example_y, ui_n_data):
    n_data = ui_n_data.value
    x = example_x[:n_data]
    y = example_y[:n_data]

    X = design_matrix_linreg(x)

    ui_n_data  # how many data points to use
    return X, n_data, x, y


@app.cell
def _(X, noise_var, np, stats, y):
    def compute_loglikelihood(params):
        """ given w=params, computes log p(y|w) using f=X w and p(y|f)=N(f,noise_var) """
        *param_broadcast_dim, param_dim = params.shape
        n_data_samples, data_dim = X.shape
        assert data_dim == param_dim

        f = params @ X.T

        assert f.shape == (*param_broadcast_dim, n_data_samples)
        assert y.shape == (n_data_samples,)

        # now we compute \sum_i log p(y_i | f_i, noise)
        return stats.norm(loc=f, scale=np.sqrt(noise_var)).logpdf(
                np.broadcast_to(y, f.shape)
            ).sum(axis=-1)

    def compute_likelihood(params):
        return np.exp(compute_loglikelihood(params))
    return compute_likelihood, compute_loglikelihood


@app.cell
def _(mo):
    mo.md(r"""As a very crude way of approximating the posterior, we can use simple rejection sampling:""")
    return


@app.cell
def _(
    compute_f_samples,
    compute_loglikelihood,
    design_matrix_linreg,
    np,
    plot_contour,
    plt,
    posterior_dist,
    prior_dist,
    x,
    y,
):
    _n_samples_prior = 1000
    param_samples_prior = prior_dist.rvs(_n_samples_prior)

    def filter_posterior(param_samples):
        # unnormalized log posterior = log prior + log likelihood
        logprior = prior_dist.logpdf(param_samples)
        loglik = compute_loglikelihood(param_samples)
        logpost_unnormalized = loglik + logprior
        p_post_unnormalized = np.exp(logpost_unnormalized)

        # we want to keep only those samples that correspond to the posterior
        # here we use rejection sampling:
        p_post_normalized = p_post_unnormalized / p_post_unnormalized.max()
        runif = np.random.uniform(low=0, high=1, size=len(param_samples))
        accepted_indices = runif < p_post_normalized  # an index is more likely to be kept if its posterior probability is high
        accepted_params = param_samples[accepted_indices]
        return accepted_params

    param_samples_posterior = filter_posterior(param_samples_prior)

    x_grid = np.linspace(-4, 4, 100)

    f_samples_prior = compute_f_samples(design_matrix_linreg(x_grid), param_samples_prior)
    f_samples_posterior = compute_f_samples(design_matrix_linreg(x_grid), param_samples_posterior)

    def _():
        fig, axes = plt.subplots(1, 3, figsize=(10,4))
        ax0, ax1, ax2 = axes

        ax0.set_title("Data")
        ax0.set_xlim(x_grid[0], x_grid[-1])
        ax0.set_xticks([-4,-2,0,2,4])
        ax0.set_xlabel("x")
        ax0.set_ylabel("y")

        ax0.plot(x, y, 'k.')
        ax0.plot(x_grid, f_samples_prior[:100], 'b', alpha=0.1)

        ax1.set_title("Prior")
        plot_contour(ax1, prior_dist.pdf)
        ax1.scatter(param_samples_prior[:,0], param_samples_prior[:,1], alpha=0.1)

        # Plot the accepted parameters corresponding to the posterior
        ax2.scatter(param_samples_posterior[:,0], param_samples_posterior[:,1], c='g', alpha=0.3)
        plot_contour(ax2, posterior_dist.pdf)  # true posterior distribution

        ax0.plot(x_grid, f_samples_posterior, 'g', alpha=0.3)

        plt.tight_layout()
        return fig

    _()
    return (
        f_samples_posterior,
        f_samples_prior,
        filter_posterior,
        param_samples_posterior,
        param_samples_prior,
        x_grid,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For a Gaussian prior and Gaussian likelihood, the posterior is also a Gaussian distribution.

        We want to find the parameters of the posterior $p(\mathbf{w}\big|\mathbf{y}) = \mathrm{N}\left(\mathbf{w}\big|\mu, \Sigma\right)$.

        This can be computed analytically:

        \begin{align}
        \Sigma &= \left(\frac{1}{\sigma^2}\mathbf{X}^T\mathbf{X} + \mathbf{S}^{-1}\right)^{-1}\\
        \mu &= \Sigma\left(\frac{1}{\sigma^2}\mathbf{X}^T\mathbf{y} + \mathbf{S}^{-1}\mathbf{m}\right)
        \end{align}
        """
    )
    return


@app.cell
def _(X, noise_var, np, prior_dist, stats, y):
    def compute_param_posterior(X, y, prior_dist=prior_dist, noise_var=noise_var):
        """ given p(w)=prior_dist, X, y, and p(e)=N(0,noise_var), compute the posterior distribution over parameters p(w|X,y) """
        m = prior_dist.mean
        S = prior_dist.cov
        assert len(X.shape)==2 and X.shape[1] == len(m)

        S_inv = np.linalg.inv(S)
        Sigma_inv = (X.T @ X) / noise_var + S_inv
        rhs = (X.T @ y) / noise_var + S_inv @ m
        mu = np.linalg.solve(Sigma_inv, rhs)
        Sigma = np.linalg.inv(Sigma_inv)
        return stats.multivariate_normal(mu, Sigma)

    posterior_dist = compute_param_posterior(X, y)
    return compute_param_posterior, posterior_dist


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's look at the Bayesian linear regression posterior in more detail:""")
    return


@app.cell(hide_code=True)
def _(mo):
    ui_n_data = mo.ui.slider(start=1, stop=20, label="number of data points", show_value=True)
    ui_show_leastsquares = mo.ui.checkbox(label="show least-squares fit")
    ui_show_prior = mo.ui.checkbox(label="show prior distribution")
    ui_show_priorsamples = mo.ui.checkbox(label="show prior samples")
    ui_show_posterior = mo.ui.checkbox(label="show posterior distribution")
    ui_show_posteriorsamples = mo.ui.checkbox(label="show posterior samples")
    ui_show_legend = mo.ui.checkbox(label="display legend")
    ui_n_samples = mo.ui.slider(start=1, stop=20, label="number of samples", show_value=True)
    return (
        ui_n_data,
        ui_n_samples,
        ui_show_leastsquares,
        ui_show_legend,
        ui_show_posterior,
        ui_show_posteriorsamples,
        ui_show_prior,
        ui_show_priorsamples,
    )


@app.cell(hide_code=True)
def _(
    mo,
    ui_n_data,
    ui_n_samples,
    ui_show_leastsquares,
    ui_show_legend,
    ui_show_posterior,
    ui_show_posteriorsamples,
    ui_show_prior,
    ui_show_priorsamples,
):
    mo.vstack([
        ui_n_data,
        ui_show_leastsquares,
        mo.hstack([ui_show_prior,    ui_show_posterior,]),
        mo.hstack([ui_show_priorsamples,    ui_show_posteriorsamples, ui_n_samples,]),
        ui_show_legend,
    ], align='start')
    return


@app.cell
def _(
    X,
    compute_f_marginals,
    compute_f_samples,
    compute_likelihood,
    design_matrix_linreg,
    np,
    param_lims,
    plot_contour,
    plt,
    posterior_dist,
    prior_dist,
    stats,
    ui_n_samples,
    ui_show_leastsquares,
    ui_show_legend,
    ui_show_posterior,
    ui_show_posteriorsamples,
    ui_show_prior,
    ui_show_priorsamples,
    x,
    y,
):
    def _():
        xlims = (-4, 4)
        x_grid = np.linspace(*xlims, 100)

        n_samples = ui_n_samples.value

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        ax_data, ax_prior, ax_lik, ax_post = axes.flatten()

        for ax in [ax_prior, ax_lik, ax_post]:
            ax.set_xlim(*param_lims)
            ax.set_xticks([-2,-1,0,1,2])
            ax.set_ylim(*param_lims)
            ax.set_yticks([-2,-1,0,1,2])

        def plot_data():
            ax_data.set_xlim(*xlims)
            ax_data.set_ylim(-7, 7)
            ax_data.plot(x, y, 'k.', markersize=12)

        def plot_leastsquares(c='k'):
            if len(x) <= 1: return  # cannot compute least squares for single data point

            lr = stats.linregress(x, y)
            f_pred = lr.slope * x_grid + lr.intercept

            ax_data.plot(x_grid, f_pred, c+'-', label='least-squares fit')
            ax_lik.plot([lr.slope], [lr.intercept], c+'*', ms=20, zorder=3,
                       label='least-squares fit')

        def plot_samples(ax_param, param_samples, c='b', label=None):
            f_samples = compute_f_samples(design_matrix_linreg(x_grid), param_samples)
            ax_data.plot(x_grid, f_samples, c+'-', alpha=0.25)
            ax_data.plot([], [], c+'-', label=label)
            ax_param.plot(param_samples[:, 0], param_samples[:, 1], c+'.', markersize=10)

        def plot_func_marginals(dist, c='b', label=None):
            mean_f, var_f = compute_f_marginals(design_matrix_linreg(x_grid), dist)
            std_f = np.sqrt(var_f)
            ax_data.plot(x_grid, mean_f, c, label=label)
            ax_data.plot(x_grid, mean_f + 2 * std_f, c + '--')
            ax_data.plot(x_grid, mean_f - 2 * std_f, c + '--')

        ax_data.set_title(f'N = {X.shape[0]}', fontweight='bold')
        plot_data()

        if ui_show_leastsquares.value:
            plot_leastsquares()


        if ui_show_prior.value or ui_show_priorsamples.value:
            ax_prior.set_title('Prior', fontweight='bold')
        else:
            ax_prior.axis('off')

        if ui_show_priorsamples.value:
            plot_samples(ax_prior, prior_dist.rvs(n_samples), 'b', label='prior samples')

        if ui_show_prior.value:
            plot_func_marginals(prior_dist, 'r', label='prior mean±2std')
            plot_contour(ax_prior, prior_dist.pdf)


        if ui_show_posterior.value or ui_show_posteriorsamples.value:
            ax_post.set_title('Posterior', fontweight='bold')
        else:
            ax_post.axis('off')

        if ui_show_posterior.value:
            plot_func_marginals(posterior_dist, 'g', label='posterior mean±2std')
            plot_contour(ax_post, posterior_dist.pdf)

        if ui_show_posteriorsamples.value:
            plot_samples(ax_post, posterior_dist.rvs(n_samples), 'm')


        ax_lik.set_title('Likelihood', fontweight='bold')
        plot_contour(ax_lik, compute_likelihood)


        if ui_show_legend.value:
            ax_data.legend(loc='best')
            ax_lik.legend(loc='lower left')

        plt.tight_layout()
        return fig

    _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Extending the features

        What if our data is nonlinear, as in the following?
        """
    )
    return


@app.cell
def _(nlX, nlY, plt):
    # nlX and nlY are defined further below

    def plot_nonlin_data():
        plt.plot(nlX, nlY, '.')

    plt.figure()
    plot_nonlin_data()
    plt.gcf()
    return (plot_nonlin_data,)


@app.cell
def _(mo):
    mo.md(
        r"""
        We can make our design matrix richer by defining more features!

        Let's revisit the 1D linear regression discussed so far:
        """
    )
    return


@app.cell
def _(np):
    xp = np.linspace(-3, 3, 50)[:, None]
    # xp_fine = np.linspace(xp.min(), xp.max(), 201)
    xp_fine = np.linspace(-4.1, 4.1, 201)
    return xp, xp_fine


@app.cell
def _(design_matrix_linreg, plt, xp):
    plt.figure()
    plt.title("Features of the simple 1D linear regression design matrix")
    plt.plot(xp, design_matrix_linreg(xp))
    plt.text(-1.8, -1.3, 'slope', rotation=30, color='C0')
    plt.text(-1.8, 1.2, 'intercept', color='C1')
    plt.xlim(-3, 3)
    plt.gcf()
    return


@app.cell
def _(compute_f_marginals, np, plt, stats, xp_fine):
    def plot_prior_samples(design_matrix_fn):
        X = design_matrix_fn(xp_fine)
        dim = X.shape[1]

        prior_dist = stats.multivariate_normal(np.zeros(dim), np.eye(dim))

        param_samples = prior_dist.rvs(30)

        fig = plt.figure()
        plt.xlim(xp_fine[0], xp_fine[-1])

        plt.plot(xp_fine, X @ param_samples.T, 'b-', alpha=0.25)

        mu_f, var_f = compute_f_marginals(X, prior_dist)
        std_f = np.sqrt(var_f)
        plt.plot(xp_fine, mu_f, 'r')
        plt.plot(xp_fine, mu_f + 2 * std_f, 'r--')
        plt.plot(xp_fine, mu_f - 2 * std_f, 'r--')

        return fig
    return (plot_prior_samples,)


@app.cell
def _(design_matrix_linreg, plot_prior_samples):
    plot_prior_samples(design_matrix_linreg)
    return


@app.cell
def _(compute_f_marginals, compute_param_posterior, np, plt, stats, xp_fine):
    def plot_regression(x, y, design_matrix_fn, noise_std=1e-3):
        X = design_matrix_fn(x)
        dim = X.shape[1]

        prior_dist = stats.multivariate_normal(np.zeros(dim), np.eye(dim))
        post_dist = compute_param_posterior(X, y, prior_dist, noise_var=noise_std**2)

        np.random.seed(21345)
        param_samples = post_dist.rvs(30)

        fig = plt.figure()
        plt.xlim(xp_fine[0], xp_fine[-1])

        Xgrid = design_matrix_fn(xp_fine)
        plt.plot(xp_fine, Xgrid @ param_samples.T, 'b-', alpha=0.25)

        mu_f, var_f = compute_f_marginals(Xgrid, post_dist)
        std_f = np.sqrt(var_f)
        plt.plot(xp_fine, mu_f, 'r')
        plt.plot(xp_fine, mu_f + 2 * std_f, 'r--')
        plt.plot(xp_fine, mu_f - 2 * std_f, 'r--')

        return fig
    return (plot_regression,)


@app.cell
def _(design_matrix_linreg, nlX, nlY, plot_nonlin_data, plot_regression, plt):
    plot_regression(nlX, nlY, design_matrix_linreg, noise_std=0.5)
    plot_nonlin_data()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Now consider polynomial regression""")
    return


@app.cell
def _(np, plt, xp_fine):
    def design_matrix_polyreg(x):
        X = np.column_stack([
            np.ones_like(x),
            x,
            x**2,
            # x**3,
            # x**4,
        ])
        return X

    plt.figure()
    plt.xlim(-3,3)
    plt.plot(xp_fine, design_matrix_polyreg(xp_fine));
    plt.gcf()
    return (design_matrix_polyreg,)


@app.cell
def _(design_matrix_polyreg, plot_prior_samples, plt):
    plot_prior_samples(design_matrix_polyreg)
    plt.ylim(-200,200)
    plt.gcf()
    return


@app.cell
def _(design_matrix_polyreg, nlX, nlY, plot_nonlin_data, plot_regression, plt):
    plot_regression(nlX, nlY, design_matrix_polyreg, noise_std=0.5)
    plot_nonlin_data()
    plt.gcf()
    return


@app.cell
def _(mo):
    mo.md(r"""You can see what happens if you add more columns corresponding to higher orders into `design_matrix_polyreg`.""")
    return


@app.cell
def _(mo):
    mo.md(r"""Alternatively, instead of "global" features such as $x$, $x^2$, etc., we can define *local* features as below:""")
    return


@app.cell
def _(np, plt, xp_fine):
    Nbasis = 5
    rbf_w = 0.5

    def gaussian(x, mu, ell):
        return np.exp(- (x - mu)**2 / (2*ell**2))

    def design_matrix_rbf(x):
        cols = []
        for c in np.linspace(-2, 2, Nbasis):
            cols.append(gaussian(x, c, rbf_w))
        X = np.column_stack(cols)
        return X

    plt.figure()
    plt.xlim(-3,3)
    plt.plot(xp_fine, design_matrix_rbf(xp_fine));
    plt.gcf()
    return Nbasis, design_matrix_rbf, gaussian, rbf_w


@app.cell
def _(design_matrix_rbf, plot_prior_samples):
    plot_prior_samples(design_matrix_rbf)
    return


@app.cell
def _(design_matrix_rbf, nlX, nlY, plot_nonlin_data, plot_regression):
    _fig = plot_regression(nlX, nlY, design_matrix_rbf, noise_std=0.5)
    plot_nonlin_data()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""We can make the design matrix richer by adding more basis functions, and extending how far they spread:""")
    return


@app.cell
def _(gaussian, np, plot_prior_samples, rbf_w):
    def make_design_matrix_with_n_rbf(n_basis, xmax=2):
        def _design_matrix_rbf(x):
            cols = []
            for c in np.linspace(-xmax, xmax, n_basis):
                cols.append(gaussian(x, c, rbf_w)) #/np.sqrt(n_basis))
            X = np.column_stack(cols)
            return X
        return _design_matrix_rbf

    plot_prior_samples(make_design_matrix_with_n_rbf(n_basis=20, xmax=3))
    return (make_design_matrix_with_n_rbf,)


@app.cell
def _(
    make_design_matrix_with_n_rbf,
    nlX,
    nlY,
    plot_nonlin_data,
    plot_regression,
):
    _fig = plot_regression(nlX, nlY, make_design_matrix_with_n_rbf(20, 3), noise_std=0.5)
    plot_nonlin_data()
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        What if we could have an **infinite** number of basis functions???

        We can do this by putting a distribution on the function values directly: $p(\mathbf{f})$ ...
        """
    )
    return


@app.cell
def _(np):
    nlX = np.array([
       5.7007757e+00, 1.3868311e+00, 3.6410555e+00, 2.9158948e+00, 5.3477938e+00, 4.5725810e+00, 2.7388060e+00, 1.1102186e-01,
       4.9284430e+00, 2.6682202e+00, 3.6925941e+00, 4.7516222e+00, 5.5308778e+00, 4.4292435e+00, 1.0575969e+00, 2.4342373e+00,
       5.6128182e+00, 5.5014266e+00, 2.4616212e+00, 5.3618972e+00, 3.4734783e-01, 2.1172088e+00, 4.8789990e+00, 5.9167804e-02,
       8.3334529e-01, 1.2165913e+00, 1.1923305e+00, 3.6227549e+00, 1.6331275e+00, 1.1928856e+00, 9.1643562e-02, 4.4807141e+00,
       2.6705786e+00, 5.5908875e+00, 2.7959661e+00, 2.5118968e+00, 5.0773285e+00, 3.1509150e+00, 1.2158841e+00, 4.0328248e+00,
       5.0287107e+00, 1.1783708e-01, 4.0876630e+00, 2.2768861e+00, 4.9907761e+00, 3.0168773e+00, 4.2568284e+00, 2.5733542e+00,
       1.8277042e+00, 1.1379225e+00, 1.1605869e+00, 4.0933393e+00, 1.8165864e+00, 3.2500431e+00, 9.0523786e-01, 4.1873909e+00,
       2.2702380e+00, 5.1600696e+00, 5.1219308e+00, 3.5613775e+00, 2.9793147e+00, 5.3986151e+00, 4.9297750e+00, 3.8694623e+00,
       4.9078460e+00, 3.9613653e+00, 2.0518237e+00, 1.7383554e+00, 2.0471614e+00, 3.2044741e+00, 4.3626793e+00, 1.8557410e+00,
       5.0309763e+00, 3.4084348e+00, 2.2224813e+00, 4.2164395e+00, 3.2794269e+00, 2.6692812e+00, 4.1674034e+00, 3.7278608e+00,
       4.7689265e+00, 5.7410607e+00, 3.1355421e+00, 5.2808532e+00, 1.0377368e+00, 5.8784814e+00, 1.6286836e+00, 1.5139761e+00,
       5.2544514e+00, 4.4238359e+00, 8.1911245e-01, 7.0540124e-02, 5.3633878e+00, 1.1948284e+00, 1.7923381e+00, 3.9686555e+00,
       1.7064515e+00, 2.8153457e+00, 3.8868674e-01, 5.9300096e+00, 3.4967501e+00, 2.5409775e+00, 3.0930705e+00, 2.0037089e+00,
       2.5974396e+00, 1.3556992e+00, 3.4788412e+00, 4.5621901e+00, 3.1789387e+00, 3.8431590e+00, 1.2544164e+00, 2.2789102e+00,
       4.6999719e+00, 4.0850745e+00, 2.7665708e+00, 3.4069723e+00, 4.7652639e+00, 3.5509556e-01, 3.6172145e+00, 3.0161282e-01,
       2.4922492e+00, 1.8299921e+00, 5.2462030e+00, 9.0056992e-02, 4.6077023e+00, 5.8250696e+00, 5.9404956e+00, 4.7331702e+00,
       2.6319512e+00, 2.9898678e+00, 1.2837800e+00, 3.8609537e+00, 1.9202135e+00, 5.7605916e+00, 4.3597906e+00, 2.4717192e+00,
       4.4673947e+00, 1.6076835e+00, 2.6395459e+00, 5.6002806e+00, 4.0999939e+00, 1.2753592e+00, 5.0354294e+00, 3.7727076e+00,
       8.0263649e-01, 1.2427964e+00, 3.6431937e+00, 3.7793271e+00, 2.2228610e+00, 3.4508867e+00, 2.7085490e+00, 2.6337195e-01,
       1.6311074e-01, 1.8761103e+00, 7.7175448e-02, 2.3038037e+00, 4.0986958e+00, 5.5705477e-01, 2.1202994e-01, 3.6743729e+00,
       3.6512422e+00, 9.4558908e-02, 9.8129601e-02, 1.1404475e+00, 3.5215108e+00, 3.4548654e-01, 2.2054082e+00, 3.7887070e+00,
       4.3058065e+00, 4.1560164e+00, 5.0447436e-01, 2.7261309e+00, 2.6509698e+00, 2.1195027e+00, 9.2163818e-01, 4.0538679e+00,
       4.1952800e+00, 4.3650548e+00, 2.8703063e+00, 3.3290519e+00, 7.2628268e-01, 2.7045236e+00, 4.2952977e+00, 5.3570496e+00,
       1.6386148e+00, 1.5286158e+00, 5.1936209e+00, 1.3941022e+00, 4.8292305e+00, 5.4503853e+00, 1.3913659e+00, 1.4358754e+00,
       2.9852690e-01, 4.7030445e-01, 3.8448925e+00, 1.1453194e+00, 5.0632170e+00, 1.0434015e+00, 1.0247569e+00, 5.9657729e+00
    ]) - 3
    nlY = np.array([
      -4.5367780e-01, -1.7468796e+00, 1.1564607e-01, 1.7028679e-01, -1.0809944e+00, 2.4939631e-01, 6.2081363e-01, -3.8931265e-01,
      -2.5971203e-01, 4.3530999e-01, 7.7746233e-02, 2.0159319e-01, -9.5856613e-01, 1.2473979e+00, -1.6191998e+00, 3.7308291e-01,
      -3.8183677e-01, -7.3535426e-01, 3.3555844e-01, -1.0773216e+00, -5.3513582e-01, -7.7958778e-01, 1.0297623e-01, 3.6783484e-01,
      -1.5898677e+00, -1.4197883e+00, -1.1922887e+00, -4.6486943e-01, -1.9089003e+00, -1.4049308e+00, -2.2790547e-01, 9.3191177e-01,
       7.0563809e-01, -4.7464219e-01, 9.1757352e-01, 6.0504345e-01, -8.8946049e-02, -2.6753578e-01, -1.5914154e+00, 5.0737377e-01,
      -8.9162807e-01, -7.1762831e-02, 2.7445429e-01, 4.3153344e-01, -6.1051696e-01, 4.2151929e-01, 8.1436169e-01, 2.3281771e-01,
      -1.9854765e+00, -1.5560383e+00, -1.9023357e+00, 8.0910826e-01, -1.2498807e+00, 5.2807157e-01, -1.1101869e+00, 5.0364570e-01,
       8.0254996e-03, -1.0177655e+00, -6.6971438e-01, -9.7693578e-02, 4.0492375e-01, -9.5947804e-01, 9.5328292e-02, -3.8784976e-01,
      -2.5900636e-02, 6.1409592e-01, -5.1317788e-01, -1.3024822e+00, -7.0469097e-01, 3.2307287e-01, 9.5155796e-01, -1.3152174e+00,
      -5.7484794e-01, -1.8457759e-01, -7.0320708e-01, 6.4144146e-01, 6.5483120e-02, 6.7950460e-01, 1.1377892e+00, -1.0445089e-01,
       4.1139124e-01, -1.4955381e-01, 5.3196304e-01, -1.1740466e+00, -1.3804133e+00, 6.3739798e-02, -1.9732065e+00, -1.9967737e+00,
      -5.0228110e-01, 7.7289801e-01, -1.0117830e+00, -4.8264087e-02, -1.0793363e+00, -1.8111369e+00, -1.3142035e+00, 6.1841552e-02,
      -1.3409769e+00, 7.9069173e-01, -8.0245497e-01, 2.0311058e-02, -5.1933167e-01, -2.3416586e-01, 6.2282557e-01, -1.0981787e+00,
       6.1823724e-01, -1.6572587e+00, -1.2740843e-01, 3.4881056e-01, -1.1360621e-01, -1.1916209e-02, -2.0586768e+00, -4.8510598e-01,
       8.6707862e-01, 5.5013435e-01, 2.0157393e-01, -1.2015578e-01, -1.1294046e-01, -9.8617226e-01, -2.3142540e-01, -1.8610424e-01,
       3.6714728e-01, -1.1566302e+00, -1.2204031e+00, -2.7877699e-01, 5.0594554e-01, -5.7772136e-01, -4.4265272e-01, 6.0534989e-01,
       5.2015974e-01, 2.1889455e-01, -1.4456386e+00, 2.0331987e-01, -1.4753150e+00, 1.3053871e-02, 9.1682906e-01, 7.6432793e-01,
       1.1606214e+00, -1.9170090e+00, 1.2356613e-01, -7.1277011e-01, 4.7425019e-01, -1.9744788e+00, -3.2169756e-01, 4.7886336e-01,
      -1.3704895e+00, -1.9706934e+00, -1.9364614e-01, 2.4296925e-01, -5.9314788e-01, -5.2120168e-01, 5.1426983e-01, -4.4551844e-01,
      -1.0081046e-01, -1.4857508e+00, -1.5706189e-01, -2.5221102e-01, 5.6953679e-01, -3.5363778e-01, -5.0336203e-01, -5.3892554e-01,
       8.8696943e-03, -4.5487360e-01, -1.7536194e-01, -1.8112513e+00, -9.0458595e-03, -3.3414227e-01, -4.5840843e-01, -6.3538871e-01,
       8.7041691e-01, 1.1822965e+00, -3.8220046e-01, 4.6027738e-02, 4.7173193e-01, -8.6971597e-01, -1.6510072e+00, 1.1217285e-01,
       8.2306501e-01, 7.2285343e-01, 5.2315895e-01, -1.8834054e-01, -1.1944997e+00, 6.4602940e-01, 1.0657384e+00, -2.0792999e-01,
      -2.1522765e+00, -2.1513605e+00, -4.1117445e-01, -1.9734661e+00, -3.5289056e-01, -7.0468976e-01, -1.3490872e+00, -1.6070568e+00,
       2.1285165e-01, -4.5838135e-01, 7.0495045e-01, -1.7473682e+00, -8.8853240e-01, -1.5803886e+00, -1.1030290e+00, -4.0577463e-01
    ])
    tX = nlX[::4]
    tY = nlY[::4]
    return nlX, nlY, tX, tY


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
