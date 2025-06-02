# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "drawdata==0.3.8",
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
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    return np, plt, stats


@app.cell
def _(mo):
    mo.md(
        r"""
        # Kernel functions and Gaussian process prior

        The **kernel function** $k(x, x')$ defines the core characteristics of a Gaussian process, e.g. differentiability
        """
    )
    return


@app.cell
def _(np):
    def gaussian_kernel(x1, x2, lengthscale=1.0, variance=1.0):
        sqdist = np.subtract.outer(x1, x2)**2
        return variance * np.exp(-0.5 * sqdist / lengthscale**2)

    def matern12(x1, x2, lengthscale=1.0, variance=1.0):
        absdist = np.abs(np.subtract.outer(x1, x2))
        return variance * np.exp(-0.5 * absdist / lengthscale)

    def linear_kernel(x1, x2, c=0.0, variance=1.0):
        return variance * (np.outer(x1, x2) + c)

    def polynomial_kernel(x1, x2, c=1.0, degree=3, variance=1.0):
        return variance * (np.outer(x1, x2) + c) ** degree

    def periodic_kernel(x1, x2, lengthscale=1.0, period=2.0, variance=1.0):
        dists = np.abs(np.subtract.outer(x1, x2))
        return variance * np.exp(-2 * (np.sin(np.pi * dists / period)**2) / lengthscale**2)

    def product_of_gaussian_and_linear_kernel(x1, x2, lengthscale=1):
        return gaussian_kernel(x1, x2, lengthscale=lengthscale) * linear_kernel(x1, x2)
    
    def sum_of_periodic_and_linear_kernel(x1, x2):
        return periodic_kernel(x1, x2) + linear_kernel(x1, x2)
    return (
        gaussian_kernel,
        linear_kernel,
        matern12,
        periodic_kernel,
        polynomial_kernel,
        product_of_gaussian_and_linear_kernel,
        sum_of_periodic_and_linear_kernel,
    )


@app.cell
def _(mo):
    mo.md(r"""## Gaussian process prior samples""")
    return


@app.cell
def _(np, plt, stats):
    def sample_gp(x, kernel_func, n_samples=3, **kernel_params):
        K = kernel_func(x, x, **kernel_params)

        mean = np.zeros_like(x)  # we could also use a non-zero mean function
        cov = K + 1e-6*np.eye(len(x))  # we add a small value on the diagonal to ensure positive-definiteness

        samples = stats.multivariate_normal.rvs(mean=mean, cov=cov, size=n_samples)
        return samples, K

    def plot_samples(kernel_func, n_samples=5, **kernel_params):
        """ visualizes samples from the Gaussian process prior defined by `kernel_func` with its `kernel_params` """
        x = np.linspace(-5, 5, 300)
        samples, _ = sample_gp(x, kernel_func, n_samples, **kernel_params)
    
        fig = plt.figure(figsize=(10, 4))
        for i in range(samples.shape[0]):
            plt.plot(x, samples[i], label=f'Sample {i+1}')

        kernel_name = kernel_func.__name__.replace('_', ' ').title()
        extra = ", ".join(f"{k}={v}" for (k, v) in kernel_params.items())
        if extra != "": extra = ", "+extra
        plt.title(f"Samples from Gaussian process prior with {kernel_name}" + extra)
        plt.xlim(x.min(), x.max())
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend(loc='upper left')
        return fig
    return plot_samples, sample_gp


@app.cell
def _(
    gaussian_kernel,
    linear_kernel,
    matern12,
    periodic_kernel,
    plot_samples,
    sum_of_periodic_and_linear_kernel,
):
    [
        plot_samples(k)
        for k in [gaussian_kernel, matern12, linear_kernel, periodic_kernel, sum_of_periodic_and_linear_kernel]
    ]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Kernel hyperparameters
        Kernels commonly have **hyperparameters** that change their properties. For example, the Gaussian kernel has a `lengthscale` hyperparameter that controls the distance over which function values remain correlated with each other:
        """
    )
    return


@app.cell
def _(gaussian_kernel, plot_samples):
    [
        plot_samples(gaussian_kernel, lengthscale=l)
        for l in [0.3, 1, 3]
    ]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The Gaussian kernel also has a `variance` hyperparameter:""")
    return


@app.cell
def _(gaussian_kernel, plot_samples):
    [
        plot_samples(gaussian_kernel, variance=v)
        for v in [0.1, 1, 5]
    ]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Gaussian process posterior

        Now that we've got some feeling for the *prior*, let's see how conditioning on the observed data points gives us the posterior.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Data

        First, let's pick some data.
        """
    )
    return


@app.cell
def _():
    use_preset_data = True  # just use a predefined set of data
    # use_preset_data = False  # draw your own data!
    return (use_preset_data,)


@app.cell(hide_code=True)
def preset_non_linear_data(np):
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


@app.cell(hide_code=True)
def draw_data_widget(mo, use_preset_data):
    from drawdata import ScatterWidget

    datawidget = ScatterWidget()
    datawidget.data = [{
      "x": 300,
      "y": 300,
      "color": "#1f77b4",
      "label": "a"
    }, {
      "x": 500,
      "y": 200,
      "color": "#1f77b4",
      "label": "a"
    }]
    datawidget = mo.ui.anywidget(datawidget)

    # run_button to run model
    run_button = mo.ui.run_button(label="Process data", kind="success")

    mo.vstack([
        mo.hstack([run_button]),
        datawidget,
    ]) if not use_preset_data else None
    return ScatterWidget, datawidget, run_button


@app.cell(hide_code=True)
def draw_data_extractor(datawidget, mo, np, run_button, use_preset_data):
    if use_preset_data:
        raw_X = raw_y = None
    else:
        has_data = False
    
        def extract_X_y(data):
            if len(data[0]) == 0:
                return np.zeros((0, 1)), np.zeros((0,))
            if data[0].shape[1] == 1:  # data[0] is X [N,1] and data[1] is y [N]
                raw_X, raw_y = data
            else:  # data[0] is x and y stacked into [N,2], data[1] is label that we ignore
                xy, _ = data
                raw_X = xy[:, :1]
                raw_y = xy[:, 1]
            return raw_X, raw_y
    
        raw_X, raw_y = extract_X_y(datawidget.data_as_X_y)
        has_data = len(raw_X) >= 1
    
        def warn_no_data():
            warning_msg = mo.md(""" /// warning
        Need more data, please draw at least one point in the scatter widget
        """)
            mo.stop(not has_data, warning_msg)
    
        warn_no_data()
    
        mo.stop(
            not run_button.value,  # if button hasn't been clicked yet
            mo.md(""" /// tip 
            click the 'Process data' button to execute the drawdata-dependent code
            """)
        )
    return extract_X_y, has_data, raw_X, raw_y, warn_no_data


@app.cell
def data_setup(nlX, nlY, plt, raw_X, raw_y, use_preset_data):
    if use_preset_data:
        # use pre-determined nonlinear data set
        X, y = nlX[:, None], nlY
    else:
        # use data from the drawdata ScatterWidget below
        X, y = raw_X, raw_y

    def z_normalize(arr):
        return (arr - arr.mean()) / arr.std()

    # What happens if you don't normalize the data?
    normalize_data = True
    # normalize_data = False  # 🔧

    if normalize_data:
        X = z_normalize(X)
        y = z_normalize(y)

    def _get_lims(arr):
        _range = arr.max() - arr.min()
        return (arr.min() - 0.3*_range, arr.max() + 0.3*_range)

    x_lims = _get_lims(X)
    y_lims = _get_lims(y)

    n_data = len(y)
    assert X.shape == (n_data, 1)
    assert y.shape == (n_data,)

    def plot_data():
        fig = plt.figure()
        plt.title("Data")
        plt.scatter(X, y)
        plt.xlim(*x_lims)
        plt.ylim(*y_lims)
        return fig

    plot_data()
    return X, n_data, normalize_data, plot_data, x_lims, y, y_lims, z_normalize


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Posterior computation

        Similar to how conditioning a multivariate normal distribution results in another (smaller) multivariate normal distribution, conditioning a Gaussian process on some observations results in another Gaussian process. The key difference is that instead of blocks of a finite-size covariance matrix, we now evaluate the covariance function on the input points corresponding to conditioning set (training data) and evaluation set (test points).
        """
    )
    return


@app.cell
def _(np):
    def gp_posterior(x_train, y_train, x_test, kernel_func, noise_std=1e-2, **kernel_params):
        K = kernel_func(x_train, x_train, **kernel_params) + noise_std**2 * np.eye(len(x_train))
        K_s = kernel_func(x_train, x_test, **kernel_params)
        K_ss = kernel_func(x_test, x_test, **kernel_params) + 1e-8 * np.eye(len(x_test))

        K_inv = np.linalg.inv(K)

        # this is effectively the same as for the multivariate normal distributions
        mean_s = K_s.T @ K_inv @ y_train
        cov_s = K_ss - K_s.T @ K_inv @ K_s
        return mean_s, cov_s
    return (gp_posterior,)


@app.cell
def _(X, gaussian_kernel, gp_posterior, np, plot_data, plt, x_lims, y):
    def plot_posterior(train_x, train_y, x_test, kernel_func, noise_std, **kernel_params):
        mean_s, cov_s = gp_posterior(train_x, train_y, x_test, kernel_func, noise_std=noise_std, **kernel_params)
        std_s = np.sqrt(np.diag(cov_s))
    
        kernel_name = kernel_func.__name__.replace('_', ' ').title()
        extra = ", ".join(f"{k}={np.round(v,3)}" for (k, v) in kernel_params.items())
        if extra != "": extra = ", "+extra
        plt.title(f"Gaussian process posterior with {kernel_name}" + extra + f", noise std.dev={np.round(noise_std,3)}")
    
        plt.plot(x_test, mean_s, 'k', lw=2, label='Mean prediction')
        plt.fill_between(x_test, mean_s - 2*std_s, mean_s + 2*std_s, color='gray', alpha=0.3, label='+/- 2 std.dev.')


    # Plot
    plot_data()
    xtest = np.linspace(*x_lims, 200)
    plot_posterior(
        X.squeeze(axis=1), y, xtest,
        gaussian_kernel,  # 🔧
        noise_std=1, lengthscale=1,  # 🔧 underfitting?
        # noise_std=0.1, lengthscale=0.1,  # 🔧 overfitting?
    )
    plt.legend()
    plt.show()

    return plot_posterior, xtest


@app.cell
def _(mo):
    mo.md(
        r"""
        # Hyperparameter selection and marginal likelihood

        To look at how to choose hyperparameter values, we will consider a very small toy dataset. Which of the following two posteriors is better?
        """
    )
    return


@app.cell
def _(gaussian_kernel, mo, np, plot_posterior, plt):
    # Example data
    np.random.seed(123)
    x_train = np.linspace(-4, 4, 21)
    y_train = np.sin(x_train) + 0.5*np.random.randn(*x_train.shape)

    x_test = np.linspace(-5, 5, 200)

    def plot_toy_data():
        plt.plot(x_train, y_train, 'ro', label='Observations')
        plt.xlim(x_test.min(), x_test.max())
        plt.ylim(-3, 3)
        plt.xlabel("x")
        plt.ylabel("f(x)")

    def plot_toy_posterior(kernel_func, **kwargs):
        fig = plt.figure(figsize=(8, 5))
        plot_toy_data()
        plot_posterior(x_train, y_train, x_test, kernel_func, **kwargs)
        plt.legend()
        return fig

    mo.hstack([
        plot_toy_posterior(gaussian_kernel, noise_std=1, lengthscale=1),  # longer lengthscale, higher noise
        plot_toy_posterior(gaussian_kernel, noise_std=0.1, lengthscale=0.4),  # shorter lengthscale, lower noise
    ])
    return plot_toy_data, plot_toy_posterior, x_test, x_train, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        How can we pick between them?

        - expert/domain knowledge
        - cross-validation
        - put prior distribution on hyperparameters (-> MCMC)
        - **optimize marginal likelihood**
        """
    )
    return


@app.cell
def _(np):
    def log_marginal_likelihood(x_train, y_train, kernel_func, noise_std, **kernel_params):
        K = kernel_func(x_train, x_train, **kernel_params)
        Sigma = noise_std**2 * np.eye(len(x_train))
        L = np.linalg.cholesky(K + Sigma + 1e-6*np.eye(len(K)))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))  # inv(K + Sigma) @ y_train
        logdetK = 2 * np.sum(np.log(np.diag(L)))  # log |K + Sigma|
        return -0.5 * y_train.T @ alpha - 0.5 * logdetK - 0.5 * len(x_train) * np.log(2*np.pi)
    return (log_marginal_likelihood,)


@app.cell
def _(gaussian_kernel, log_marginal_likelihood, mo, np, x_train, y_train):
    @mo.cache
    def get_lml_L_N_Z(kernel_variance):
        """ evaluates log-marginal likelihood on a grid of lengthscales and noise scales """
        # Log-spaced lengthscales and noise variances
        lengthscales = np.logspace(np.log10(0.01), np.log10(20), 50)
        noise_stds = np.logspace(np.log10(1e-2), np.log10(2), 50)

        Z = np.zeros((len(lengthscales), len(noise_stds)))
        for i, l in enumerate(lengthscales):
            for j, n in enumerate(noise_stds):
                Z[i, j] = log_marginal_likelihood(
                    x_train, y_train,
                    gaussian_kernel,
                    noise_std=n,
                    lengthscale=l,
                    variance=kernel_variance
                )

        L, N = np.meshgrid(lengthscales, noise_stds)
        return L, N, Z.T
    return (get_lml_L_N_Z,)


@app.cell
def _(get_lml_L_N_Z, np, plt):
    def plot_marginal_likelihood_surface_gaussian_kernel_lengthscale_noise(kernel_variance=1.0):
        L, N, Z = get_lml_L_N_Z(kernel_variance)

        fig = plt.figure(figsize=(8, 6))
        cp = plt.contour(L, N, Z, levels=np.quantile(Z.flatten(), np.linspace(0, 1, 30)), cmap='viridis')
        plt.colorbar(cp)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel('Lengthscale')
        plt.ylabel('Noise std.dev.')
        plt.title(f'Log Marginal Likelihood (Kernel variance = {kernel_variance} fixed)')
        return fig

    plot_marginal_likelihood_surface_gaussian_kernel_lengthscale_noise()
    plt.show()
    return (
        plot_marginal_likelihood_surface_gaussian_kernel_lengthscale_noise,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Numerical optimization of marginal likelihood

        In practice, we would not evaluate the marginal likelihood on a grid, but instead use numeric optimization to find a good point estimate. Let's try that out:
        """
    )
    return


@app.cell
def _(mo, np):
    def mylogspace(start, stop, n):
        return np.round(np.logspace(np.log10(start), np.log10(stop), n), 3)

    ui_init_l = mo.ui.slider(steps=mylogspace(0.01, 10, 10), value=1, label="init. lengthscale")
    ui_init_n = mo.ui.slider(steps=mylogspace(0.01, 10, 10), value=1, label="init. noise std.dev")
    ui_init_v = mo.ui.slider(steps=mylogspace(0.01, 10, 10), value=1, label="init. kernel variance")

    ui_init_all = mo.hstack([ui_init_l, ui_init_n, ui_init_v])
    mo.vstack([
        mo.md("""We have to start optimizing from some initial choice of hyperparameters:"""),
        ui_init_all
    ])
    return mylogspace, ui_init_all, ui_init_l, ui_init_n, ui_init_v


@app.cell
def _(ui_init_l, ui_init_n, ui_init_v):
    init_noisestd = ui_init_n.value
    init_variance = ui_init_v.value
    init_lengthscale = ui_init_l.value
    return init_lengthscale, init_noisestd, init_variance


@app.cell
def _(
    gaussian_kernel,
    init_lengthscale,
    init_noisestd,
    init_variance,
    log_marginal_likelihood,
    np,
    x_train,
    y_train,
):
    from scipy.optimize import minimize

    initial_params = [init_lengthscale, init_variance, init_noisestd]  # lengthscale, kernel variance, noise std.dev.
    initial_log_params = np.log(initial_params)

    def neg_log_marginal_likelihood(log_params, x_train, y_train):
        log_lengthscale, log_variance, log_noisestd = log_params
        kernel_func = gaussian_kernel
        kernel_params = dict(lengthscale=np.exp(log_lengthscale), variance=np.exp(log_variance))
        noise_std = np.exp(log_noisestd)
        try:
            lml = log_marginal_likelihood(x_train, y_train, kernel_func, noise_std=noise_std, **kernel_params)
        except np.linalg.LinAlgError:
            lml = -np.inf  # for failed Cholesky
        return -lml

    # ------------------------------
    # Optimization
    # ------------------------------
    trajectory_log = [initial_log_params.copy()]

    def callback(log_params):
        trajectory_log.append(log_params.copy())

    res = minimize(
        fun=neg_log_marginal_likelihood,
        x0=initial_log_params,
        args=(x_train, y_train),
        callback=callback,
        method='L-BFGS-B'
    )

    opt_trajectory = np.exp(np.array(trajectory_log))
    opt_trajectory_lengthscale = opt_trajectory[:, 0]
    opt_trajectory_variance = opt_trajectory[:, 1]
    opt_trajectory_noisestd = opt_trajectory[:, 2]

    # ------------------------------
    # Extract and Print Results
    # ------------------------------
    opt_lengthscale, opt_variance, opt_noisestd = np.exp(res.x)
    print("Optimization success:", res.success)
    print("Optimized lengthscale:   ", opt_lengthscale)
    print("Optimized variance:      ", opt_variance)
    print("Optimized noise variance:", opt_noisestd)
    print("Final negative log marginal likelihood:", res.fun)

    return (
        callback,
        initial_log_params,
        initial_params,
        minimize,
        neg_log_marginal_likelihood,
        opt_lengthscale,
        opt_noisestd,
        opt_trajectory,
        opt_trajectory_lengthscale,
        opt_trajectory_noisestd,
        opt_trajectory_variance,
        opt_variance,
        res,
        trajectory_log,
    )


@app.cell
def _(
    gaussian_kernel,
    init_lengthscale,
    init_noisestd,
    mo,
    opt_lengthscale,
    opt_noisestd,
    opt_trajectory_lengthscale,
    opt_trajectory_noisestd,
    opt_variance,
    plot_marginal_likelihood_surface_gaussian_kernel_lengthscale_noise,
    plot_toy_posterior,
    plt,
    ui_init_all,
):
    _fig_lml = plot_marginal_likelihood_surface_gaussian_kernel_lengthscale_noise(
        kernel_variance=1
        # kernel_variance=np.round(opt_variance, 3)
    )
    plt.plot([init_lengthscale], [init_noisestd], 'bo')
    plt.plot(opt_trajectory_lengthscale, opt_trajectory_noisestd)
    plt.plot([opt_lengthscale], [opt_noisestd], 'r*', ms=10)

    _fig_pred = plot_toy_posterior(
        gaussian_kernel,
        noise_std=opt_noisestd,
        lengthscale=opt_lengthscale,
        variance=opt_variance,
    )

    mo.vstack([ui_init_all, mo.hstack([_fig_lml]), mo.hstack([_fig_pred])])
    return


if __name__ == "__main__":
    app.run()
