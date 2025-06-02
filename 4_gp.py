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
def draw_data_widget(mo):
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
        datawidget,
        mo.hstack([run_button])
    ])
    return ScatterWidget, datawidget, run_button


@app.cell
def draw_data_extractor(datawidget, mo, np, run_button):
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
        click 'Process data' to execute the data-dependent code
        """)
    )
    return extract_X_y, has_data, raw_X, raw_y, warn_no_data


@app.cell
def _(raw_X):
    raw_X.max()
    return


@app.cell
def data_setup(nlX, nlY, plt):
    ### use pre-determined nonlinear data set
    X, y = nlX[:, None], nlY

    ### use data from the drawdata ScatterWidget above
    # X, y = raw_X, raw_y

    def z_normalize(arr):
        return (arr - arr.mean()) / arr.std()

    # What happens if you don't normalize the data?
    normalize_data = True
    normalize_data = False
    
    if normalize_data:
        X = z_normalize(X)
        y = z_normalize(y)

    def _get_lims(arr):
        _range = arr.max() - arr.min()
        return (arr.min() - 0.1*_range, arr.max() + 0.1*_range)

    x_lims = _get_lims(X)
    y_lims = _get_lims(y)

    n_data = len(y)
    assert X.shape == (n_data, 1)
    assert y.shape == (n_data,)

    plt.figure()
    plt.title("Data")
    plt.scatter(X, y)
    plt.xlim(*x_lims)
    plt.ylim(*y_lims)
    plt.gcf()
    return X, n_data, normalize_data, x_lims, y, y_lims, z_normalize


@app.cell
def _(np):

    def gaussian_kernel(x1, x2, lengthscale=1.0, variance=1.0):
        sqdist = np.subtract.outer(x1, x2)**2
        return variance * np.exp(-0.5 * sqdist / lengthscale**2)

    def linear_kernel(x1, x2, c=0.0, variance=1.0):
        return variance * (np.outer(x1, x2) + c)

    def polynomial_kernel(x1, x2, c=1.0, degree=3, variance=1.0):
        return variance * (np.outer(x1, x2) + c) ** degree

    def periodic_kernel(x1, x2, lengthscale=1.0, period=2.0, variance=1.0):
        dists = np.abs(np.subtract.outer(x1, x2))
        return variance * np.exp(-2 * (np.sin(np.pi * dists / period)**2) / lengthscale**2)

    def sum_of_periodic_and_linear_kernel(x1, x2):
        return periodic_kernel(x1, x2) + linear_kernel(x1, x2)
    return (
        gaussian_kernel,
        linear_kernel,
        periodic_kernel,
        polynomial_kernel,
        sum_of_periodic_and_linear_kernel,
    )


@app.cell
def _(np, periodic_kernel, plt, stats):
    def sample_gp(x, kernel_func, n_samples=3, **kernel_params):
        K = kernel_func(x, x, **kernel_params)
    
        samples = stats.multivariate_normal.rvs(mean=np.zeros(len(x)), cov=K + 1e-6*np.eye(len(x)), size=n_samples)
        return samples, K

    def plot_samples(kernel_func, n_samples=5
                     , **kernel_params):
        x = np.linspace(-5, 5, 300)
        samples, _ = sample_gp(x, kernel_func, n_samples, **kernel_params)
    
        fig = plt.figure(figsize=(10, 4))
        for i in range(samples.shape[0]):
            plt.plot(x, samples[i], label=f'Sample {i+1}')
        plt.title(f"Samples from Gaussian process prior with {kernel_func.__name__.replace('_', ' ').title()}")
        plt.xlim(x.min(), x.max())
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend(loc='upper left')
        return fig

    plot_samples(periodic_kernel)
    return plot_samples, sample_gp


@app.cell
def _(np):
    def gp_posterior(x_train, y_train, x_test, kernel_func, noise_std=1e-2, **kernel_params):
        K = kernel_func(x_train, x_train, **kernel_params) + noise_std**2 * np.eye(len(x_train))
        K_s = kernel_func(x_train, x_test, **kernel_params)
        K_ss = kernel_func(x_test, x_test, **kernel_params) + 1e-8 * np.eye(len(x_test))

        K_inv = np.linalg.inv(K)
    
        mean_s = K_s.T @ K_inv @ y_train
        cov_s = K_ss - K_s.T @ K_inv @ K_s
        return mean_s, cov_s
    return (gp_posterior,)


@app.cell
def _(gaussian_kernel, gp_posterior, np, plt):
    # Example data
    x_train = np.linspace(-4, 4, 21)
    y_train = np.sin(x_train) + 0.5*np.random.randn(*x_train.shape)

    x_test = np.linspace(-5, 5, 100)
    mean_s, cov_s = gp_posterior(x_train, y_train, x_test, gaussian_kernel)#, lengthscale=1.0)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(x_train, y_train, 'ro', label='Observations')
    plt.plot(x_test, mean_s, 'k', lw=2, label='Mean prediction')
    plt.fill_between(x_test, mean_s - 2*np.sqrt(np.diag(cov_s)), mean_s + 2*np.sqrt(np.diag(cov_s)), color='gray', alpha=0.3)
    plt.title("GP Posterior with RBF Kernel")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()

    return cov_s, mean_s, x_test, x_train, y_train


@app.cell
def _(np):
    def log_marginal_likelihood(x_train, y_train, kernel_func, noise=1e-2, **kernel_params):
        K = kernel_func(x_train, x_train, **kernel_params) + noise * np.eye(len(x_train))
        L = np.linalg.cholesky(K + 1e-6*np.eye(len(K)))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        logdetK = 2 * np.sum(np.log(np.diag(L)))
        return -0.5 * y_train.T @ alpha - 0.5 * logdetK - 0.5 * len(x_train) * np.log(2*np.pi)

    return (log_marginal_likelihood,)


@app.cell
def _(log_marginal_likelihood, np, plt, rbf_kernel, x_train, y_train):
    def _():
        # Fixed kernel variance
        kernel_variance = 1.0

        # Log-spaced lengthscales and noise variances
        lengthscales = np.logspace(-1.5, 0.5, 50)  # ~0.1 to 2.0
        noise_vars = np.logspace(-6, 0.5, 50)   # ~1e-4 to 0.5

        Z = np.zeros((len(lengthscales), len(noise_vars)))
        for i, l in enumerate(lengthscales):
            for j, nv in enumerate(noise_vars):
                Z[i, j] = log_marginal_likelihood(
                    x_train, y_train,
                    rbf_kernel,
                    noise=nv,
                    lengthscale=l,
                    variance=kernel_variance
                )

        L, N = np.meshgrid(lengthscales, noise_vars)

        plt.figure(figsize=(8, 6))
        cp = plt.contour(L, N, Z.T, levels=np.quantile(Z.flatten(), np.linspace(0, 1, 50)), cmap='viridis')
        plt.colorbar(cp)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel('Lengthscale (log scale)')
        plt.ylabel('Noise Variance (log scale)')
        plt.title('Log Marginal Likelihood (Log-Log Axes, Fixed Kernel Variance)')
        return plt.show()


    _()

    return


if __name__ == "__main__":
    app.run()
