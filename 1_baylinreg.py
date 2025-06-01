# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "anthropic==0.49.0",
#     "drawdata==0.3.7",
#     "marimo",
#     "matplotlib==3.10.1",
#     "numpy==2.2.4",
#     "scikit-learn==1.6.1",
#     "scipy==1.15.2",
# ]
# ///

import marimo

__generated_with = "0.12.5"
app = marimo.App(width="medium")


app._unparsable_cell(
    r"""
    import numpy as np
    from drawdata import ScatterWidget
    """,
    name="_"
)


@app.cell
def _(ScatterWidget, mo):
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
    run_button = mo.ui.run_button(label="Run Linear Regression", kind="success")

    mo.vstack([datawidget, mo.hstack([run_button])])
    return datawidget, run_button


@app.cell
def _(datawidget, mo, np, run_button):
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
    has_data = len(raw_X) >= 2

    def warn_no_data():
        warning_msg = mo.md(""" /// warning
    Need more data, please draw at least two points in the scatter widget
    """)
        mo.stop(not has_data, warning_msg)

    warn_no_data()

    mo.stop(
        not run_button.value,  # if button hasn't been clicked yet
        mo.md(""" /// tip 
        click 'Run Linear Regression' to see the model
        """)
    )
    return extract_X_y, has_data, raw_X, raw_y, warn_no_data


@app.cell
def _(raw_X, raw_y):
    # What happens if you don't normalize the data?
    normalize_data = True

    if normalize_data:
        x_lims = y_lims = (-3, 3)
        X = (raw_X - raw_X.mean()) / raw_X.std()
        y = (raw_y - raw_y.mean()) / raw_y.std()
    else:
        x_lims = (-100, 900)
        y_lims = (-100, 600)
        X = raw_X
        y = raw_y
    return X, normalize_data, x_lims, y, y_lims


@app.cell
def _(np):
    def with_constant(X):
        return np.hstack([np.ones_like(X), X])

    def predict_linear(X, theta):
        """Predict y values using linear model with intercept and slope."""
        X_aug = with_constant(X)
        return X_aug @ theta.T
    return predict_linear, with_constant


@app.cell
def _(np, plt):
    def get_theta_grid_contour_plot(func, label):
        # Define grid limits and resolution
        theta_min, theta_max = -3, 3
        grid_size = 100

        # Create grid of theta values
        theta0_grid = np.linspace(theta_min, theta_max, grid_size)
        theta1_grid = np.linspace(theta_min, theta_max, grid_size)
        theta0_mesh, theta1_mesh = np.meshgrid(theta0_grid, theta1_grid)

        # Calculate RMSE for each theta combination
        grid = np.zeros_like(theta0_mesh)
        for i in range(grid_size):
            for j in range(grid_size):
                theta = np.array([theta0_mesh[i, j], theta1_mesh[i, j]])
                grid[i, j] = func(theta)

        def plot():
            # Plot contour of loss surface
            contour = plt.contourf(theta0_mesh, theta1_mesh, grid, 50, cmap='viridis')
            plt.colorbar(contour, label=label)
            # Add contour lines
            plt.contour(theta0_mesh, theta1_mesh, grid, 20, colors='white', alpha=0.5, linestyles='solid')

        return plot
    return (get_theta_grid_contour_plot,)


@app.cell
def _(X, get_theta_grid_contour_plot, np, predict_linear, y):
    def rmse(theta):
        y_pred = predict_linear(X, theta)
        return np.sqrt(np.mean((y - y_pred)**2))

    rmse_contour_plotter = get_theta_grid_contour_plot(rmse, 'RMSE')
    return rmse, rmse_contour_plotter


@app.cell
def _(plt, rmse_contour_plotter):
    def plot_rmse_surface():
        rmse_contour_plotter()
        plt.xlabel('Intercept (θ₀)')
        plt.ylabel('Slope (θ₁)')
        plt.title('Loss Surface: RMSE as a function of model parameters')
    return (plot_rmse_surface,)


@app.cell
def _(get_theta_grid_contour_plot, mvn, plt):
    pdf_contour_plotter = get_theta_grid_contour_plot(lambda theta: mvn.pdf(theta), 'pdf')

    def plot_pdf_surface():
        pdf_contour_plotter()
        plt.xlabel('Intercept (θ₀)')
        plt.ylabel('Slope (θ₁)')
        plt.title('Loss Surface: RMSE as a function of model parameters')
    return pdf_contour_plotter, plot_pdf_surface


@app.cell
def _(mean_vector, plot_rmse_surface, plt):
    plot_rmse_surface()
    plt.scatter([mean_vector[0]], [mean_vector[1]], color='red', marker='*', ec='white', s=200, zorder=3)
    return


@app.cell
def _(
    X,
    checkbox_draw_dist,
    cov_matrix,
    mean_vector,
    mo,
    np,
    plt,
    predict_linear,
    slider_intercept_mean,
    slider_slope_mean,
    stats,
    x_lims,
    y,
    y_lims,
):
    plt.figure(figsize=(5,3), dpi=150)
    plt.xlim(*x_lims)
    plt.ylim(*y_lims)
    plt.scatter(X, y)

    def _():
        X_plot = np.linspace(*x_lims, 300)[:, None]
        if checkbox_draw_dist.value:
            theta = stats.multivariate_normal(mean=mean_vector, cov=cov_matrix).rvs(size=100)
            alpha = 0.2
        else:
            theta = mean_vector
            alpha = 1
        y_plot_pred = predict_linear(X_plot, theta)
        plt.plot(X_plot, y_plot_pred, '-', c='k', alpha=alpha)

    _()

    mo.vstack([mo.hstack([slider_slope_mean, slider_intercept_mean, checkbox_draw_dist]), mo.hstack([plt.gcf()])])
    return


@app.cell
def _(plot_pdf_surface, plt):
    plot_pdf_surface()
    plt.gca()
    return


@app.cell
def _(mo):
    checkbox_draw_dist = mo.ui.checkbox(label="distribution?")
    return (checkbox_draw_dist,)


@app.cell
def _(mo, np):
    slider_intercept_mean = mo.ui.slider(steps=np.round(np.linspace(-3, 3, 31), decimals=3), label="intercept mean", value=0)
    slider_slope_mean = mo.ui.slider(steps=np.round(np.linspace(-3, 3, 31), decimals=3), label="slope mean", value=0)
    slider_intercept_std = mo.ui.slider(steps=np.logspace(-2, 2, 21), label="intercept std", value=1)
    slider_slope_std = mo.ui.slider(steps=np.logspace(-2, 2, 21), label="slope std", value=1)
    slider_correlation = mo.ui.slider(steps=np.round(np.linspace(-1, 1, 21)[1:-1], decimals=3), label="correlation", value=0.2)
    mo.vstack([slider_intercept_mean, slider_slope_mean, slider_intercept_std, slider_slope_std, slider_correlation])
    return (
        slider_correlation,
        slider_intercept_mean,
        slider_intercept_std,
        slider_slope_mean,
        slider_slope_std,
    )


@app.cell
def _():
    import marimo as mo
    from scipy import stats
    return mo, stats


@app.cell
def _(np, slider_intercept_mean, slider_slope_mean):
    mean_vector = np.array([
        slider_intercept_mean.value,
        slider_slope_mean.value,
    ])
    mean_vector
    return (mean_vector,)


@app.cell
def _(np, slider_correlation, slider_intercept_std, slider_slope_std):
    def _():
        s1 = slider_intercept_std.value
        s2 = slider_slope_std.value
        rho = slider_correlation.value
        return np.array([
            [s1**2, s1*s2*rho],
            [s1*s2*rho, s2**2],
        ])
    cov_matrix = _()
    cov_matrix
    return (cov_matrix,)


@app.cell
def _(cov_matrix, mean_vector, stats):
    # Create the multivariate normal distribution
    mvn = stats.multivariate_normal(mean=mean_vector, cov=cov_matrix)

    # Example: Generate random samples from the distribution
    theta_samples = mvn.rvs(size=100)
    return mvn, theta_samples


@app.cell
def _():
    # theta = np.random.randn(100, 2)
    return


@app.cell
def _(X_lims, np, plt, theta, with_constant):
    X_plot = np.linspace(*X_lims, 300)[:, None]
    X_plot_aug = with_constant(X_plot)
    y_plot_pred = X_plot_aug @ theta.T

    plt.xlim(*X_lims)
    plt.ylim(*X_lims)
    plt.plot(X_plot, y_plot_pred, c='b', alpha=0.3)
    return X_plot, X_plot_aug, y_plot_pred


@app.cell
def _(X, np, plt, y):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Make predictions
    y_pred = model.predict(X)

    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    # Print results
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")
    print(f"Mean squared error: {mse:.2f}")
    print(f"R² score: {r2:.2f}")

    # Plot data vs predictions
    plt.scatter(X, y, label="Data")

    # plot regression curve
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_range = model.predict(x_range)
    plt.plot(x_range, y_range, color="green", label="Prediction")

    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression")
    plt.legend()
    return (
        LinearRegression,
        mean_squared_error,
        model,
        mse,
        r2,
        r2_score,
        x_range,
        y_pred,
        y_range,
    )


@app.cell
def _():
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    return (plt,)


@app.cell
def _():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    return PolynomialFeatures, make_pipeline


@app.cell
def _(
    LinearRegression,
    PolynomialFeatures,
    X,
    make_pipeline,
    mean_squared_error,
    np,
    plt,
    r2_score,
    y,
):
    def _():
        # Set the polynomial degree
        n = 4

        # Create polynomial regression model
        poly_model = make_pipeline(
            PolynomialFeatures(degree=n), LinearRegression()
        )

        # Fit the model
        poly_model.fit(X, y)

        # Make predictions
        y_pred = poly_model.predict(X)

        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        print(f"Polynomial Regression (degree={n}):")
        print(f"Mean squared error: {mse:.2f}")
        print(f"R² score: {r2:.2f}")

        # Create a smooth curve for plotting the polynomial regression line
        X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_plot = poly_model.predict(X_plot)

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color="blue", alpha=0.5, label="Training data")
        plt.plot(
            X_plot,
            y_plot,
            color="red",
            linewidth=2,
            label=f"Polynomial regression (degree={n})",
        )
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title(f"Polynomial Regression (degree={n})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        return plt.gca()


    _()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
