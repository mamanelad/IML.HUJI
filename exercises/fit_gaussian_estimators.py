from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mean_original = 10
    variance_original = 1
    samples_amount = 1000

    # Draw 1000 random samples from a normal (Gaussian) distribution N(10,1).
    X = np.random.normal(loc=mean_original, scale=variance_original,
                         size=samples_amount)

    # fit a univariate Gaussian distribution to the data,
    # which means estimating the mean and variance of the distribution
    # using the sample data.
    fit = UnivariateGaussian().fit(X)

    round_num = 3
    mu_round = np.round(fit.mu_, round_num)
    var_round = np.round(fit.var_, round_num)

    print((mu_round, var_round))

    # Question 2 - Empirically showing sample mean is consistent
    differences = []
    step_size = 10

    # Iterate over a range of sample sizes,
    # increasing by the step size each time
    for curr_sample_amount in range(step_size, len(X) + 1, step_size):
        # Create a new Univariate Gaussian object and call its fit()
        # method on a subset of the data X
        u_g = UnivariateGaussian()
        u_g.fit(X[:curr_sample_amount])

        # Calculate the absolute difference between the
        # true population mean and the estimated sample mean
        mu_diff = np.abs(mean_original - u_g.mu_)

        # Add the difference to the mu_hat list
        differences.append(mu_diff)

    go.Figure(go.Scatter(x=list(range(len(differences))), y=differences,
                         mode="markers",
                         marker=dict(color="black")),
              layout=dict(template="simple_white",
                          title="Empirical Mean Estimation Error Across Sample Sizes",
                          xaxis_title=r"$\text{Amount of Samples }n$",
                          yaxis_title=
                          r"$\text{Estimated Mean }\hat{\mu}_n$")) \
        .write_image("Empirical_Mean_Estimation_Error_Across_Sample_Sizes.png")

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf = fit.pdf(X)
    go.Figure(go.Scatter(x=X, y=pdf, mode="markers",
                         marker=dict(color="black")),
              layout=dict(template="simple_white",
                          title="PDF Comparison",
                          xaxis_title=r"$X$",
                          yaxis_title=
                          r"$\mathcal{N}(\hat{\mu},\hat{\sigma}^2)$")) \
        .write_image("empirical_fitted.pdf.png")


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mean_original = [0, 0, 4, 0]
    variance_original = np.array([[1, 0.2, 0, 0.5],
                                  [0.2, 2, 0, 0],
                                  [0, 0, 1, 0],
                                  [0.5, 0, 0, 1]])
    samples_amount = 1000
    X = np.random.multivariate_normal(mean_original, cov=variance_original,
                                      size=samples_amount)

    fit = MultivariateGaussian().fit(X)

    round_num = 3
    mu_round = np.round(fit.mu_, round_num)
    cov_round = np.round(fit.cov_, round_num)

    print(mu_round)
    print(cov_round)

    # Question 5 - Likelihood evaluation
    # raise NotImplementedError()
    num_to_generate = 200
    max_interval_value = 10
    min_interval_value = -10
    f_vals = np.linspace(min_interval_value, max_interval_value,
                         num_to_generate)

    log_likelihood_array = np.zeros((num_to_generate, num_to_generate))

    for i in range(len(f_vals)):
        for j in range(len(f_vals)):
            log_likelihood_array[i, j] = MultivariateGaussian.log_likelihood(
                np.array([f_vals[i], 0, f_vals[j], 0]), variance_original, X)

    go.Figure(go.Heatmap(x=f_vals, y=f_vals, z=log_likelihood_array),
              layout=dict(template="simple_white",
                          title="Log-Likelihood of Gaussian: f1, f3",
                          xaxis_title=r"$\mu_3$",
                          yaxis_title=r"$\mu_1$")) \
        .write_image("loglikelihood_Gaussian_heatmap.png")

    # # Question 6 - Maximum likelihood
    # Get maximum value in the log likelihood array.
    max_index = np.argmax(log_likelihood_array)

    # Convert the flattened index to row and column indices.
    max_row, max_col = np.unravel_index(max_index, log_likelihood_array.shape)

    # Extract max values of f1 and f3.
    max_f1 = f_vals[max_row]
    max_f3 = f_vals[max_col]

    # Round the values.
    round_num = 3
    max_f1 = np.round(max_f1, round_num)
    max_f3 = np.round(max_f3, round_num)

    # Combine the values
    max_f = [max_f1, max_f3]

    print("Maximum log-likelihood (f1, f3):", max_f)


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
