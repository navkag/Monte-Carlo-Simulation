import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt


def box_muller(sample_size):  # Sample from N(0, 1) using Box-Muller method.
    samples = list()

    for i in range(sample_size):
        np.random.seed(i % 10000)
        u1, u2 = np.random.uniform(size=2)

        r = np.sqrt(-2 * np.log(u1))
        theta = 2 * np.pi * u2

        z1, z2 = r * np.sin(theta), r * np.cos(theta)
        samples.append([z1, z2])

    return samples


def transformer(mu, SIGMA, samples):   # Transform each sample from N(0, 1) to N(mu, SIGMA).
    """
    :param mu: (2, 1) shape numpy array.
    :param SIGMA: (2, 2) shape numpy array.
    :param samples: (num_samples, 2) shape having N(0, 1) dist numpy array.
    :return: transformed samples with mu mean and sigma variance numpy array.
    """
    sigma_1 = np.sqrt(SIGMA[0, 0])
    sigma_2 = np.sqrt(SIGMA[1, 1])
    rho = SIGMA[0, 1] / (sigma_1 * sigma_2)

    a_11 = sigma_1
    a_12 = 0
    a_21 = rho * sigma_2
    a_22 = np.sqrt(1 - np.square(rho)) * sigma_2

    A = np.array([[a_11, a_12],
                  [a_21, a_22]])

    transformed_samples = (mu + np.matmul(A, samples.T)).T

    return transformed_samples


def main():
    a = [-0.5, 0, 0.5, 1]

    for val in a:
        mu = np.array([5, 8]).reshape(2, -1)
        SIGMA = np.array([[1, 2 * val],
                          [2 * val, 4]])

        samples = np.array(box_muller(sample_size=10_000))
        transformed_samples = transformer(mu, SIGMA, samples)

        # Plot the transformed samples.
        x1, y1 = transformed_samples[:, 0], transformed_samples[:, 1]

        plt.figure(figsize=(12, 8))
        plt.hist2d(x1, y1, bins=50)
        plt.colorbar()
        plt.title(f'2D histogram for a = {val}')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()

        # Get the actual distribution.
        actual_distribution = np.random.multivariate_normal(mu.ravel(), SIGMA, size=10_000)
        x2, y2 = actual_distribution[:, 0], actual_distribution[:, 1]
        kde = gaussian_kde([x2, y2])
        xi, yi = np.meshgrid(np.linspace(x2.min(), x2.max(), 100), np.linspace(y2.min(), y2.max(), 100))
        zi = kde(np.vstack([xi.flatten(), yi.flatten()]))

        plt.figure(figsize=(12, 8))
        plt.hist2d(x1, y1, bins=50)
        plt.colorbar()
        plt.contour(xi, yi, zi.reshape(xi.shape), levels=20, alpha=0.5, colors="black")
        plt.title(f'2D histogram overlaid with contour lines for a = {val}')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()


if __name__ == "__main__":
    main()
