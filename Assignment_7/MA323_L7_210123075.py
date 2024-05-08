import numpy as np


def MC_estimator(sample_size):
    np.random.seed(42)
    sample_size = int(sample_size / 2)
    uniform_var = np.random.uniform(size=sample_size)
    Y = (np.exp(np.sqrt(uniform_var)) + np.exp(np.sqrt(1 - uniform_var))) / 2
    I_M = np.average(Y)

    s_n = np.sqrt(np.sum(np.square(Y - I_M)) / (sample_size - 1))

    # Determine the confidence interval for I_M.
    confidence_interval = ((I_M - 1.96 * (s_n / np.sqrt(sample_size))).round(5), (I_M + 1.96 * (s_n / np.sqrt(sample_size))).round(5))

    # Exact value of I.
    lower_lim, upper_lim = 0, 1
    I = compute_exact_value(upper_lim) - compute_exact_value(lower_lim)
    print(f"Sample size: {sample_size * 2}")
    print(f"Exact value(I): {I}, estimated value(I_M): {I_M}")
    print(f"Variance: {s_n ** 2}")
    return confidence_interval


def compute_exact_value(limit):
    return 2 * np.exp(np.sqrt(limit)) * (np.sqrt(limit) - 1)


def main():
    samples = [100, 1000, 10_000, 100_000]
    print(f"Antithetic Method of Estimation")

    for sample in samples:
        confidence_interval = MC_estimator(sample)
        print(f"Confidence interval of 95 %: {confidence_interval}")
        print(f"Interval length: {(confidence_interval[1] - confidence_interval[0]).round(5)}\n")


if __name__ == "__main__":
    main()
