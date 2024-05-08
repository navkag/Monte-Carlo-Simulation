import numpy as np


def MC_estimator(sample_size):
    np.random.seed(42)
    uniform_var = np.random.uniform(size=sample_size)
    Y = np.exp(np.sqrt(uniform_var))
    I_M = np.average(Y)

    # Calculate the value of S_n.
    S_i = 0
    mu_i = Y[0]

    for i in range(1, sample_size):
        del_i = Y[i] - mu_i
        mu_i = mu_i + del_i / (i + 1)
        S_i = S_i + (np.square(del_i) * i) / (i + 1)

    s_n = np.sqrt(S_i / (sample_size - 1))

    # Determine the confidence interval for I_M.
    confidence_interval = ((I_M - 1.96 * (s_n / np.sqrt(sample_size))).round(5), (I_M + 1.96 * (s_n / np.sqrt(sample_size))).round(5))

    # Exact value of I.
    lower_lim, upper_lim = 0, 1
    I = compute_exact_value(upper_lim) - compute_exact_value(lower_lim)
    print(f"Sample size: {sample_size}")
    print(f"Exact value(I): {I}, estimated value(I_M): {I_M}")
    return confidence_interval


def compute_exact_value(limit):
    return 2 * np.exp(np.sqrt(limit)) * (np.sqrt(limit) - 1)


def main():
    samples = [100, 1000, 10_000, 100_000]

    for sample in samples:
        confidence_interval = MC_estimator(sample)
        print(f"Confidence interval of 95 %: {confidence_interval}")
        print(f"Interval length: {(confidence_interval[1] - confidence_interval[0]).round(5)}\n")


if __name__ == "__main__":
    main()
