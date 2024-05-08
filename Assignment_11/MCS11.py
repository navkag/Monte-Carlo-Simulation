import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def box_muller(sample_size):  # Sample from N(0, 1) using Box-Muller method.
    samples = list()

    for i in range(sample_size):
        u1, u2 = np.random.uniform(size=2)

        r = np.sqrt(-2 * np.log(u1))
        theta = 2 * np.pi * u2

        z1, z2 = r * np.sin(theta), r * np.cos(theta)
        samples.append([z1, z2])

    return samples


def gen_mixed_dist_samples(sample_size, K, pi, mu, sigma):
    samples = list()
    for s in range(sample_size):
        u = np.random.uniform()
        q = np.cumsum(pi)
        
        for k in range(1, K + 1):
            if q[k - 1] < u <= q[k]:
                samples.append(box_muller(1)[0][0] * sigma[k - 1] + mu[k - 1]) 
                break
    
    return samples


def generate_sample_paths(sample_size, mu, sigma, initial_value):
    samples = list()
    samples.append(initial_value)
    
    for s in range(sample_size):
        samples.append(samples[-1] + mu * (1 / sample_size) + sigma * np.sqrt(1 / sample_size) * box_muller(1)[0][0])

    return samples


def main():
    print("Question-1 _______________________________________")
    samples = gen_mixed_dist_samples(sample_size=5000, K=3, pi=[0, 1/2, 1/3, 1/6], mu=[-1,0,1], sigma=(1/4, 1, 1/2))
    print(f"Average of the generated random numbers: {np.mean(samples)}")
    
    print("\nQuestion-2 _______________________________________")
    paths = list()
    
    plt.figure(figsize=(12, 8))
    for i in range(10):
        paths.append(generate_sample_paths(5000, 0, 1, 0))
        path_name = f"Path - {i + 1}"
        plt.plot(np.linspace(0, 5, 5001), paths[-1], label=path_name)
        
    plt.title("Question2: Brownian Motion")
    plt.xlabel("Time")
    plt.ylabel("W(t)")
    plt.legend()
    plt.show()
    
    paths = np.array(paths)
    print(f"Expected value of W[2]: {np.mean(paths[:, 2000])}")
    print(f"Expected value of W[5]: {np.mean(paths[:, 5000])}")
    
    print("\nQuestion-3 _______________________________________")
    new_paths = list()
    
    plt.figure(figsize=(12, 8))
    for i in range(10):
        new_paths.append(generate_sample_paths(5000, 0.06, 0.3, 5))
        path_name = f"Path - {i + 1}"
        plt.plot(np.linspace(0, 5, 5001), new_paths[-1], label=path_name)
        
    plt.title("Question3: Brownian Motion")
    plt.xlabel("Time")
    plt.ylabel("X(t)")
    plt.legend()
    plt.show()
    
    new_paths = np.array(new_paths)
    print(f"Expected value of W[2]: {np.mean(new_paths[:, 2000])}")
    print(f"Expected value of W[5]: {np.mean(new_paths[:, 5000])}")
    
    
if __name__ == "__main__":
    main()
                    