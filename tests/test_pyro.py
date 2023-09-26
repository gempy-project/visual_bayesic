import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
import torch
import seaborn as sns

def test_pyro_sampling():
    samples = pyro.sample("my_samples", dist.Normal(0, 1), sample_shape=(100,))
    sns.distplot(samples)
    plt.title("Samples from a standard Normal")
    plt.show()


def test_pyro_model():
     # Define the model
    def model(data):
        mu = pyro.sample("mu", dist.Normal(0, 1))  # Prior
        return pyro.sample("obs", dist.Normal(mu, 1), obs=data)  # Likelihood

    # Generate some synthetic data
    data = torch.tensor([0.5, -0.2, 0.3])

    # Perform inference using MCMC
    from pyro.infer import MCMC, NUTS

    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
    mcmc.run(data)

    # Extract posterior samples
    posterior_samples = mcmc.get_samples()["mu"].numpy()

    # Plot posterior
    plt.hist(posterior_samples, bins=30, density=True)
    plt.title("Posterior distribution of mu")
    plt.xlabel("mu")
    plt.ylabel("Density")
    plt.show()