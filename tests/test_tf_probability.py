import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import seaborn as sns


def test_sample_from_normal():
    normal = tfp.distributions.Normal(loc=0., scale=1.)
    samples = normal.sample(1000)
    sns.distplot(samples)
    plt.title("Samples from a standard Normal")
    plt.show()
