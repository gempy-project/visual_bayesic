import tensorflow_probability as tfp

def test_sample_from_normal():
    normal = tfd.Normal(loc=0., scale=1.)
