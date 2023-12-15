Result description and some words about uninformative prior
-----------------------------------------------------------

.. _result-description-and-some-words-about-uninformative-prior:

The first thing that comes to mind by looking at the results is that the model choice has a clear impact on the final result although the more observations are included the more similar both models get indistinctively of the priors. Therefore, it is interesting to compare the two cases with just one observation (Figure :numref:`fig-models1`, above), curiously enough, the probabilistic model with the normal distribution underrepresents the model uncertainty—assuming that the simulations c) and d) are closer to the real data generating process—while the uniform clearly portrays too much variance. Once again, considering that the only window to the data generating model are indirect observation of a phenomenon, in regimes of low and/or noisy data points there is only so much information we can extract from the data. This is to say that there are not truly uninformative prior distributions—although there are theoretical limits such as maximum entropy—and therefore the best we can do is to use the knowledge of the practitioner as *first educated guess*.

Type of probabilistic models
````````````````````````````

.. _type-of-probabilistic-models:

.. figure:: ../_static/Model_type.png
   :align: center
   :width: 100%

   Example of computational graph expressing simple probabilistic model
   :name: fig-model-type

Depending on the type of phenomenon analyzed and the model complexity, two different categories of problems emerge: Wide data and Long data. Long data problems are characterized by a large number of repetitions of the same phenomenon, allowing all sorts of machine learning and big data approaches. Wide data, on the other hand, features few repetitions of several phenomena that rely on complex mathematical models to relate them [1]_. Structural geology is an unequivocal example of the second case due to the sheer size and scale it is aimed to be modeled. For this reason, a systematic way to include domain knowledge into the system via informative prior distributions becomes a powerful tool in the endeavor of integrating as much information as possible in a single mathematical model.

Priors as joint probability
```````````````````````````

.. _priors-as-joint-probability:

In low data regimes where domain knowledge plays such an essential role, it is crucial to include as much **coherent** information as possible. Simply giving the best estimate of the data generating function may work for simple systems where our brains are capable to relate available data and knowledge. Nevertheless, as complexity rises, a more systematic way of combining data and knowledge becomes fundamental. Models are in essence a tool to generate best guesses following mathematical axioms. This view of models as tools to combine different sources of data and knowledge to help to define the best guess of a latent data generating process may seem an unnecessary convoluted way to explain Bayesian statistics. However, in our opinion, this view helps to change the perspective of prior distributions—from terms of "belief" as a source of bias—to a more general perspective of using joint probability as a means to combine complex mathematical models and observations in a mathematically sound and transparent manner.

.. [1] The term comes from the shape of the database. If you imagine columns to be different phenomena—i.e., properties, including space and time—and rows as repetition of measurements, it will be wide data when the proportion of columns vs. rows is large, and it will be long data in the opposite case.

License
-------
The code in this case study is copyrighted by Miguel de la Varga and licensed under the new BSD (3-clause) license:

https://opensource.org/licenses/BSD-3-Clause

The text and figures in this case study are copyrighted by Miguel de la Varga and licensed under the CC BY-NC 4.0 license:

https://creativecommons.org/licenses/by-nc/4.0/
Make sure to replace the links with actual hyperlinks if you're using a platform that supports it (e.g., Markdown or HTML). Otherwise, the plain URLs work fine for plain text.
