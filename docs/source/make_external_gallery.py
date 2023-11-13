"""
Modified after https://github.com/pyvista/pyvista/blob/ab70c26edbcfb107286c827bd4914562056219fb/docs/make_external_gallery.py

A helper script to generate the external examples gallery.
"""
import os
from io import StringIO


def format_icon(title, description, link, image):
    body = r"""
   .. grid-item-card:: {}
      :link: {}
      :text-align: center
      :class-title: pyvista-card-title

      .. image:: {}
"""
    content = body.format(title, link, image)
    return content


class Example():
    def __init__(self, title, description, link, image):
        self.title = title
        self.description = description
        self.link = link
        self.image = image

    def format(self):
        return format_icon(self.title, self.description, self.link, self.image)


###############################################################################

articles = dict(
    gempy_well=Example(
        title="GemPy - Subsurface Link",
        description="Build a model from Subsurface object and export result back to subsurface",
        link="https://docs.gempy.org/integrations/gempy_subsurface.html#sphx-glr-integrations-gempy-subsurface-py",
        image="https://docs.gempy.org/_images/sphx_glr_gempy_subsurface_002.png",
    ),
    segysag=Example(
        title="Using segysak with subsurface",
        description="Loading a segy cube into `subsurface.StructuredData`.",
        link="https://segysak.readthedocs.io/en/latest/examples/example_subsurface.html",
        image="https://raw.githubusercontent.com/trhallam/segysak/main/docs/_static/logo_small.png",
    ),
)


###############################################################################

def make_example_gallery():
    """Make the example gallery."""
    path = "./external/external_examples.rst"

    with StringIO() as new_fid:
        new_fid.write(
            """.. _external_examples:

External Examples
=================

Here are a list of longer, more technical examples of what PyVista can do.

.. caution::

    Please note that these examples link to external websites. If any of these
    links are broken, please raise an `issue
    <https://github.com/pyvista/pyvista/issues>`_.


Do you have a technical processing workflow or visualization routine you would
like to share? If so, please consider sharing your work here submitting a PR
at `pyvista/pyvista <https://github.com/pyvista/pyvista/>`_ and we would be
glad to add it.


.. grid:: 3
   :gutter: 1

"""
        )
        # Reverse to put the latest items at the top
        for example in list(articles.values())[::-1]:
            new_fid.write(example.format())

        new_fid.write(
            """

.. raw:: html

    <div class="sphx-glr-clear"></div>


"""
        )
        new_fid.seek(0)
        new_text = new_fid.read()

    # check if it's necessary to overwrite the table
    existing = ""
    if os.path.exists(path):
        with open(path) as existing_fid:
            existing = existing_fid.read()

    # write if different or does not exist
    if new_text != existing:
        with open(path, "w", encoding="utf-8") as fid:
            fid.write(new_text)

    return
