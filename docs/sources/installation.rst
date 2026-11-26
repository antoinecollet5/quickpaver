.. _installation_ref:

=================
Installation
=================

Stable release
--------------

To install quickpaver, run this command in your terminal:

.. code-block:: console

    $ pip install quickpaver

This is the preferred method to install quickpaver, as it will always install the most recent stable release.
Prebuilt wheels are available for CPython 3.10 to 3.15 (including the free-threaded
builds) on Linux, macOS and Windows: no compiler is needed.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for quickpaver can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/antoinecollet5/quickpaver

Building from source requires a `Rust toolchain <https://rustup.rs>`_ (the compiled
extension is built by `maturin <https://www.maturin.rs>`_). Once you have a copy of the
source, you can install it with:

.. code-block:: console

    $ pip install -e .

or, for development, with `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: console

    $ uv sync


Tutorials
---------

For a quick overview of quickpaver functionality, see the :ref:`user guide<user_guide_ref>`.

You can also refer to the :ref:`reference guide<api_ref>` for an exhaustive
list of all what is possible with quickpaver.

.. _Github repo: https://github.com/antoinecollet5/quickpaver
.. _tarball: https://github.com/antoinecollet5/quickpaver/tarball/master
