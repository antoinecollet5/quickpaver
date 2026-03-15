=======
quickpaver
=======

|License| |Stars| |Python| |PyPI| |Downloads| |Build Status| |Documentation Status| |Coverage| |Codacy| |Precommit: enabled| |Ruff| |ty| |DOI|

🐍 A python package providing a tiling/paving toolbox.

**The complete and up to date documentation can be found here**: https://quickpaver.readthedocs.io.

===============
🎯 Motivations
===============

Designing paving layouts (aka tiling) often requires repetitive geometric calculations and careful programming to ensure that patterns and dimensions are correct. Keeping track of tiles adjacency is not trivial either and performing these tasks manually can be time-consuming and prone to errors, especially when exploring multiple design configurations or working with complex layouts.

This package aims to simplify and automate the generation of paving layouts by providing a programmatic way to create and manipulate classic paving patterns.

The implementation relies on `Shapely <https://shapely.readthedocs.io/en/stable/>`_ for geometric operations and leverages the `GEOS <https://libgeos.org/>`_ engine’s vectorization capabilities. This allows many geometric computations to be performed efficiently, enabling the generation and processing of large numbers of paving elements with good performance. As a result, the package provides both flexibility and speed, making it suitable for practical design tasks as well as research and experimentation.

===============
🚀 Quick start
===============

To install `quickpaver`, the easiest way is through `pip`:

.. code-block::

    pip install quickpaver

Or alternatively using `conda`

.. code-block::

    conda install quickpaver

You might also clone the repository and install from source

.. code-block::

    pip install -e .

Once the installation is done, `quickpaver` is straighforward to use and proposes...

🏗️ Complete example with supporting paper coming Q1 2026.

===========
🔑 License
===========

This project is released under the **BSD 3-Clause License**.

Copyright (c) 2026, Antoine COLLET. All rights reserved.

For more details, see the `LICENSE <https://github.com/antoinecollet5/quickpaver/blob/master/LICENSE>`_ file included in this repository.

==============
⚠️ Disclaimer
==============

This software is provided "as is", without warranty of any kind, express or implied,
including but not limited to the warranties of merchantability, fitness for a particular purpose,
or non-infringement. In no event shall the authors or copyright holders be liable for
any claim, damages, or other liability, whether in an action of contract, tort,
or otherwise, arising from, out of, or in connection with the software or the use
or other dealings in the software.

By using this software, you agree to accept full responsibility for any consequences,
and you waive any claims against the authors or contributors.

==========
📧 Contact
==========

For questions, suggestions, or contributions, you can reach out via:

- Email: antoinecollet5@gmail.com
- GitHub: https://github.com/antoinecollet5/quickpaver

We welcome contributions!

=============
📚 References
=============

TODO

* Free software: SPDX-License-Identifier: BSD-3-Clause

.. |License| image:: https://img.shields.io/badge/License-BSD_3--Clause-blue.svg
    :target: https://github.com/antoinecollet5/quickpaver/blob/master/LICENSE

.. |Stars| image:: https://img.shields.io/github/stars/antoinecollet5/quickpaver.svg?style=social&label=Star&maxAge=2592000
    :target: https://github.com/antoinecollet5/quickpaver/stargazers
    :alt: Stars

.. |Python| image:: https://img.shields.io/pypi/pyversions/quickpaver.svg
    :target: https://pypi.org/pypi/quickpaver
    :alt: Python

.. |PyPI| image:: https://img.shields.io/pypi/v/quickpaver.svg
    :target: https://pypi.org/pypi/quickpaver
    :alt: PyPI

.. |Downloads| image:: https://static.pepy.tech/badge/quickpaver
    :target: https://pepy.tech/project/quickpaver
    :alt: Downoads

.. |Build Status| image:: https://github.com/antoinecollet5/quickpaver/actions/workflows/main.yml/badge.svg
    :target: https://github.com/antoinecollet5/quickpaver/actions/workflows/main.yml
    :alt: Build Status

.. |Documentation Status| image:: https://readthedocs.org/projects/quickpaver/badge/?version=latest
    :target: https://quickpaver.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |Coverage| image:: https://codecov.io/gh/antoinecollet5/quickpaver/graph/badge.svg?token=8lE90wylXL
    :target: https://codecov.io/gh/antoinecollet5/quickpaver
    :alt: Coverage

.. |Codacy| image:: https://app.codacy.com/project/badge/Grade/c41f65d98b824de394162520b0d8a17a
    :target: https://app.codacy.com/gh/antoinecollet5/quickpaver/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade
    :alt: codacy

.. |Precommit: enabled| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit
   :target: https://github.com/pre-commit/pre-commit

.. |Ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff
    :alt: Ruff

.. |ty| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json
    :target: https://github.com/astral-sh/ty
    :alt: Checked with ty

.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.18900358.svg
   :target: https://doi.org/10.5281/zenodo.18900358


.. image:: https://api.codacy.com/project/badge/Grade/00500e48dcae4314a24b0aea1b45d1f0
   :alt: Codacy Badge
   :target: https://app.codacy.com/gh/antoinecollet5/quickpaver?utm_source=github.com&utm_medium=referral&utm_content=antoinecollet5/quickpaver&utm_campaign=Badge_Grade