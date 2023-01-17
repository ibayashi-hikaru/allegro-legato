.. index:: pair_style extep

pair_style extep command
========================

Syntax
""""""

.. code-block:: LAMMPS

   pair_style extep

Examples
""""""""

.. code-block:: LAMMPS

   pair_style extep
   pair_coeff * * BN.extep B N

Description
"""""""""""

Style *extep* computes the Extended Tersoff Potential (ExTeP)
interactions as described in :ref:`(Los2017) <Los2017>`.

----------

Restrictions
""""""""""""
none

Related commands
""""""""""""""""

`pair_tersoff <pair tersoff>`

Default
"""""""

none

----------

.. _Los2017:

**(Los2017)** J. H. Los et al. "Extended Tersoff potential for boron nitride:
Energetics and elastic properties of pristine and defective h-BN",
Phys. Rev. B 96 (184108), 2017.
