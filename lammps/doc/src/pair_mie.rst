.. index:: pair_style mie/cut
.. index:: pair_style mie/cut/gpu

pair_style mie/cut command
==========================

Accelerator Variants: *mie/cut/gpu*

Syntax
""""""

.. code-block:: LAMMPS

   pair_style mie/cut cutoff

* cutoff = global cutoff for mie/cut interactions (distance units)

Examples
""""""""

.. code-block:: LAMMPS

   pair_style mie/cut 10.0
   pair_coeff 1 1 0.72 3.40 23.00 6.66
   pair_coeff 2 2 0.30 3.55 12.65 6.00
   pair_coeff 1 2 0.46 3.32 16.90 6.31

Description
"""""""""""

The *mie/cut* style computes the Mie potential, given by

.. math::

   E =  C \epsilon \left[ \left(\frac{\sigma}{r}\right)^{\gamma_{rep}} - \left(\frac{\sigma}{r}\right)^{\gamma_{att}} \right]
                         \qquad r < r_c

Rc is the cutoff and C is a function that depends on the repulsive and
attractive exponents, given by:

.. math::

   C = \left(\frac{\gamma_{rep}}{\gamma_{rep}-\gamma_{att}}\right) \left(\frac{\gamma_{rep}}{\gamma_{att}}\right)^{\left(\frac{\gamma_{att}}{\gamma_{rep}-\gamma_{att}}\right)}

Note that for 12/6 exponents, C is equal to 4 and the formula is the
same as the standard Lennard-Jones potential.

The following coefficients must be defined for each pair of atoms
types via the :doc:`pair_coeff <pair_coeff>` command as in the examples
above, or in the data file or restart files read by the
:doc:`read_data <read_data>` or :doc:`read_restart <read_restart>`
commands, or by mixing as described below:

* epsilon (energy units)
* sigma (distance units)
* gammaR
* gammaA
* cutoff (distance units)

The last coefficient is optional.  If not specified, the global
cutoff specified in the pair_style command is used.

----------

Mixing, shift, table, tail correction, restart, rRESPA info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

For atom type pairs I,J and I != J, the epsilon and sigma coefficients
and cutoff distance for all of the mie/cut pair styles can be mixed.
If not explicitly defined, both the repulsive and attractive gamma
exponents for different atoms will be calculated following the same
mixing rule defined for distances.  The default mix value is
*geometric*\ . See the "pair_modify" command for details.

This pair style supports the :doc:`pair_modify <pair_modify>` shift
option for the energy of the pair interaction.

This pair style supports the :doc:`pair_modify <pair_modify>` tail
option for adding a long-range tail correction to the energy and
pressure of the pair interaction.

This pair style writes its information to :doc:`binary restart files <restart>`, so pair_style and pair_coeff commands do not need
to be specified in an input script that reads a restart file.

This pair style supports the use of the *inner*, *middle*, and *outer*
keywords of the :doc:`run_style respa <run_style>` command, meaning the
pairwise forces can be partitioned by distance at different levels of
the rRESPA hierarchy.  See the :doc:`run_style <run_style>` command for
details.

----------

Restrictions
""""""""""""

This pair style is part of the EXTRA-PAIR package.  It is only enabled if
LAMMPS was built with that package.  See the
:doc:`Build package <Build_package>` page for more info.

Related commands
""""""""""""""""

:doc:`pair_coeff <pair_coeff>`

Default
"""""""

none

----------

.. _Mie:

**(Mie)** G. Mie, Ann Phys, 316, 657 (1903).

.. _Avendano:

**(Avendano)** C. Avendano, T. Lafitte, A. Galindo, C. S. Adjiman,
G. Jackson, E. Muller, J Phys Chem B, 115, 11154 (2011).
