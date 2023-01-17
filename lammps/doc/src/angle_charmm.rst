.. index:: angle_style charmm
.. index:: angle_style charmm/intel
.. index:: angle_style charmm/kk
.. index:: angle_style charmm/omp

angle_style charmm command
==========================

Accelerator Variants: *charmm/intel*, *charmm/kk*, *charmm/omp*

Syntax
""""""

.. code-block:: LAMMPS

   angle_style charmm

Examples
""""""""

.. code-block:: LAMMPS

   angle_style charmm
   angle_coeff 1 300.0 107.0 50.0 3.0

Description
"""""""""""

The *charmm* angle style uses the potential

.. math::

   E = K (\theta - \theta_0)^2 + K_{ub} (r - r_{ub})^2

with an additional Urey_Bradley term based on the distance :math:`r` between
the first and third atoms in the angle.  :math:`K`, :math:`\theta_0`,
:math:`K_{ub}`, and :math:`R_{ub}` are coefficients defined for each angle
type.

See :ref:`(MacKerell) <angle-MacKerell>` for a description of the CHARMM force
field.

The following coefficients must be defined for each angle type via the
:doc:`angle_coeff <angle_coeff>` command as in the example above, or in
the data file or restart files read by the :doc:`read_data <read_data>`
or :doc:`read_restart <read_restart>` commands:

* :math:`K` (energy)
* :math:`\theta_0` (degrees)
* :math:`K_{ub}` (energy/distance\^2)
* :math:`r_{ub}` (distance)

:math:`\theta_0` is specified in degrees, but LAMMPS converts it to
radians internally; hence :math:`K` is effectively energy per
radian\^2.

----------

Styles with a *gpu*, *intel*, *kk*, *omp*, or *opt* suffix are
functionally the same as the corresponding style without the suffix.
They have been optimized to run faster, depending on your available
hardware, as discussed on the :doc:`Speed packages <Speed_packages>` doc
page.  The accelerated styles take the same arguments and should
produce the same results, except for round-off and precision issues.

These accelerated styles are part of the GPU, INTEL, KOKKOS,
OPENMP and OPT packages, respectively.  They are only enabled if
LAMMPS was built with those packages.  See the :doc:`Build package <Build_package>` page for more info.

You can specify the accelerated styles explicitly in your input script
by including their suffix, or you can use the :doc:`-suffix command-line switch <Run_options>` when you invoke LAMMPS, or you can use the
:doc:`suffix <suffix>` command in your input script.

See :doc:`Speed packages <Speed_packages>` page for more
instructions on how to use the accelerated styles effectively.

----------

Restrictions
""""""""""""

This angle style can only be used if LAMMPS was built with the
MOLECULE package.  See the :doc:`Build package <Build_package>` doc page
for more info.

Related commands
""""""""""""""""

:doc:`angle_coeff <angle_coeff>`

Default
"""""""

none

----------

.. _angle-MacKerell:

**(MacKerell)** MacKerell, Bashford, Bellott, Dunbrack, Evanseck, Field,
Fischer, Gao, Guo, Ha, et al, J Phys Chem, 102, 3586 (1998).
