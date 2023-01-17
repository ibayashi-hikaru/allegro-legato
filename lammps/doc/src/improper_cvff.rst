.. index:: improper_style cvff
.. index:: improper_style cvff/intel
.. index:: improper_style cvff/omp

improper_style cvff command
===========================

Accelerator Variants: *cvff/intel*, *cvff/omp*

Syntax
""""""

.. code-block:: LAMMPS

   improper_style cvff

Examples
""""""""

.. code-block:: LAMMPS

   improper_style cvff
   improper_coeff 1 80.0 -1 4

Description
"""""""""""

The *cvff* improper style uses the potential

.. math::

   E = K [1 + d  \cos (n \phi) ]

where phi is the improper dihedral angle.

If the 4 atoms in an improper quadruplet (listed in the data file read
by the :doc:`read_data <read_data>` command) are ordered I,J,K,L then
the improper dihedral angle is between the plane of I,J,K and the
plane of J,K,L.  Note that because this is effectively a dihedral
angle, the formula for this improper style is the same as for
:doc:`dihedral_style harmonic <dihedral_harmonic>`.

Note that defining 4 atoms to interact in this way, does not mean that
bonds necessarily exist between I-J, J-K, or K-L, as they would in a
linear dihedral.  Normally, the bonds I-J, I-K, I-L would exist for an
improper to be defined between the 4 atoms.

The following coefficients must be defined for each improper type via
the :doc:`improper_coeff <improper_coeff>` command as in the example
above, or in the data file or restart files read by the
:doc:`read_data <read_data>` or :doc:`read_restart <read_restart>`
commands:

* :math:`K` (energy)
* :math:`d` (+1 or -1)
* :math:`n` (0,1,2,3,4,6)

----------

.. include:: accel_styles.rst

----------

Restrictions
""""""""""""

This improper style can only be used if LAMMPS was built with the
MOLECULE package.  See the :doc:`Build package <Build_package>` doc page
for more info.

Related commands
""""""""""""""""

:doc:`improper_coeff <improper_coeff>`

Default
"""""""

none
