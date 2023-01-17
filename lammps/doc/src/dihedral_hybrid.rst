.. index:: dihedral_style hybrid

dihedral_style hybrid command
=============================

Syntax
""""""

.. code-block:: LAMMPS

   dihedral_style hybrid style1 style2 ...

* style1,style2 = list of one or more dihedral styles

Examples
""""""""

.. code-block:: LAMMPS

   dihedral_style hybrid harmonic helix
   dihedral_coeff 1 harmonic 6.0 1 3
   dihedral_coeff 2* helix 10 10 10

Description
"""""""""""

The *hybrid* style enables the use of multiple dihedral styles in one
simulation.  An dihedral style is assigned to each dihedral type.  For
example, dihedrals in a polymer flow (of dihedral type 1) could be
computed with a *harmonic* potential and dihedrals in the wall
boundary (of dihedral type 2) could be computed with a *helix*
potential.  The assignment of dihedral type to style is made via the
:doc:`dihedral_coeff <dihedral_coeff>` command or in the data file.

In the dihedral_coeff commands, the name of a dihedral style must be
added after the dihedral type, with the remaining coefficients being
those appropriate to that style.  In the example above, the 2
dihedral_coeff commands set dihedrals of dihedral type 1 to be
computed with a *harmonic* potential with coefficients 6.0, 1, 3 for
K, d, n.  All other dihedral types (2-N) are computed with a *helix*
potential with coefficients 10, 10, 10 for A, B, C.

If dihedral coefficients are specified in the data file read via the
:doc:`read_data <read_data>` command, then the same rule applies.
E.g. "harmonic" or "helix", must be added after the dihedral type, for
each line in the "Dihedral Coeffs" section, e.g.

.. parsed-literal::

   Dihedral Coeffs

   1 harmonic 6.0 1 3
   2 helix 10 10 10
   ...

If *class2* is one of the dihedral hybrid styles, the same rule holds
for specifying additional AngleTorsion (and EndBondTorsion, etc)
coefficients either via the input script or in the data file.
I.e. *class2* must be added to each line after the dihedral type.  For
lines in the AngleTorsion (or EndBondTorsion, etc) Coeffs section of the data
file for dihedral types that are not *class2*, you must use an
dihedral style of *skip* as a placeholder, e.g.

.. parsed-literal::

   AngleTorsion Coeffs

   1 skip
   2 class2 1.0 1.0 1.0 3.0 3.0 3.0 30.0 50.0
   ...

Note that it is not necessary to use the dihedral style *skip* in the
input script, since AngleTorsion (or EndBondTorsion, etc) coefficients
need not be specified at all for dihedral types that are not *class2*\ .

A dihedral style of *none* with no additional coefficients can be used
in place of a dihedral style, either in a input script dihedral_coeff
command or in the data file, if you desire to turn off interactions
for specific dihedral types.

----------

Restrictions
""""""""""""

This dihedral style can only be used if LAMMPS was built with the
MOLECULE package.  See the :doc:`Build package <Build_package>` doc page
for more info.

Unlike other dihedral styles, the hybrid dihedral style does not store
dihedral coefficient info for individual sub-styles in a :doc:`binary restart files <restart>`.  Thus when restarting a simulation from a
restart file, you need to re-specify dihedral_coeff commands.

Related commands
""""""""""""""""

:doc:`dihedral_coeff <dihedral_coeff>`

Default
"""""""

none
