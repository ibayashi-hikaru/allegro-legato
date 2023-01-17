.. index:: compute smd/ulsph/effm

compute smd/ulsph/effm command
==============================

Syntax
""""""

.. code-block:: LAMMPS

   compute ID group-ID smd/ulsph/effm

* ID, group-ID are documented in :doc:`compute <compute>` command
* smd/ulsph/effm = style name of this compute command

Examples
""""""""

.. code-block:: LAMMPS

   compute 1 all smd/ulsph/effm

Description
"""""""""""

Define a computation that outputs the effective shear modulus for
particles interacting via the updated Lagrangian SPH pair style.

See `this PDF guide <PDF/SMD_LAMMPS_userguide.pdf>`_ to using Smooth
Mach Dynamics in LAMMPS.

Output info
"""""""""""

This compute calculates a per-particle vector, which can be accessed
by any command that uses per-particle values from a compute as input.
See the :doc:`Howto output <Howto_output>` page for an overview of
LAMMPS output options.

The per-particle vector contains the current effective per atom shear
modulus as computed by the :doc:`pair smd/ulsph <pair_smd_ulsph>` pair
style.

Restrictions
""""""""""""

This compute is part of the MACHDYN package.  It is only enabled if
LAMMPS was built with that package. See the :doc:`Build package <Build_package>` page for more info. This compute can
only be used for particles which interact with the updated Lagrangian
SPH pair style.

Related commands
""""""""""""""""

:doc:`pair smd/ulsph <pair_smd_ulsph>`

Default
"""""""

none
