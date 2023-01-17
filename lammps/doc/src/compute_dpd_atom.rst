.. index:: compute dpd/atom

compute dpd/atom command
========================

Syntax
""""""

.. parsed-literal::

   compute ID group-ID dpd/atom

* ID, group-ID are documented in :doc:`compute <compute>` command
* dpd/atom = style name of this compute command

Examples
""""""""

.. code-block:: LAMMPS

   compute 1 all dpd/atom

Description
"""""""""""

Define a computation that accesses the per-particle internal
conductive energy (:math:`u^{cond}`), internal mechanical
energy (:math:`u^{mech}`), internal chemical energy (:math:`u^{chem}`)
and internal temperatures (:math:`\theta`) for each particle in a group.
See the :doc:`compute dpd <compute_dpd>` command if you want the total
internal conductive energy, the total internal mechanical energy, the
total chemical energy and
average internal temperature of the entire system or group of dpd
particles.

Output info
"""""""""""

This compute calculates a per-particle array with 4 columns (:math:`u^{cond}`,
:math:`u^{mech}`, :math:`u^{chem}`, :math:`\theta`), which can be accessed
by indices 1-4 by any
command that uses per-particle values from a compute as input.  See
the :doc:`Howto output <Howto_output>` page for an overview of
LAMMPS output options.

The per-particle array values will be in energy (:math:`u^{cond}`,
:math:`u^{mech}`, :math:`u^{chem}`)
and temperature (:math:`theta`) :doc:`units <units>`.

Restrictions
""""""""""""

This command is part of the DPD-REACT package.  It is only enabled if
LAMMPS was built with that package.  See the :doc:`Build package <Build_package>` page for more info.

This command also requires use of the :doc:`atom_style dpd <atom_style>`
command.

Related commands
""""""""""""""""

:doc:`dump custom <dump>`, :doc:`compute dpd <compute_dpd>`

Default
"""""""

none

----------

.. _Larentzos2:

**(Larentzos)** J.P. Larentzos, J.K. Brennan, J.D. Moore, and
W.D. Mattson, "LAMMPS Implementation of Constant Energy Dissipative
Particle Dynamics (DPD-E)", ARL-TR-6863, U.S. Army Research
Laboratory, Aberdeen Proving Ground, MD (2014).
