.. index:: fix atom/swap

fix atom/swap command
=====================

Syntax
""""""

.. parsed-literal::

   fix ID group-ID atom/swap N X seed T keyword values ...

* ID, group-ID are documented in :doc:`fix <fix>` command
* atom/swap = style name of this fix command
* N = invoke this fix every N steps
* X = number of swaps to attempt every N steps
* seed = random # seed (positive integer)
* T = scaling temperature of the MC swaps (temperature units)
* one or more keyword/value pairs may be appended to args
* keyword = *types* or *mu* or *ke* or *semi-grand* or *region*

  .. parsed-literal::

       *types* values = two or more atom types
       *mu* values = chemical potential of swap types (energy units)
       *ke* value = *no* or *yes*
         *no* = no conservation of kinetic energy after atom swaps
         *yes* = kinetic energy is conserved after atom swaps
       *semi-grand* value = *no* or *yes*
         *no* = particle type counts and fractions conserved
         *yes* = semi-grand canonical ensemble, particle fractions not conserved
       *region* value = region-ID
         region-ID = ID of region to use as an exchange/move volume

Examples
""""""""

.. code-block:: LAMMPS

   fix 2 all atom/swap 1 1 29494 300.0 ke no types 1 2
   fix myFix all atom/swap 100 1 12345 298.0 region my_swap_region types 5 6
   fix SGMC all atom/swap 1 100 345 1.0 semi-grand yes types 1 2 3 mu 0.0 4.3 -5.0

Description
"""""""""""

This fix performs Monte Carlo swaps of atoms of one given atom type
with atoms of the other given atom types. The specified T is used in
the Metropolis criterion dictating swap probabilities.

Perform X swaps of atoms of one type with atoms of another type
according to a Monte Carlo probability. Swap candidates must be in the
fix group, must be in the region (if specified), and must be of one of
the listed types. Swaps are attempted between candidates that are
chosen randomly with equal probability among the candidate
atoms. Swaps are not attempted between atoms of the same type since
nothing would happen.

All atoms in the simulation domain can be moved using regular time
integration displacements, e.g. via :doc:`fix nvt <fix_nh>`, resulting
in a hybrid MC+MD simulation. A smaller-than-usual timestep size may
be needed when running such a hybrid simulation, especially if the
swapped atoms are not well equilibrated.

The *types* keyword is required. At least two atom types must be
specified.

The *ke* keyword can be set to *no* to turn off kinetic energy
conservation for swaps. The default is *yes*, which means that swapped
atoms have their velocities scaled by the ratio of the masses of the
swapped atom types. This ensures that the kinetic energy of each atom
is the same after the swap as it was before the swap, even though the
atom masses have changed.

The *semi-grand* keyword can be set to *yes* to switch to the
semi-grand canonical ensemble as discussed in :ref:`(Sadigh) <Sadigh>`. This
means that the total number of each particle type does not need to be
conserved. The default is *no*, which means that the only kind of swap
allowed exchanges an atom of one type with an atom of a different
given type. In other words, the relative mole fractions of the swapped
atoms remains constant. Whereas in the semi-grand canonical ensemble,
the composition of the system can change. Note that when using
*semi-grand*, atoms in the fix group whose type is not listed
in the *types* keyword are ineligible for attempted
conversion. An attempt is made to switch
the selected atom (if eligible) to one of the other listed types
with equal probability. Acceptance of each attempt depends upon the Metropolis criterion.

The *mu* keyword allows users to specify chemical
potentials. This is required and allowed only when using *semi-grand*\ .
All chemical potentials are absolute, so there is one for
each swap type listed following the *types* keyword.
In semi-grand canonical ensemble simulations the chemical composition
of the system is controlled by the difference in these values. So
shifting all values by a constant amount will have no effect
on the simulation.

This command may optionally use the *region* keyword to define swap
volume.  The specified region must have been previously defined with a
:doc:`region <region>` command.  It must be defined with side = *in*\ .
Swap attempts occur only between atoms that are both within the
specified region. Swaps are not otherwise attempted.

You should ensure you do not swap atoms belonging to a molecule, or
LAMMPS will soon generate an error when it tries to find those atoms.
LAMMPS will warn you if any of the atoms eligible for swapping have a
non-zero molecule ID, but does not check for this at the time of
swapping.

If not using *semi-grand* this fix checks to ensure all atoms of the
given types have the same atomic charge. LAMMPS does not enforce this
in general, but it is needed for this fix to simplify the
swapping procedure. Successful swaps will swap the atom type and charge
of the swapped atoms. Conversely, when using *semi-grand*, it is assumed that all the atom
types involved in switches have the same charge. Otherwise, charge
would not be conserved. As a consequence, no checks on atomic charges are
performed, and successful switches update the atom type but not the
atom charge. While it is possible to use *semi-grand* with groups of
atoms that have different charges, these charges will not be changed when the
atom types change.

Since this fix computes total potential energies before and after
proposed swaps, so even complicated potential energy calculations are
OK, including the following:

* long-range electrostatics (kspace)
* many body pair styles
* hybrid pair styles
* eam pair styles
* triclinic systems
* need to include potential energy contributions from other fixes

Some fixes have an associated potential energy. Examples of such fixes
include: :doc:`efield <fix_efield>`, :doc:`gravity <fix_gravity>`,
:doc:`addforce <fix_addforce>`, :doc:`langevin <fix_langevin>`,
:doc:`restrain <fix_restrain>`, :doc:`temp/berendsen <fix_temp_berendsen>`,
:doc:`temp/rescale <fix_temp_rescale>`, and :doc:`wall fixes <fix_wall>`.
For that energy to be included in the total potential energy of the
system (the quantity used when performing GCMC moves),
you MUST enable the :doc:`fix_modify <fix_modify>` *energy* option for
that fix.  The doc pages for individual :doc:`fix <fix>` commands
specify if this should be done.

Restart, fix_modify, output, run start/stop, minimize info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

This fix writes the state of the fix to :doc:`binary restart files <restart>`.  This includes information about the random
number generator seed, the next timestep for MC exchanges, the number
of exchange attempts and successes etc.  See
the :doc:`read_restart <read_restart>` command for info on how to
re-specify a fix in an input script that reads a restart file, so that
the operation of the fix continues in an uninterrupted fashion.

.. note::

   For this to work correctly, the timestep must **not** be changed
   after reading the restart with :doc:`reset_timestep <reset_timestep>`.
   The fix will try to detect it and stop with an error.

None of the :doc:`fix_modify <fix_modify>` options are relevant to this
fix.

This fix computes a global vector of length 2, which can be accessed
by various :doc:`output commands <Howto_output>`.  The vector values are
the following global cumulative quantities:

* 1 = swap attempts
* 2 = swap successes

The vector values calculated by this fix are "extensive".

No parameter of this fix can be used with the *start/stop* keywords of
the :doc:`run <run>` command.  This fix is not invoked during :doc:`energy minimization <minimize>`.

Restrictions
""""""""""""

This fix is part of the MC package.  It is only enabled if LAMMPS was
built with that package.  See the :doc:`Build package <Build_package>`
doc page for more info.

Related commands
""""""""""""""""

:doc:`fix nvt <fix_nh>`, :doc:`neighbor <neighbor>`,
:doc:`fix deposit <fix_deposit>`, :doc:`fix evaporate <fix_evaporate>`,
:doc:`delete_atoms <delete_atoms>`, :doc:`fix gcmc <fix_gcmc>`

Default
"""""""

The option defaults are ke = yes, semi-grand = no, mu = 0.0 for
all atom types.

----------

.. _Sadigh:

**(Sadigh)** B Sadigh, P Erhart, A Stukowski, A Caro, E Martinez, and
L Zepeda-Ruiz, Phys. Rev. B, 85, 184203 (2012).
