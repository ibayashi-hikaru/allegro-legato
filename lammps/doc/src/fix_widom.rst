.. index:: fix widom

fix widom command
=================

Syntax
""""""

.. parsed-literal::

   fix ID group-ID widom N M type seed T keyword values ...

* ID, group-ID are documented in :doc:`fix <fix>` command
* widom = style name of this fix command
* N = invoke this fix every N steps
* M = number of Widom insertions to attempt every N steps
* type = atom type for inserted atoms (must be 0 if mol keyword used)
* seed = random # seed (positive integer)
* T = temperature of the system (temperature units)
* zero or more keyword/value pairs may be appended to args

  .. parsed-literal::

     keyword = *mol*, *region*, *full_energy*, *charge*, *intra_energy*
       *mol* value = template-ID
         template-ID = ID of molecule template specified in a separate :doc:`molecule <molecule>` command
       *region* value = region-ID
         region-ID = ID of region where Widom insertions are allowed
       *full_energy* = compute the entire system energy when performing Widom insertions
       *charge* value = charge of inserted atoms (charge units)
       *intra_energy* value = intramolecular energy (energy units)

Examples
""""""""

.. code-block:: LAMMPS

   fix 2 gas widom 1 50000 1 19494 2.0
   fix 3 water widom 1000 100 0 29494 300.0 mol h2omol full_energy

Description
"""""""""""

This fix performs Widom insertions of atoms or molecules at the given
temperature as discussed in :ref:`(Frenkel) <Frenkel1>`. Specific uses
include computation of Henry constants of small molecules in microporous
materials or amorphous systems.


Every N timesteps the fix attempts M number of Widom insertions of atoms
or molecules.

If the *mol* keyword is used, only molecule insertions are performed.
Conversely, if the *mol* keyword is not used, only atom insertions are
performed.

This command may optionally use the *region* keyword to define an
insertion volume.  The specified region must have been previously
defined with a :doc:`region <region>` command.  It must be defined with
side = *in*\ .  Insertion attempts occur only within the specified
region. For non-rectangular regions, random trial points are generated
within the rectangular bounding box until a point is found that lies
inside the region. If no valid point is generated after 1000 trials, no
insertion is performed. If an attempted insertion places the atom or
molecule center-of-mass outside the specified region, a new attempted
insertion is generated. This process is repeated until the atom or
molecule center-of-mass is inside the specified region.

Note that neighbor lists are re-built every timestep that this fix is
invoked, so you should not set N to be too small. See the :doc:`neighbor
<neighbor>` command for details.

When an atom or molecule is to be inserted, its coordinates are chosen
at a random position within the current simulation cell or region.
Relative coordinates for atoms in a molecule are taken from the
template molecule provided by the user. The center of mass of the
molecule is placed at the insertion point. The orientation of the
molecule is chosen at random by rotating about this point.

Individual atoms are inserted, unless the *mol* keyword is used.  It
specifies a *template-ID* previously defined using the :doc:`molecule
<molecule>` command, which reads a file that defines the molecule.  The
coordinates, atom types, charges, etc., as well as any bonding and
special neighbor information for the molecule can be specified in the
molecule file.  See the :doc:`molecule <molecule>` command for details.
The only settings required to be in this file are the coordinates and
types of atoms in the molecule.

Note that fix widom does not use configurational bias MC or any other
kind of sampling of intramolecular degrees of freedom.  Inserted
molecules can have different orientations, but they will all have the
same intramolecular configuration, which was specified in the molecule
command input.

For atoms, inserted particles have the specified atom type.  For
molecules, they use the same atom types as in the template molecule
supplied by the user.

The excess chemical potential mu_ex is defined as:

.. math::

   \mu_{ex} = -kT \ln(<\exp(-(U_{N+1}-U_{N})/{kT})>)

where *k* is Boltzman's constant, *T* is the user-specified temperature,
U_N and U_{N+1} is the potential energy of the system with N and N+1
particles.

The *full_energy* option means that the fix calculates the total
potential energy of the entire simulated system, instead of just the
energy of the part that is changed. By default, this option is off, in
which case only partial energies are computed to determine the energy
difference due to the proposed change.

The *full_energy* option is needed for systems with complicated
potential energy calculations, including the following:

* long-range electrostatics (kspace)
* many-body pair styles
* hybrid pair styles
* eam pair styles
* tail corrections
* need to include potential energy contributions from other fixes

In these cases, LAMMPS will automatically apply the *full_energy*
keyword and issue a warning message.

When the *mol* keyword is used, the *full_energy* option also includes
the intramolecular energy of inserted and deleted molecules, whereas
this energy is not included when *full_energy* is not used. If this is
not desired, the *intra_energy* keyword can be used to define an amount
of energy that is subtracted from the final energy when a molecule is
inserted, and subtracted from the initial energy when a molecule is
deleted. For molecules that have a non-zero intramolecular energy, this
will ensure roughly the same behavior whether or not the *full_energy*
option is used.

Some fixes have an associated potential energy. Examples of such fixes
include: :doc:`efield <fix_efield>`, :doc:`gravity <fix_gravity>`,
:doc:`addforce <fix_addforce>`, :doc:`restrain <fix_restrain>`, and
:doc:`wall fixes <fix_wall>`.  For that energy to be included in the
total potential energy of the system (the quantity used when performing
Widom insertions), you MUST enable the :doc:`fix_modify <fix_modify>`
*energy* option for that fix.  The doc pages for individual :doc:`fix
<fix>` commands specify if this should be done.

Use the *charge* option to insert atoms with a user-specified point
charge. Note that doing so will cause the system to become non-neutral.
LAMMPS issues a warning when using long-range electrostatics (kspace)
with non-neutral systems. See the :doc:`compute group/group
<compute_group_group>` documentation for more details about simulating
non-neutral systems with kspace on.

Restart, fix_modify, output, run start/stop, minimize info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

This fix writes the state of the fix to :doc:`binary restart files
<restart>`.  This includes information about the random number
generator seed, the next timestep for Widom insertions etc.  See the
:doc:`read_restart <read_restart>` command for info on how to
re-specify a fix in an input script that reads a restart file, so that
the operation of the fix continues in an uninterrupted fashion.

.. note::

   For this to work correctly, the timestep must **not** be changed
   after reading the restart with :doc:`reset_timestep
   <reset_timestep>`.  The fix will try to detect it and stop with an
   error.

None of the :doc:`fix_modify <fix_modify>` options are relevant to this
fix.

This fix computes a global vector of length 3, which can be accessed by
various :doc:`output commands <Howto_output>`.  The vector values are
the following global cumulative quantities:

* 1 = average excess chemical potential on each timestep
* 2 = average difference in potential energy on each timestep
* 3 = volume of the insertion region

The vector values calculated by this fix are "extensive".

No parameter of this fix can be used with the *start/stop* keywords of
the :doc:`run <run>` command.  This fix is not invoked during
:doc:`energy minimization <minimize>`.

Restrictions
""""""""""""

This fix is part of the MC package.  It is only enabled if LAMMPS was
built with that package.  See the :doc:`Build package <Build_package>`
doc page for more info.

Do not set "neigh_modify once yes" or else this fix will never be
called.  Reneighboring is **required**.

Can be run in parallel, but aspects of the GCMC part will not scale well
in parallel. Only usable for 3D simulations.


Related commands
""""""""""""""""

:doc:`fix gcmc <fix_gcmc>`
:doc:`fix atom/swap <fix_atom_swap>`,
:doc:`neighbor <neighbor>`,
:doc:`fix deposit <fix_deposit>`, :doc:`fix evaporate <fix_evaporate>`,


Default
"""""""

The option defaults are mol = no, intra_energy = 0.0 and full_energy =
no, except for the situations where full_energy is required, as listed
above.

----------

.. _Frenkel1:

**(Frenkel)** Frenkel and Smit, Understanding Molecular Simulation,
Academic Press, London, 2002.
