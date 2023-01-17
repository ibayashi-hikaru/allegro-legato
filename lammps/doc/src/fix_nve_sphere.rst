.. index:: fix nve/sphere
.. index:: fix nve/sphere/omp
.. index:: fix nve/sphere/kk

fix nve/sphere command
======================

Accelerator Variants: *nve/sphere/omp*, *nve/sphere/kk*

Syntax
""""""

.. parsed-literal::

   fix ID group-ID nve/sphere

* ID, group-ID are documented in :doc:`fix <fix>` command
* nve/sphere = style name of this fix command
* zero or more keyword/value pairs may be appended
* keyword = *update* or *disc*

  .. parsed-literal::

       *update* value = *dipole* or *dipole/dlm*
         dipole = update orientation of dipole moment during integration
         dipole/dlm = use DLM integrator to update dipole orientation
       *disc* value = none = treat particles as 2d discs, not spheres

Examples
""""""""

.. code-block:: LAMMPS

   fix 1 all nve/sphere
   fix 1 all nve/sphere update dipole
   fix 1 all nve/sphere disc
   fix 1 all nve/sphere update dipole/dlm

Description
"""""""""""

Perform constant NVE integration to update position, velocity, and
angular velocity for finite-size spherical particles in the group each
timestep.  V is volume; E is energy.  This creates a system trajectory
consistent with the microcanonical ensemble.

This fix differs from the :doc:`fix nve <fix_nve>` command, which
assumes point particles and only updates their position and velocity.

If the *update* keyword is used with the *dipole* value, then the
orientation of the dipole moment of each particle is also updated
during the time integration.  This option should be used for models
where a dipole moment is assigned to finite-size particles,
e.g. spheroids via use of the :doc:`atom_style hybrid sphere dipole <atom_style>` command.

The default dipole orientation integrator can be changed to the
Dullweber-Leimkuhler-McLachlan integration scheme
:ref:`(Dullweber) <nh-Dullweber>` when using *update* with the value
*dipole/dlm*\ . This integrator is symplectic and time-reversible,
giving better energy conservation and allows slightly longer timesteps
at only a small additional computational cost.

If the *disc* keyword is used, then each particle is treated as a 2d
disc (circle) instead of as a sphere.  This is only possible for 2d
simulations, as defined by the :doc:`dimension <dimension>` keyword.
The only difference between discs and spheres in this context is their
moment of inertia, as used in the time integration.

----------

.. include:: accel_styles.rst

----------

Restart, fix_modify, output, run start/stop, minimize info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

No information about this fix is written to :doc:`binary restart files <restart>`.  None of the :doc:`fix_modify <fix_modify>` options
are relevant to this fix.  No global or per-atom quantities are stored
by this fix for access by various :doc:`output commands <Howto_output>`.
No parameter of this fix can be used with the *start/stop* keywords of
the :doc:`run <run>` command.  This fix is not invoked during :doc:`energy minimization <minimize>`.

Restrictions
""""""""""""

This fix requires that atoms store torque and angular velocity (omega)
and a radius as defined by the :doc:`atom_style sphere <atom_style>`
command.  If the *dipole* keyword is used, then they must also store a
dipole moment as defined by the :doc:`atom_style dipole <atom_style>`
command.

All particles in the group must be finite-size spheres.  They cannot
be point particles.

Use of the *disc* keyword is only allowed for 2d simulations, as
defined by the :doc:`dimension <dimension>` keyword.

Related commands
""""""""""""""""

:doc:`fix nve <fix_nve>`, :doc:`fix nve/asphere <fix_nve_asphere>`

Default
"""""""

none

----------

.. _nve-Dullweber:

**(Dullweber)** Dullweber, Leimkuhler and McLachlan, J Chem Phys, 107,
5840 (1997).
