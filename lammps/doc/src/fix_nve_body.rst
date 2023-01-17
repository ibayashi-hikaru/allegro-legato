.. index:: fix nve/body

fix nve/body command
====================

Syntax
""""""

.. parsed-literal::

   fix ID group-ID nve/body

* ID, group-ID are documented in :doc:`fix <fix>` command
* nve/body = style name of this fix command

Examples
""""""""

.. code-block:: LAMMPS

   fix 1 all nve/body

Description
"""""""""""

Perform constant NVE integration to update position, velocity,
orientation, and angular velocity for body particles in the group each
timestep.  V is volume; E is energy.  This creates a system trajectory
consistent with the microcanonical ensemble.  See the :doc:`Howto body <Howto_body>` page for more details on using body
particles.

This fix differs from the :doc:`fix nve <fix_nve>` command, which
assumes point particles and only updates their position and velocity.

Restart, fix_modify, output, run start/stop, minimize info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

No information about this fix is written to :doc:`binary restart files <restart>`.  None of the :doc:`fix_modify <fix_modify>` options
are relevant to this fix.  No global or per-atom quantities are stored
by this fix for access by various :doc:`output commands <Howto_output>`.
No parameter of this fix can be used with the *start/stop* keywords of
the :doc:`run <run>` command.  This fix is not invoked during :doc:`energy minimization <minimize>`.

Restrictions
""""""""""""

This fix is part of the BODY package.  It is only enabled if LAMMPS
was built with that package.  See the :doc:`Build package <Build_package>` page for more info.

This fix requires that atoms store torque and angular momentum and a
quaternion as defined by the :doc:`atom_style body <atom_style>`
command.

All particles in the group must be body particles.  They cannot be
point particles.

Related commands
""""""""""""""""

:doc:`fix nve <fix_nve>`, :doc:`fix nve/sphere <fix_nve_sphere>`, :doc:`fix nve/asphere <fix_nve_asphere>`

Default
"""""""

none
