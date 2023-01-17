.. index:: fix smd/integrate_tlsph

fix smd/integrate_tlsph command
===============================

Syntax
""""""

.. code-block:: LAMMPS

   fix ID group-ID smd/integrate_tlsph keyword values

* ID, group-ID are documented in :doc:`fix <fix>` command
* smd/integrate_tlsph = style name of this fix command
* zero or more keyword/value pairs may be appended
* keyword = *limit_velocity*

.. parsed-literal::

     *limit_velocity* value = max_vel
       max_vel = maximum allowed velocity

Examples
""""""""

.. code-block:: LAMMPS

   fix 1 all smd/integrate_tlsph
   fix 1 all smd/integrate_tlsph limit_velocity 1000

Description
"""""""""""

The fix performs explicit time integration for particles which
interact according with the Total-Lagrangian SPH pair style.

See `this PDF guide <PDF/SMD_LAMMPS_userguide.pdf>`_ to using Smooth Mach
Dynamics in LAMMPS.

The *limit_velocity* keyword will control the velocity, scaling the
norm of the velocity vector to max_vel in case it exceeds this
velocity limit.

Restart, fix_modify, output, run start/stop, minimize info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Currently, no part of MACHDYN supports restarting nor
minimization. This fix has no outputs.

Restrictions
""""""""""""

This fix is part of the MACHDYN package.  It is only enabled if
LAMMPS was built with that package.  See the :doc:`Build package <Build_package>` page for more info.

Related commands
""""""""""""""""

:doc:`smd/integrate_ulsph <fix_smd_integrate_ulsph>`

Default
"""""""

none
