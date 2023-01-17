.. index:: pair_style smd/hertz

pair_style smd/hertz command
============================

Syntax
""""""

.. code-block:: LAMMPS

   pair_style smd/hertz scale_factor

Examples
""""""""

.. code-block:: LAMMPS

   pair_style smd/hertz 1.0
   pair_coeff 1 1 <contact_stiffness>

Description
"""""""""""

The *smd/hertz* style calculates contact forces between SPH particles
belonging to different physical bodies.

The contact forces are calculated using a Hertz potential, which
evaluates the overlap between two particles (whose spatial extents are
defined via its contact radius).  The effect is that a particles
cannot penetrate into each other.  The parameter <contact_stiffness>
has units of pressure and should equal roughly one half of the Young's
modulus (or bulk modulus in the case of fluids) of the material model
associated with the SPH particles.

The parameter *scale_factor* can be used to scale the particles'
contact radii. This can be useful to control how close particles can
approach each other. Usually, *scale_factor* =1.0.

----------

Mixing, shift, table, tail correction, restart, rRESPA info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

No mixing is performed automatically.  Currently, no part of MACHDYN
supports restarting nor minimization.  rRESPA does not apply to this
pair style.

----------

Restrictions
""""""""""""

This fix is part of the MACHDYN package.  It is only enabled if
LAMMPS was built with that package.  See the :doc:`Build package <Build_package>` page for more info.

Related commands
""""""""""""""""

:doc:`pair_coeff <pair_coeff>`

Default
"""""""

none
