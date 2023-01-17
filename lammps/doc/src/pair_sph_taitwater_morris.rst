.. index:: pair_style sph/taitwater/morris

pair_style sph/taitwater/morris command
=======================================

Syntax
""""""

.. code-block:: LAMMPS

   pair_style sph/taitwater/morris

Examples
""""""""

.. code-block:: LAMMPS

   pair_style sph/taitwater/morris
   pair_coeff * * 1000.0 1430.0 1.0 2.4

Description
"""""""""""

The sph/taitwater/morris style computes pressure forces between SPH
particles according to Tait's equation of state:

.. math::

   p = B \biggl[\left(\frac{\rho}{\rho_0}\right)^{\gamma} - 1\biggr]

where :math:`\gamma = 7` and :math:`B = c_0^2 \rho_0 / \gamma`, with
:math:`\rho_0` being the reference density and :math:`c_0` the reference
speed of sound.

This pair style also computes laminar viscosity :ref:`(Morris) <Morris>`.

See `this PDF guide <PDF/SPH_LAMMPS_userguide.pdf>`_ to using SPH in
LAMMPS.

The following coefficients must be defined for each pair of atoms
types via the :doc:`pair_coeff <pair_coeff>` command as in the examples
above.

* :math:`\rho_0` reference density (mass/volume units)
* :math:`c_0` reference soundspeed (distance/time units)
* :math:`\nu` dynamic viscosity (mass\*distance/time units)
* h kernel function cutoff (distance units)

----------

Mixing, shift, table, tail correction, restart, rRESPA info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

This style does not support mixing.  Thus, coefficients for all
I,J pairs must be specified explicitly.

This style does not support the :doc:`pair_modify <pair_modify>`
shift, table, and tail options.

This style does not write information to :doc:`binary restart files <restart>`.  Thus, you need to re-specify the pair_style and
pair_coeff commands in an input script that reads a restart file.

This style can only be used via the *pair* keyword of the :doc:`run_style respa <run_style>` command.  It does not support the *inner*,
*middle*, *outer* keywords.

Restrictions
""""""""""""

This pair style is part of the SPH package.  It is only enabled
if LAMMPS was built with that package.  See the :doc:`Build package <Build_package>` page for more info.

Related commands
""""""""""""""""

:doc:`pair_coeff <pair_coeff>`, pair_sph/rhosum

Default
"""""""

none

----------

.. _Morris:

**(Morris)** Morris, Fox, Zhu, J Comp Physics, 136, 214-226 (1997).
