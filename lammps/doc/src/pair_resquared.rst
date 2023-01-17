.. index:: pair_style resquared
.. index:: pair_style resquared/gpu
.. index:: pair_style resquared/omp

pair_style resquared command
============================

Accelerator Variants: *resquared/gpu*, *resquared/omp*

Syntax
""""""

.. code-block:: LAMMPS

   pair_style resquared cutoff

* cutoff = global cutoff for interactions (distance units)

Examples
""""""""

.. code-block:: LAMMPS

   pair_style resquared 10.0
   pair_coeff * * 1.0 1.0 1.7 3.4 3.4 1.0 1.0 1.0

Description
"""""""""""

Style *resquared* computes the RE-squared anisotropic interaction
:ref:`(Everaers) <Everaers3>`, :ref:`(Babadi) <Babadi>` between pairs of
ellipsoidal and/or spherical Lennard-Jones particles.  For ellipsoidal
interactions, the potential considers the ellipsoid as being comprised
of small spheres of size :math:`\sigma`.  LJ particles are a single sphere of
size :math:`\sigma`.  The distinction is made to allow the pair style to make
efficient calculations of ellipsoid/solvent interactions.

Details for the equations used are given in the references below and
in `this supplementary document <PDF/pair_resquared_extra.pdf>`_.

Use of this pair style requires the NVE, NVT, or NPT fixes with the
*asphere* extension (e.g. :doc:`fix nve/asphere <fix_nve_asphere>`) in
order to integrate particle rotation.  Additionally, :doc:`atom_style ellipsoid <atom_style>` should be used since it defines the
rotational state and the size and shape of each ellipsoidal particle.

The following coefficients must be defined for each pair of atoms
types via the :doc:`pair_coeff <pair_coeff>` command as in the examples
above, or in the data file or restart files read by the
:doc:`read_data <read_data>` or :doc:`read_restart <read_restart>`
commands:

* A12 = Energy Prefactor/Hamaker constant (energy units)
* :math:`\sigma` = atomic interaction diameter (distance units)
* :math:`\epsilon_{i,a}` = relative well depth of type I for side-to-side interactions
* :math:`\epsilon_{i,b}` = relative well depth of type I for face-to-face interactions
* :math:`\epsilon_{i,c}` = relative well depth of type I for end-to-end interactions
* :math:`\epsilon_{j,a}` = relative well depth of type J for side-to-side interactions
* :math:`\epsilon_{j,b}` = relative well depth of type J for face-to-face interactions
* :math:`\epsilon_{j,c}` = relative well depth of type J for end-to-end interactions
* cutoff (distance units)

The parameters used depend on the type of the interacting particles,
i.e. ellipsoids or LJ spheres.  The type of a particle is determined
by the diameters specified for its 3 shape parameters.  If all 3 shape
parameters = 0.0, then the particle is treated as an LJ sphere.  The
:math:`\epsilon_{i,*}` or :math:`\epsilon_{j,*}` parameters are ignored
for LJ spheres.  If
the 3 shape parameters are > 0.0, then the particle is treated as an
ellipsoid (even if the 3 parameters are equal to each other).

A12 specifies the energy prefactor which depends on the types of the
two interacting particles.

For ellipsoid/ellipsoid interactions, the interaction is computed by
the formulas in the supplementary document referenced above.  A12 is
the Hamaker constant as described in :ref:`(Everaers) <Everaers3>`. In LJ
units:

.. math::

   A_{12} = 4\pi^2\epsilon_{\mathrm{LJ}}(\rho\sigma^3)^2

where :math:`\rho` gives the number density of the spherical particles
composing the ellipsoids and :math:`\epsilon_{\mathrm{LJ}}` determines
the interaction strength of the spherical particles.

For ellipsoid/LJ sphere interactions, the interaction is also computed
by the formulas in the supplementary document referenced above.  A12
has a modified form (see `here <PDF/pair_resquared_extra.pdf>`_ for
details):

.. math::

   A_{12} = 4\pi^2\epsilon_{\mathrm{LJ}}(\rho\sigma^3)

For ellipsoid/LJ sphere interactions, a correction to the distance-
of-closest approach equation has been implemented to reduce the error
from two particles of disparate sizes; see `this supplementary document <PDF/pair_resquared_extra.pdf>`_.

For LJ sphere/LJ sphere interactions, the interaction is computed
using the standard Lennard-Jones formula, which is much cheaper to
compute than the ellipsoidal formulas.  A12 is used as epsilon in the
standard LJ formula:

.. math::

   A_{12} = \epsilon_{\mathrm{LJ}}

and the specified :math:`\sigma` is used as the :math:`\sigma` in the
standard LJ formula.

When one of both of the interacting particles are ellipsoids, then
:math:`\sigma` specifies the diameter of the continuous distribution of
constituent particles within each ellipsoid used to model the RE-squared
potential.  Note that this is a different meaning for :math:`\sigma`
than the :doc:`pair_style gayberne <pair_gayberne>` potential uses.

The :math:`\epsilon_i` and :math:`\epsilon_j` coefficients are defined
for atom types, not for pairs of atom types.  Thus, in a series of
pair_coeff commands, they only need to be specified once for each atom
type.

Specifically, if any of :math:`\epsilon_{i,a}`, :math:`\epsilon_{i,b}`,
:math:`\epsilon_{i,c}` are non-zero, the three values are assigned to
atom type I.  If all the :math:`\epsilon_i` values are zero, they are
ignored.  If any of :math:`\epsilon_{j,a}`, :math:`\epsilon_{j,b}`,
:math:`\epsilon_{j,c}` are non-zero, the three values are assigned to
atom type J.  If all three :math:`\epsilon_i` values are zero, they are
ignored.  Thus the typical way to define the :math:`\epsilon_i` and
:math:`\epsilon_j` coefficients is to list their values in "pair_coeff
I J" commands when I = J, but set them to 0.0 when I != J.  If you do
list them when I != J, you should insure they are consistent with their
values in other pair_coeff commands.

Note that if this potential is being used as a sub-style of
:doc:`pair_style hybrid <pair_hybrid>`, and there is no "pair_coeff I I"
setting made for RE-squared for a particular type I (because I-I
interactions are computed by another hybrid pair potential), then you
still need to insure the epsilon a,b,c coefficients are assigned to
that type in a "pair_coeff I J" command.

For large uniform molecules it has been shown that the :math:`\epsilon_{*,*}`
energy parameters are approximately representable in terms of local
contact curvatures :ref:`(Everaers) <Everaers3>`:

.. math::

   \epsilon_a = \sigma \cdot { \frac{a}{ b \cdot c } }; \epsilon_b =
   \sigma \cdot { \frac{b}{ a \cdot c } }; \epsilon_c = \sigma \cdot {
   \frac{c}{ a \cdot b } }

where a, b, and c give the particle diameters.

The last coefficient is optional.  If not specified, the global cutoff
specified in the pair_style command is used.

----------

.. include:: accel_styles.rst

----------

Mixing, shift, table, tail correction, restart, rRESPA info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

For atom type pairs I,J and I != J, the epsilon and sigma coefficients
and cutoff distance can be mixed, but only for sphere pairs.  The
default mix value is *geometric*\ .  See the "pair_modify" command for
details.  Other type pairs cannot be mixed, due to the different
meanings of the energy prefactors used to calculate the interactions
and the implicit dependence of the ellipsoid-sphere interaction on the
equation for the Hamaker constant presented here.  Mixing of sigma and
epsilon followed by calculation of the energy prefactors using the
equations above is recommended.

This pair style supports the :doc:`pair_modify <pair_modify>` shift
option for the energy of the Lennard-Jones portion of the pair
interaction, but only for sphere-sphere interactions.  There is no
shifting performed for ellipsoidal interactions due to the anisotropic
dependence of the interaction.

The :doc:`pair_modify <pair_modify>` table option is not relevant
for this pair style.

This pair style does not support the :doc:`pair_modify <pair_modify>`
tail option for adding long-range tail corrections to energy and
pressure.

This pair style writes its information to :doc:`binary restart files <restart>`, so pair_style and pair_coeff commands do not need
to be specified in an input script that reads a restart file.

This pair style can only be used via the *pair* keyword of the
:doc:`run_style respa <run_style>` command.  It does not support the
*inner*, *middle*, *outer* keywords of the :doc:`run_style command <run_style>`.

----------

Restrictions
""""""""""""

This style is part of the ASPHERE package.  It is only enabled if
LAMMPS was built with that package.  See the :doc:`Build package <Build_package>` page for more info.

This pair style requires that atoms be ellipsoids as defined by the
:doc:`atom_style ellipsoid <atom_style>` command.

Particles acted on by the potential can be finite-size aspherical or
spherical particles, or point particles.  Spherical particles have all
3 of their shape parameters equal to each other.  Point particles have
all 3 of their shape parameters equal to 0.0.

The distance-of-closest-approach approximation used by LAMMPS becomes
less accurate when high-aspect ratio ellipsoids are used.

Related commands
""""""""""""""""

:doc:`pair_coeff <pair_coeff>`, :doc:`fix nve/asphere <fix_nve_asphere>`,
:doc:`compute temp/asphere <compute_temp_asphere>`, :doc:`pair_style gayberne <pair_gayberne>`

Default
"""""""

none

----------

.. _Everaers3:

**(Everaers)** Everaers and Ejtehadi, Phys Rev E, 67, 041710 (2003).

.. _Babadi:

**(Babadi)** Babadi, Ejtehadi, Everaers, J Comp Phys, 219, 770-779 (2006).
