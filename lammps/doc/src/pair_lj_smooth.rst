.. index:: pair_style lj/smooth
.. index:: pair_style lj/smooth/gpu
.. index:: pair_style lj/smooth/omp

pair_style lj/smooth command
============================

Accelerator Variants: *lj/smooth/gpu*, *lj/smooth/omp*

Syntax
""""""

.. code-block:: LAMMPS

   pair_style lj/smooth Rin Rc

* Rin = inner cutoff beyond which force smoothing will be applied (distance units)
* Rc = outer cutoff for lj/smooth interactions (distance units)

Examples
""""""""

.. code-block:: LAMMPS

   pair_style lj/smooth 8.0 10.0
   pair_coeff * * 10.0 1.5
   pair_coeff 1 1 20.0 1.3 7.0 9.0

Description
"""""""""""

Style *lj/smooth* computes a LJ interaction with a force smoothing
applied between the inner and outer cutoff.

.. math::

   E & =  4 \epsilon \left[ \left(\frac{\sigma}{r}\right)^{12} -
                         \left(\frac{\sigma}{r}\right)^6 \right]
                         \qquad r < r_{in} \\
   F & =  C_1 + C_2 (r - r_{in}) + C_3 (r - r_{in})^2 + C_4 (r - r_{in})^3
                       \qquad r_{in} < r < r_c

The polynomial coefficients C1, C2, C3, C4 are computed by LAMMPS to
cause the force to vary smoothly from the inner cutoff :math:`r_{in}` to the
outer cutoff :math:`r_c`.

At the inner cutoff the force and its first derivative
will match the non-smoothed LJ formula.  At the outer cutoff the force
and its first derivative will be 0.0.  The inner cutoff cannot be 0.0.

.. note::

   this force smoothing causes the energy to be discontinuous both
   in its values and first derivative.  This can lead to poor energy
   conservation and may require the use of a thermostat.  Plot the energy
   and force resulting from this formula via the
   :doc:`pair_write <pair_write>` command to see the effect.

The following coefficients must be defined for each pair of atoms
types via the :doc:`pair_coeff <pair_coeff>` command as in the examples
above, or in the data file or restart files read by the
:doc:`read_data <read_data>` or :doc:`read_restart <read_restart>`
commands, or by mixing as described below:

* :math:`\epsilon` (energy units)
* :math:`\sigma` (distance units)
* :math:`r_{in}` (distance units)
* :math:`r_c` (distance units)

The last 2 coefficients are optional inner and outer cutoffs.  If not
specified, the global values for :math:`r_{in}` and :math:`r_c` are used.

----------

.. include:: accel_styles.rst

----------

Mixing, shift, table, tail correction, restart, rRESPA info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

For atom type pairs I,J and I != J, the epsilon, sigma, Rin
coefficients and the cutoff distance for this pair style can be mixed.
Rin is a cutoff value and is mixed like the cutoff.  The other
coefficients are mixed according to the pair_modify mix option.  The
default mix value is *geometric*\ .  See the "pair_modify" command for
details.

This pair style supports the :doc:`pair_modify <pair_modify>` shift
option for the energy of the pair interaction.

The :doc:`pair_modify <pair_modify>` table option is not relevant
for this pair style.

This pair style does not support the :doc:`pair_modify <pair_modify>`
tail option for adding long-range tail corrections to energy and
pressure, since the energy of the pair interaction is smoothed to 0.0
at the cutoff.

This pair style writes its information to :doc:`binary restart files <restart>`, so pair_style and pair_coeff commands do not need
to be specified in an input script that reads a restart file.

This pair style can only be used via the *pair* keyword of the
:doc:`run_style respa <run_style>` command.  It does not support the
*inner*, *middle*, *outer* keywords.

----------

Restrictions
""""""""""""

This pair style is part of the EXTRA-PAIR package.  It is only enabled if
LAMMPS was built with that package.  See the
:doc:`Build package <Build_package>` page for more info.

Related commands
""""""""""""""""

:doc:`pair_coeff <pair_coeff>`, :doc:`pair lj/smooth/linear <pair_lj_smooth_linear>`

Default
"""""""

none
