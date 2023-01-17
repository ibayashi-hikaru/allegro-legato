.. index:: pair_style sw
.. index:: pair_style sw/gpu
.. index:: pair_style sw/intel
.. index:: pair_style sw/kk
.. index:: pair_style sw/omp

pair_style sw command
=====================

Accelerator Variants: *sw/gpu*, *sw/intel*, *sw/kk*, *sw/omp*

Syntax
""""""

.. code-block:: LAMMPS

   pair_style sw

Examples
""""""""

.. code-block:: LAMMPS

   pair_style sw
   pair_coeff * * si.sw Si
   pair_coeff * * GaN.sw Ga N Ga

Description
"""""""""""

The *sw* style computes a 3-body :ref:`Stillinger-Weber <Stillinger2>`
potential for the energy E of a system of atoms as

.. math::

   E & =  \sum_i \sum_{j > i} \phi_2 (r_{ij}) +
          \sum_i \sum_{j \neq i} \sum_{k > j}
          \phi_3 (r_{ij}, r_{ik}, \theta_{ijk}) \\
  \phi_2(r_{ij}) & =  A_{ij} \epsilon_{ij} \left[ B_{ij} (\frac{\sigma_{ij}}{r_{ij}})^{p_{ij}} -
                    (\frac{\sigma_{ij}}{r_{ij}})^{q_{ij}} \right]
                    \exp \left( \frac{\sigma_{ij}}{r_{ij} - a_{ij} \sigma_{ij}} \right) \\
  \phi_3(r_{ij},r_{ik},\theta_{ijk}) & = \lambda_{ijk} \epsilon_{ijk} \left[ \cos \theta_{ijk} -
                    \cos \theta_{0ijk} \right]^2
                    \exp \left( \frac{\gamma_{ij} \sigma_{ij}}{r_{ij} - a_{ij} \sigma_{ij}} \right)
                    \exp \left( \frac{\gamma_{ik} \sigma_{ik}}{r_{ik} - a_{ik} \sigma_{ik}} \right)

where :math:`\phi_2` is a two-body term and :math:`\phi_3` is a
three-body term.  The summations in the formula are over all neighbors J
and K of atom I within a cutoff distance :math:`a `\sigma`.

Only a single pair_coeff command is used with the *sw* style which
specifies a Stillinger-Weber potential file with parameters for all
needed elements.  These are mapped to LAMMPS atom types by specifying
N additional arguments after the filename in the pair_coeff command,
where N is the number of LAMMPS atom types:

* filename
* N element names = mapping of SW elements to atom types

See the :doc:`pair_coeff <pair_coeff>` page for alternate ways
to specify the path for the potential file.

As an example, imagine a file SiC.sw has Stillinger-Weber values for
Si and C.  If your LAMMPS simulation has 4 atoms types and you want
the first 3 to be Si, and the fourth to be C, you would use the following
pair_coeff command:

.. code-block:: LAMMPS

   pair_coeff * * SiC.sw Si Si Si C

The first 2 arguments must be \* \* so as to span all LAMMPS atom types.
The first three Si arguments map LAMMPS atom types 1,2,3 to the Si
element in the SW file.  The final C argument maps LAMMPS atom type 4
to the C element in the SW file.  If a mapping value is specified as
NULL, the mapping is not performed.  This can be used when a *sw*
potential is used as part of the *hybrid* pair style.  The NULL values
are placeholders for atom types that will be used with other
potentials.

Stillinger-Weber files in the *potentials* directory of the LAMMPS
distribution have a ".sw" suffix.  Lines that are not blank or
comments (starting with #) define parameters for a triplet of
elements.  The parameters in a single entry correspond to the two-body
and three-body coefficients in the formula above:

* element 1 (the center atom in a 3-body interaction)
* element 2
* element 3
* :math:`\epsilon` (energy units)
* :math:`\sigma` (distance units)
* a
* :math:`\lambda`
* :math:`\gamma`
* :math:`\cos\theta_0`
* A
* B
* p
* q
* tol

The A, B, p, and q parameters are used only for two-body interactions.
The :math:`\lambda` and :math:`\cos\theta_0` parameters are used only
for three-body interactions. The :math:`\epsilon`, :math:`\sigma` and
*a* parameters are used for both two-body and three-body
interactions. :math:`\gamma` is used only in the three-body
interactions, but is defined for pairs of atoms.  The non-annotated
parameters are unitless.

LAMMPS introduces an additional performance-optimization parameter tol
that is used for both two-body and three-body interactions.  In the
Stillinger-Weber potential, the interaction energies become negligibly
small at atomic separations substantially less than the theoretical
cutoff distances.  LAMMPS therefore defines a virtual cutoff distance
based on a user defined tolerance tol.  The use of the virtual cutoff
distance in constructing atom neighbor lists can significantly reduce
the neighbor list sizes and therefore the computational cost.  LAMMPS
provides a *tol* value for each of the three-body entries so that they
can be separately controlled. If tol = 0.0, then the standard
Stillinger-Weber cutoff is used.

The Stillinger-Weber potential file must contain entries for all the
elements listed in the pair_coeff command.  It can also contain
entries for additional elements not being used in a particular
simulation; LAMMPS ignores those entries.

For a single-element simulation, only a single entry is required
(e.g. SiSiSi).  For a two-element simulation, the file must contain 8
entries (for SiSiSi, SiSiC, SiCSi, SiCC, CSiSi, CSiC, CCSi, CCC), that
specify SW parameters for all permutations of the two elements
interacting in three-body configurations.  Thus for 3 elements, 27
entries would be required, etc.

As annotated above, the first element in the entry is the center atom
in a three-body interaction.  Thus an entry for SiCC means a Si atom
with 2 C atoms as neighbors.  The parameter values used for the
two-body interaction come from the entry where the second and third
elements are the same.  Thus the two-body parameters for Si
interacting with C, comes from the SiCC entry.  The three-body
parameters can in principle be specific to the three elements of the
configuration. In the literature, however, the three-body parameters
are usually defined by simple formulas involving two sets of pair-wise
parameters, corresponding to the ij and ik pairs, where i is the
center atom. The user must ensure that the correct combining rule is
used to calculate the values of the three-body parameters for
alloys. Note also that the function :math:`\phi_3` contains two exponential
screening factors with parameter values from the ij pair and ik
pairs. So :math:`\phi_3` for a C atom bonded to a Si atom and a second C atom
will depend on the three-body parameters for the CSiC entry, and also
on the two-body parameters for the CCC and CSiSi entries. Since the
order of the two neighbors is arbitrary, the three-body parameters for
entries CSiC and CCSi should be the same.  Similarly, the two-body
parameters for entries SiCC and CSiSi should also be the same.  The
parameters used only for two-body interactions (A, B, p, and q) in
entries whose second and third element are different (e.g. SiCSi) are not
used for anything and can be set to 0.0 if desired.
This is also true for the parameters in :math:`\phi_3` that are
taken from the ij and ik pairs (:math:`\sigma`, *a*, :math:`\gamma`)

----------

Styles with a *gpu*, *intel*, *kk*, *omp*, or *opt* suffix are
functionally the same as the corresponding style without the suffix.
They have been optimized to run faster, depending on your available
hardware, as discussed on the :doc:`Speed packages <Speed_packages>` doc
page.  The accelerated styles take the same arguments and should
produce the same results, except for round-off and precision issues.

These accelerated styles are part of the GPU, INTEL, KOKKOS,
OPENMP and OPT packages, respectively.  They are only enabled if
LAMMPS was built with those packages.  See the :doc:`Build package <Build_package>` page for more info.

You can specify the accelerated styles explicitly in your input script
by including their suffix, or you can use the :doc:`-suffix command-line switch <Run_options>` when you invoke LAMMPS, or you can use the
:doc:`suffix <suffix>` command in your input script.

When using the INTEL package with this style, there is an
additional 5 to 10 percent performance improvement when the
Stillinger-Weber parameters p and q are set to 4 and 0 respectively.
These parameters are common for modeling silicon and water.

See the :doc:`Speed packages <Speed_packages>` page for more
instructions on how to use the accelerated styles effectively.

----------

Mixing, shift, table, tail correction, restart, rRESPA info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

For atom type pairs I,J and I != J, where types I and J correspond to
two different element types, mixing is performed by LAMMPS as
described above from values in the potential file.

This pair style does not support the :doc:`pair_modify <pair_modify>`
shift, table, and tail options.

This pair style does not write its information to :doc:`binary restart files <restart>`, since it is stored in potential files.  Thus, you
need to re-specify the pair_style and pair_coeff commands in an input
script that reads a restart file.

This pair style can only be used via the *pair* keyword of the
:doc:`run_style respa <run_style>` command.  It does not support the
*inner*, *middle*, *outer* keywords.

----------

Restrictions
""""""""""""

This pair style is part of the MANYBODY package.  It is only enabled
if LAMMPS was built with that package.  See the :doc:`Build package <Build_package>` page for more info.

This pair style requires the :doc:`newton <newton>` setting to be "on"
for pair interactions.

The Stillinger-Weber potential files provided with LAMMPS (see the
potentials directory) are parameterized for metal :doc:`units <units>`.
You can use the SW potential with any LAMMPS units, but you would need
to create your own SW potential file with coefficients listed in the
appropriate units if your simulation does not use "metal" units.

Related commands
""""""""""""""""

:doc:`pair_coeff <pair_coeff>`

Default
"""""""

none

----------

.. _Stillinger2:

**(Stillinger)** Stillinger and Weber, Phys Rev B, 31, 5262 (1985).
