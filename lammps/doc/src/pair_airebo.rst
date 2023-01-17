.. index:: pair_style airebo
.. index:: pair_style airebo/intel
.. index:: pair_style airebo/omp
.. index:: pair_style airebo/morse
.. index:: pair_style airebo/morse/intel
.. index:: pair_style airebo/morse/omp
.. index:: pair_style rebo
.. index:: pair_style rebo/intel
.. index:: pair_style rebo/omp

pair_style airebo command
=========================

Accelerator Variants: *airebo/intel*, *airebo/omp*

pair_style airebo/morse command
===============================

Accelerator Variants: *airebo/morse/intel*, *airebo/morse/omp*

pair_style rebo command
=======================

Accelerator Variants: *rebo/intel*, *rebo/omp*

Syntax
""""""

.. code-block:: LAMMPS

   pair_style style cutoff LJ_flag TORSION_flag cutoff_min

* style = *airebo* or *airebo/morse* or *rebo*
* cutoff = LJ or Morse cutoff (:math:`\sigma` scale factor) (AIREBO and AIREBO-M only)
* LJ_flag = 0/1 to turn off/on the LJ or Morse term (AIREBO and AIREBO-M only, optional)
* TORSION_flag = 0/1 to turn off/on the torsion term (AIREBO and AIREBO-M only, optional)
* cutoff_min = Start of the transition region of cutoff (:math:`\sigma` scale factor) (AIREBO and AIREBO-M only, optional)

Examples
""""""""

.. code-block:: LAMMPS

   pair_style airebo 3.0
   pair_style airebo 2.5 1 0
   pair_coeff * * ../potentials/CH.airebo H C

   pair_style airebo/morse 3.0
   pair_coeff * * ../potentials/CH.airebo-m H C

   pair_style rebo
   pair_coeff * * ../potentials/CH.rebo H C

Description
"""""""""""

The *airebo* pair style computes the Adaptive Intermolecular Reactive
Empirical Bond Order (AIREBO) Potential of :ref:`(Stuart) <Stuart>` for a
system of carbon and/or hydrogen atoms.  Note that this is the initial
formulation of AIREBO from 2000, not the later formulation.

The *airebo/morse* pair style computes the AIREBO-M potential, which
is equivalent to AIREBO, but replaces the LJ term with a Morse potential.
The Morse potentials are parameterized by high-quality quantum chemistry
(MP2) calculations and do not diverge as quickly as particle density
increases. This allows AIREBO-M to retain accuracy to much higher pressures
than AIREBO (up to 40 GPa for Polyethylene). Details for this potential
and its parameterization are given in :ref:`(O'Conner) <OConnor>`.

The *rebo* pair style computes the Reactive Empirical Bond Order (REBO)
Potential of :ref:`(Brenner) <Brenner>`. Note that this is the so-called
second generation REBO from 2002, not the original REBO from 1990.
As discussed below, second generation REBO is closely related to the
initial AIREBO; it is just a subset of the potential energy terms
with a few slightly different parameters

The AIREBO potential consists of three terms:

.. math::

   E & = \frac{1}{2} \sum_i \sum_{j \neq i}
   \left[ E^{\text{REBO}}_{ij} + E^{\text{LJ}}_{ij} +
    \sum_{k \neq i,j} \sum_{l \neq i,j,k} E^{\text{TORSION}}_{kijl} \right] \\

By default, all three terms are included.  For the *airebo* style, if
the first two optional flag arguments to the pair_style command are
included, the LJ and torsional terms can be turned off.  Note that
both or neither of the flags must be included.  If both of the LJ an
torsional terms are turned off, it becomes the second-generation REBO
potential, with a small caveat on the spline fitting procedure
mentioned below.  This can be specified directly as pair_style *rebo*
with no additional arguments.

The detailed formulas for this potential are given in
:ref:`(Stuart) <Stuart>`; here we provide only a brief description.

The :math:`E^{\text{REBO}}` term has the same functional form as the hydrocarbon REBO
potential developed in :ref:`(Brenner) <Brenner>`.  The coefficients for
:math:`E^{\text{REBO}}` in AIREBO are essentially the same as Brenner's potential, but
a few fitted spline values are slightly different.  For most cases the
:math:`E^{\text{REBO}}` term in AIREBO will produce the same energies, forces and
statistical averages as the original REBO potential from which it was
derived.  The :math:`E^{\text{REBO}}` term in the AIREBO potential gives the model its
reactive capabilities and only describes short-ranged C-C, C-H and H-H
interactions (:math:`r < 2` Angstroms). These interactions have strong
coordination-dependence through a bond order parameter, which adjusts
the attraction between the I,J atoms based on the position of other
nearby atoms and thus has 3- and 4-body dependence.

The :math:`E^{\text{LJ}}` term adds longer-ranged interactions (:math:`2 < r < \text{cutoff}`) using a
form similar to the standard :doc:`Lennard Jones potential <pair_lj>`.
The :math:`E^{\text{LJ}}` term in AIREBO contains a series of switching functions so
that the short-ranged LJ repulsion (:math:`1/r^{12}`) does not interfere with
the energetics captured by the :math:`E^{\text{REBO}}` term.  The extent of the :math:`E^{\text{LJ}}`
interactions is determined by the *cutoff* argument to the pair_style
command which is a scale factor.  For each type pair (C-C, C-H, H-H)
the cutoff is obtained by multiplying the scale factor by the sigma
value defined in the potential file for that type pair.  In the
standard AIREBO potential, :math:`\sigma_{CC} = 3.4` Angstroms, so with a scale
factor of 3.0 (the argument in pair_style), the resulting :math:`E^{\text{LJ}}` cutoff
would be 10.2 Angstroms.

By default, the longer-ranged interaction is smoothly switched off
between 2.16 and 3.0 :math:`\sigma`. By specifying *cutoff_min* in addition
to *cutoff*, the switching can be configured to take place between
*cutoff_min* and *cutoff*\ . *cutoff_min* can only be specified if all
optional arguments are given.

The :math:`E^{\text{TORSION}}` term is an explicit 4-body potential that describes
various dihedral angle preferences in hydrocarbon configurations.

----------

Only a single pair_coeff command is used with the *airebo*, *airebo*
or *rebo* style which specifies an AIREBO, REBO, or AIREBO-M potential
file with parameters for C and H.  Note that as of LAMMPS version
15 May 2019 the *rebo* style in LAMMPS uses its own potential
file (CH.rebo).  These are mapped to LAMMPS atom types by specifying
N additional arguments after the filename in the pair_coeff command,
where N is the number of LAMMPS atom types:

* filename
* :math:`N` element names = mapping of AIREBO elements to atom types

See the :doc:`pair_coeff <pair_coeff>` page for alternate ways
to specify the path for the potential file.

As an example, if your LAMMPS simulation has 4 atom types and you want
the first 3 to be C, and the fourth to be H, you would use the following
pair_coeff command:

.. code-block:: LAMMPS

   pair_coeff * * CH.airebo C C C H

The first 2 arguments must be \* \* so as to span all LAMMPS atom types.
The first three C arguments map LAMMPS atom types 1,2,3 to the C
element in the AIREBO file.  The final H argument maps LAMMPS atom
type 4 to the H element in the SW file.  If a mapping value is
specified as NULL, the mapping is not performed.  This can be used
when a *airebo* potential is used as part of the *hybrid* pair style.
The NULL values are placeholders for atom types that will be used with
other potentials.

The parameters/coefficients for the AIREBO potentials are listed in
the CH.airebo file to agree with the original :ref:`(Stuart) <Stuart>`
paper.  Thus the parameters are specific to this potential and the way
it was fit, so modifying the file should be done cautiously.

Similarly the parameters/coefficients for the AIREBO-M potentials are
listed in the CH.airebo-m file to agree with the :ref:`(O'Connor) <OConnor>`
paper. Thus the parameters are specific to this potential and the way
it was fit, so modifying the file should be done cautiously. The
AIREBO-M Morse potentials were parameterized using a cutoff of
3.0 (:math:`\sigma`). Modifying this cutoff may impact simulation accuracy.

This pair style tallies a breakdown of the total AIREBO potential
energy into sub-categories, which can be accessed via the :doc:`compute pair <compute_pair>` command as a vector of values of length 3.
The 3 values correspond to the following sub-categories:

1. :math:`E_{\text{REBO}}` = REBO energy
2. :math:`E_{\text{LJ}}` = Lennard-Jones energy
3. :math:`E_{\text{TORSION}}` = Torsion energy

To print these quantities to the log file (with descriptive column
headings) the following commands could be included in an input script:

.. code-block:: LAMMPS

   compute 0 all pair airebo
   variable REBO     equal c_0[1]
   variable LJ       equal c_0[2]
   variable TORSION  equal c_0[3]
   thermo_style custom step temp epair v_REBO v_LJ v_TORSION

----------

.. include:: accel_styles.rst

----------

Mixing, shift, table, tail correction, restart, rRESPA info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

These pair styles do not support the :doc:`pair_modify <pair_modify>`
mix, shift, table, and tail options.

These pair styles do not write their information to :doc:`binary restart files <restart>`, since it is stored in potential files.  Thus, you
need to re-specify the pair_style and pair_coeff commands in an input
script that reads a restart file.

These pair styles can only be used via the *pair* keyword of the
:doc:`run_style respa <run_style>` command.  They do not support the
*inner*, *middle*, *outer* keywords.

Restrictions
""""""""""""

These pair styles are part of the MANYBODY package.  They are only
enabled if LAMMPS was built with that package.  See the :doc:`Build package <Build_package>` page for more info.

These pair potentials require the :doc:`newton <newton>` setting to be
"on" for pair interactions.

The CH.airebo and CH.airebo-m potential files provided with LAMMPS
(see the potentials directory) are parameterized for metal :doc:`units <units>`.
You can use the AIREBO, AIREBO-M or REBO potential with any LAMMPS units,
but you would need to create your own AIREBO or AIREBO-M potential file
with coefficients listed in the appropriate units, if your simulation
does not use "metal" units.

The pair styles provided here **only** support potential files parameterized
for the elements carbon and hydrogen (designated with "C" and "H" in the
*pair_coeff* command.  Using potential files for other elements will trigger
an error.

Related commands
""""""""""""""""

:doc:`pair_coeff <pair_coeff>`

Default
"""""""

none

----------

.. _Stuart:

**(Stuart)** Stuart, Tutein, Harrison, J Chem Phys, 112, 6472-6486
(2000).

.. _Brenner:

**(Brenner)** Brenner, Shenderova, Harrison, Stuart, Ni, Sinnott, J
Physics: Condensed Matter, 14, 783-802 (2002).

.. _OConnor:

**(O'Connor)** O'Connor et al., J. Chem. Phys. 142, 024903 (2015).
