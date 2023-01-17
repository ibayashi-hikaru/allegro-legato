.. index:: pair_style tersoff/zbl
.. index:: pair_style tersoff/zbl/gpu
.. index:: pair_style tersoff/zbl/kk
.. index:: pair_style tersoff/zbl/omp

pair_style tersoff/zbl command
==============================

Accelerator Variants: *tersoff/zbl/gpu*, *tersoff/zbl/kk*, *tersoff/zbl/omp*

Syntax
""""""

.. code-block:: LAMMPS

   pair_style tersoff/zbl keywords values

* keyword = *shift*

  .. parsed-literal::

       *shift* value = delta
         delta = negative shift in equilibrium bond length

Examples
""""""""

.. code-block:: LAMMPS

   pair_style tersoff/zbl
   pair_coeff * * SiC.tersoff.zbl Si C Si

Description
"""""""""""

The *tersoff/zbl* style computes a 3-body Tersoff potential
:ref:`(Tersoff_1) <zbl-Tersoff_1>` with a close-separation pairwise modification
based on a Coulomb potential and the Ziegler-Biersack-Littmark
universal screening function :ref:`(ZBL) <zbl-ZBL>`, giving the energy E of a
system of atoms as

.. math::

   E & = \frac{1}{2} \sum_i \sum_{j \neq i} V_{ij} \\
   V_{ij} & =  (1 - f_F(r_{ij} + \delta)) V^{ZBL}(r_{ij} + \delta)
               + f_F(r_{ij} + \delta) V^{Tersoff}(r_{ij} + \delta) \\
   f_F(r) & =  \frac{1}{1 + e^{-A_F(r - r_C)}}\\
   \\
   \\
   V^{ZBL}(r) & = \frac{1}{4\pi\epsilon_0} \frac{Z_1 Z_2 \,e^2}{r} \phi(r/a) \\
  a & = \frac{0.8854\,a_0}{Z_{1}^{0.23} + Z_{2}^{0.23}}\\
  \phi(x) & =  0.1818e^{-3.2x} + 0.5099e^{-0.9423x} + 0.2802e^{-0.4029x} + 0.02817e^{-0.2016x}\\
  \\
  \\
  V^{Tersoff}(r) & = f_C(r) \left[ f_R(r) + b_{ij} f_A(r) \right] \\
  f_C(r) & = \left\{ \begin{array} {r@{\quad:\quad}l}
    1 & r < R - D \\
    \frac{1}{2} - \frac{1}{2} \sin \left( \frac{\pi}{2} \frac{r-R}{D} \right) &
      R-D < r < R + D \\
    0 & r > R + D
    \end{array} \right. \\
  f_R(r) & = A \exp (-\lambda_1 r) \\
  f_A(r) & = -B \exp (-\lambda_2 r) \\
  b_{ij} & = \left( 1 + \beta^n {\zeta_{ij}}^n \right)^{-\frac{1}{2n}} \\
  \zeta_{ij} & = \sum_{k \neq i,j} f_C(r_{ik} + \delta) g(\theta_{ijk})
                   \exp \left[ {\lambda_3}^m (r_{ij} - r_{ik})^m \right] \\
  g(\theta) & =  \gamma_{ijk} \left( 1 + \frac{c^2}{d^2} -
                  \frac{c^2}{\left[ d^2 + (\cos \theta - \cos \theta_0)^2\right]} \right)

The :math:`f_F` term is a fermi-like function used to smoothly connect the ZBL
repulsive potential with the Tersoff potential.  There are 2
parameters used to adjust it: :math:`A_F` and :math:`r_C`.  :math:`A_F`
controls how "sharp"
the transition is between the two, and :math:`r_C` is essentially the cutoff
for the ZBL potential.

For the ZBL portion, there are two terms. The first is the Coulomb
repulsive term, with Z1, Z2 as the number of protons in each nucleus,
e as the electron charge (1 for metal and real units) and :math:`\epsilon_0`
as the permittivity of vacuum.  The second part is the ZBL universal
screening function, with a0 being the Bohr radius (typically 0.529
Angstroms), and the remainder of the coefficients provided by the
original paper.  This screening function should be applicable to most
systems.  However, it is only accurate for small separations
(i.e. less than 1 Angstrom).

For the Tersoff portion, :math:`f_R` is a two-body term and :math:`f_A`
includes
three-body interactions. The summations in the formula are over all
neighbors J and K of atom I within a cutoff distance = R + D.

:math:`\delta` is an optional negative shift of the
equilibrium bond length, as described below.

Only a single pair_coeff command is used with the *tersoff/zbl* style
which specifies a Tersoff/ZBL potential file with parameters for all
needed elements.  These are mapped to LAMMPS atom types by specifying
N additional arguments after the filename in the pair_coeff command,
where N is the number of LAMMPS atom types:

* filename
* N element names = mapping of Tersoff/ZBL elements to atom types

See the :doc:`pair_coeff <pair_coeff>` page for alternate ways
to specify the path for the potential file.

As an example, imagine the SiC.tersoff.zbl file has Tersoff/ZBL values
for Si and C.  If your LAMMPS simulation has 4 atoms types and you
want the first 3 to be Si, and the fourth to be C, you would use the
following pair_coeff command:

.. code-block:: LAMMPS

   pair_coeff * * SiC.tersoff Si Si Si C

The first 2 arguments must be \* \* so as to span all LAMMPS atom types.
The first three Si arguments map LAMMPS atom types 1,2,3 to the Si
element in the Tersoff/ZBL file.  The final C argument maps LAMMPS
atom type 4 to the C element in the Tersoff/ZBL file.  If a mapping
value is specified as NULL, the mapping is not performed.  This can be
used when a *tersoff/zbl* potential is used as part of the *hybrid*
pair style.  The NULL values are placeholders for atom types that will
be used with other potentials.

Tersoff/ZBL files in the *potentials* directory of the LAMMPS
distribution have a ".tersoff.zbl" suffix.  Lines that are not blank
or comments (starting with #) define parameters for a triplet of
elements.  The parameters in a single entry correspond to coefficients
in the formula above:

* element 1 (the center atom in a 3-body interaction)
* element 2 (the atom bonded to the center atom)
* element 3 (the atom influencing the 1-2 bond in a bond-order sense)
* m
* :math:`\gamma`
* :math:`\lambda_3` (1/distance units)
* c
* d
* :math:`\cos\theta_0` (can be a value < -1 or > 1)
* n
* :math:`\beta`
* :math:`\lambda_2` (1/distance units)
* B (energy units)
* R (distance units)
* D (distance units)
* :math:`\lambda_1` (1/distance units)
* A (energy units)
* :math:`Z_i`
* :math:`Z_j`
* ZBLcut (distance units)
* ZBLexpscale (1/distance units)

The n, :math:`\beta`, :math:`\lambda_2`, B, :math:`\lambda_1`, and A
parameters are only used for
two-body interactions.  The m, :math:`\gamma`, :math:`\lambda_3`, c, d,
and :math:`\cos\theta_0`
parameters are only used for three-body interactions. The R and D
parameters are used for both two-body and three-body interactions. The
:math:`Z_i`, :math:`Z_j`, ZBLcut, ZBLexpscale parameters are used in the
ZBL repulsive
portion of the potential and in the Fermi-like function.  The
non-annotated parameters are unitless.  The value of m must be 3 or 1.

The Tersoff/ZBL potential file must contain entries for all the
elements listed in the pair_coeff command.  It can also contain
entries for additional elements not being used in a particular
simulation; LAMMPS ignores those entries.

For a single-element simulation, only a single entry is required
(e.g. SiSiSi).  For a two-element simulation, the file must contain 8
entries (for SiSiSi, SiSiC, SiCSi, SiCC, CSiSi, CSiC, CCSi, CCC), that
specify Tersoff parameters for all permutations of the two elements
interacting in three-body configurations.  Thus for 3 elements, 27
entries would be required, etc.

As annotated above, the first element in the entry is the center atom
in a three-body interaction and it is bonded to the second atom and the
bond is influenced by the third atom.  Thus an entry for SiCC means Si
bonded to a C with another C atom influencing the bond.  Thus
three-body parameters for SiCSi and SiSiC entries will not, in
general, be the same.  The parameters used for the two-body
interaction come from the entry where the second element is repeated.
Thus the two-body parameters for Si interacting with C, comes from the
SiCC entry.

The parameters used for a particular
three-body interaction come from the entry with the corresponding
three elements.  The parameters used only for two-body interactions
(n, :math:`\beta`, :math:`\lambda_2`, B, :math:`\lambda_1`, and A)
in entries whose second and third
element are different (e.g. SiCSi) are not used for anything and can
be set to 0.0 if desired.

Note that the twobody parameters in entries such as SiCC and CSiSi
are often the same, due to the common use of symmetric mixing rules,
but this is not always the case. For example, the beta and n parameters in
Tersoff_2 :ref:`(Tersoff_2) <zbl-Tersoff_2>` are not symmetric.

We chose the above form so as to enable users to define all commonly
used variants of the Tersoff portion of the potential.  In particular,
our form reduces to the original Tersoff form when m = 3 and gamma =
1, while it reduces to the form of :ref:`Albe et al. <zbl-Albe>` when beta = 1
and m = 1.  Note that in the current Tersoff implementation in LAMMPS,
m must be specified as either 3 or 1.  Tersoff used a slightly
different but equivalent form for alloys, which we will refer to as
Tersoff_2 potential :ref:`(Tersoff_2) <zbl-Tersoff_2>`.

LAMMPS parameter values for Tersoff_2 can be obtained as follows:
:math:`\gamma = \omega_{ijk}`, :math:`\lambda_3 = 0` and the value of
m has no effect.  The parameters for species i and j can be calculated
using the Tersoff_2 mixing rules:

.. math::

   \lambda_1^{i,j} & = \frac{1}{2}(\lambda_1^i + \lambda_1^j)\\
   \lambda_2^{i,j} & = \frac{1}{2}(\lambda_2^i + \lambda_2^j)\\
   A_{i,j} & = (A_{i}A_{j})^{1/2}\\
   B_{i,j} & = \chi_{ij}(B_{i}B_{j})^{1/2}\\
   R_{i,j} & = (R_{i}R_{j})^{1/2}\\
   S_{i,j} & = (S_{i}S_{j})^{1/2}\\

Tersoff_2 parameters R and S must be converted to the LAMMPS
parameters R and D (R is different in both forms), using the following
relations: R=(R'+S')/2 and D=(S'-R')/2, where the primes indicate the
Tersoff_2 parameters.

In the potentials directory, the file SiCGe.tersoff provides the
LAMMPS parameters for Tersoff's various versions of Si, as well as his
alloy parameters for Si, C, and Ge. This file can be used for pure Si,
(three different versions), pure C, pure Ge, binary SiC, and binary
SiGe.  LAMMPS will generate an error if this file is used with any
combination involving C and Ge, since there are no entries for the GeC
interactions (Tersoff did not publish parameters for this
cross-interaction.)  Tersoff files are also provided for the SiC alloy
(SiC.tersoff) and the GaN (GaN.tersoff) alloys.

Many thanks to Rutuparna Narulkar, David Farrell, and Xiaowang Zhou
for helping clarify how Tersoff parameters for alloys have been
defined in various papers.  Also thanks to Ram Devanathan for
providing the base ZBL implementation.

The *shift* keyword computes the energy E of a system of atoms, whose
formula is the same as the Tersoff potential. The only modification is
that the original equilibrium bond length ( :math:`r_0`) of the system
is shifted to :math:`r_0-\delta`.  The minus sign arises because each
radial distance :math:`r` is replaced by :math:`r+\delta`.  More
information on this option is given on the main :doc:`pair_tersoff
<pair_tersoff>` page.

----------

.. include:: accel_styles.rst

----------

Mixing, shift, table, tail correction, restart, rRESPA info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

For atom type pairs I,J and I != J, where types I and J correspond to
two different element types, mixing is performed by LAMMPS as
described above from values in the potential file.

This pair style does not support the :doc:`pair_modify <pair_modify>`
shift, table, and tail options.

This pair style does not write its information to :doc:`binary restart
files <restart>`, since it is stored in potential files.  Thus, you need
to re-specify the pair_style and pair_coeff commands in an input script
that reads a restart file.

This pair style can only be used via the *pair* keyword of the
:doc:`run_style respa <run_style>` command.  It does not support the
*inner*, *middle*, *outer* keywords.

----------

Restrictions
""""""""""""

This pair style is part of the MANYBODY package.  It is only enabled if
LAMMPS was built with that package.  See the :doc:`Build package
<Build_package>` page for more info.

This pair style requires the :doc:`newton <newton>` setting to be "on"
for pair interactions.

The *shift* keyword is currently not supported for the *tersoff/gpu* and
*tersoff/kk* variants of this pair style.

The tersoff/zbl potential files provided with LAMMPS (see the potentials
directory) are parameterized for :doc:`"metal" units <units>`.  Also the
pair style supports converting potential file parameters on-the-fly
between "metal" and "real" units.  You can use the tersoff/zbl pair
style with any LAMMPS units, but you would need to create your own
tersoff/zbl potential file with coefficients listed in the appropriate
units if your simulation does not use "metal" or "real" units.

Related commands
""""""""""""""""

:doc:`pair_coeff <pair_coeff>`

Default
"""""""

none

----------

.. _zbl-Tersoff_1:

**(Tersoff_1)** J. Tersoff, Phys Rev B, 37, 6991 (1988).

.. _zbl-ZBL:

**(ZBL)** J.F. Ziegler, J.P. Biersack, U. Littmark, 'Stopping and Ranges
of Ions in Matter' Vol 1, 1985, Pergamon Press.

.. _zbl-Albe:

**(Albe)** J. Nord, K. Albe, P. Erhart and K. Nordlund, J. Phys.:
Condens. Matter, 15, 5649(2003).

.. _zbl-Tersoff_2:

**(Tersoff_2)** J. Tersoff, Phys Rev B, 39, 5566 (1989); errata (PRB 41, 3248)
