.. index:: pair_style adp
.. index:: pair_style adp/omp

pair_style adp command
======================

Accelerator Variants: *adp/omp*

Syntax
""""""

.. code-block:: LAMMPS

   pair_style adp

Examples
""""""""

.. code-block:: LAMMPS

   pair_style adp
   pair_coeff * * Ta.adp Ta
   pair_coeff * * ../potentials/AlCu.adp Al Al Cu

Description
"""""""""""

Style *adp* computes pairwise interactions for metals and metal alloys
using the angular dependent potential (ADP) of :ref:`(Mishin) <Mishin>`,
which is a generalization of the :doc:`embedded atom method (EAM) potential <pair_eam>`.  The LAMMPS implementation is discussed in
:ref:`(Singh) <Singh>`.  The total energy Ei of an atom I is given by

.. math::

   E_i            & = F_\alpha \left( \sum_{j\neq i} \rho_\beta (r_{ij}) \right) + \frac{1}{2} \sum_{j\neq i}\phi_{\alpha\beta}(r_{ij})+ \frac{1}{2} \sum_s (\mu_i^s)^2 + \frac{1}{2} \sum_{s,t} (\lambda_i^{st})^2 - \frac{1}{6} \nu_i^2 \\
   \mu_i^s        & = \sum_{j\neq i}u_{\alpha\beta}(r_{ij})r_{ij}^s\\
   \lambda_i^{st} & = \sum_{j\neq i}w_{\alpha\beta}(r_{ij})r_{ij}^sr_{ij}^t\\
   \nu_i          & = \sum_s\lambda_i^{ss}

where :math:`F` is the embedding energy which is a function of the atomic
electron density :math:`\rho`, :math:`\phi` is a pair potential interaction,
:math:`\alpha` and :math:`\beta` are the element types of atoms :math:`I` and
:math:`J`, and :math:`s` and :math:`t = 1,2,3` and refer to the cartesian
coordinates.  The :math:`\mu` and :math:`\lambda` terms represent the dipole
and quadruple distortions of the local atomic environment which extend the
original EAM framework by introducing angular forces.

Note that unlike for other potentials, cutoffs for ADP potentials are
not set in the pair_style or pair_coeff command; they are specified in
the ADP potential files themselves.  Likewise, the ADP potential files
list atomic masses; thus you do not need to use the :doc:`mass <mass>`
command to specify them.

**ADP potentials are available from:**

* The NIST WWW site at http://www.ctcms.nist.gov/potentials.
  Note that ADP potentials obtained from NIST must be converted
  into the extended DYNAMO *setfl* format discussed below.
* The OpenKIM Project at
  `https://openkim.org/browse/models/by-type <https://openkim.org/browse/models/by-type>`_
  provides ADP potentials that can be used directly in LAMMPS with the
  :doc:`kim command <kim_commands>` interface.

----------

Only a single pair_coeff command is used with the *adp* style which
specifies an extended DYNAMO *setfl* file, which contains information
for :math:`M` elements.  These are mapped to LAMMPS atom types by specifying :math:`N`
additional arguments after the filename in the pair_coeff command,
where :math:`N` is the number of LAMMPS atom types:

* filename
* :math:`N` element names = mapping of extended *setfl* elements to atom types

See the :doc:`pair_coeff <pair_coeff>` page for alternate ways to
specify the path for the potential file.

As an example, the potentials/AlCu.adp file, included in the
potentials directory of the LAMMPS distribution, is an extended *setfl*
file which has tabulated ADP values for w elements and their alloy
interactions: Cu and Al.  If your LAMMPS simulation has 4 atoms types
and you want the first 3 to be Al, and the fourth to be Cu, you would use
the following pair_coeff command:

.. code-block:: LAMMPS

   pair_coeff * * AlCu.adp Al Al Al Cu

The first 2 arguments must be \* \* so as to span all LAMMPS atom types.
The first three Al arguments map LAMMPS atom types 1,2,3 to the Al
element in the extended *setfl* file.  The final Cu argument maps
LAMMPS atom type 4 to the Al element in the extended *setfl* file.
Note that there is no requirement that your simulation use all the
elements specified by the extended *setfl* file.

If a mapping value is specified as NULL, the mapping is not performed.
This can be used when an *adp* potential is used as part of the
*hybrid* pair style.  The NULL values are placeholders for atom types
that will be used with other potentials.

*Adp* files in the *potentials* directory of the LAMMPS distribution
have an ".adp" suffix.  A DYNAMO *setfl* file extended for ADP is
formatted as follows.  Basically it is the standard *setfl* format
with additional tabulated functions u and w added to the file after
the tabulated pair potentials.  See the :doc:`pair_eam <pair_eam>`
command for further details on the *setfl* format.

* lines 1,2,3 = comments (ignored)
* line 4: :math:`N_{\text{elements}}` Element1 Element2 ... ElementN
* line 5: :math:`N_{\rho}`, :math:`d_{\rho}`, :math:`N_r`, :math:`d_r`, cutoff

Following the 5 header lines are :math:`N_{\text{elements}}` sections, one for each
element, each with the following format:

* line 1 = atomic number, mass, lattice constant, lattice type (e.g. FCC)
* embedding function :math:`F(\rho)` (:math:`N_{\rho}` values)
* density function :math:`\rho(r)` (:math:`N_r` values)

Following the :math:`N_{\text{elements}}` sections, :math:`N_r` values for each pair potential
:math:`\phi(r)` array are listed for all :math:`i,j` element pairs in the same format
as other arrays.  Since these interactions are symmetric (:math:`i,j = j,i`)
only :math:`\phi` arrays with :math:`i \geq j` are listed, in the following order:

.. math::

   i,j = (1,1), (2,1), (2,2), (3,1), (3,2), (3,3), (4,1), ..., (N_{\text{elements}},N_{\text{elements}}).

The tabulated values for each :math:`\phi` function are listed as
:math:`r*\phi` (in units of eV-Angstroms), since they are for atom pairs, the
same as for :doc:`other EAM files <pair_eam>`.

After the :math:`\phi(r)` arrays, each of the :math:`u(r)` arrays are listed in the
same order with the same assumptions of symmetry.  Directly following
the :math:`u(r)`, the :math:`w(r)` arrays are listed.  Note that :math:`\phi(r)` is the only
array tabulated with a scaling by :math:`r`.

----------

.. include:: accel_styles.rst

----------

Mixing, shift, table, tail correction, restart, rRESPA info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

For atom type pairs I,J and I != J, where types I and J correspond to
two different element types, no special mixing rules are needed, since
the ADP potential files specify alloy interactions explicitly.

This pair style does not support the :doc:`pair_modify <pair_modify>`
shift, table, and tail options.

This pair style does not write its information to :doc:`binary restart files <restart>`, since it is stored in tabulated potential files.
Thus, you need to re-specify the pair_style and pair_coeff commands in
an input script that reads a restart file.

This pair style can only be used via the *pair* keyword of the
:doc:`run_style respa <run_style>` command.  It does not support the
*inner*, *middle*, *outer* keywords.

----------

Restrictions
""""""""""""

This pair style is part of the MANYBODY package.  It is only enabled
if LAMMPS was built with that package.

Related commands
""""""""""""""""

:doc:`pair_coeff <pair_coeff>`, :doc:`pair_eam <pair_eam>`

Default
"""""""

none

----------

.. _Mishin:

**(Mishin)** Mishin, Mehl, and Papaconstantopoulos, Acta Mater, 53, 4029
(2005).

.. _Singh:

**(Singh)** Singh and Warner, Acta Mater, 58, 5797-5805 (2010),
