.. index:: pair_style polymorphic

pair_style polymorphic command
==============================

Syntax
""""""

.. code-block:: LAMMPS

   pair_style polymorphic

style = *polymorphic*

Examples
""""""""

.. code-block:: LAMMPS

   pair_style polymorphic
   pair_coeff * * FeCH_BOP_I.poly Fe C H
   pair_coeff * * TlBr_msw.poly Tl Br
   pair_coeff * * CuTa_eam.poly Cu Ta
   pair_coeff * * GaN_tersoff.poly Ga N
   pair_coeff * * GaN_sw.poly Ga N

Description
"""""""""""

The *polymorphic* pair style computes a 3-body free-form potential
(:ref:`Zhou3 <Zhou3>`) for the energy E of a system of atoms as

.. math::

   E & = \frac{1}{2}\sum_{i=1}^{i=N}\sum_{j=1}^{j=N}\left[\left(1-\delta_{ij}\right)\cdot U_{IJ}\left(r_{ij}\right)-\left(1-\eta_{ij}\right)\cdot F_{IJ}\left(X_{ij}\right)\cdot V_{IJ}\left(r_{ij}\right)\right] \\
   X_{ij} & = \sum_{k=i_1,k\neq j}^{i_N}W_{IK}\left(r_{ik}\right)\cdot G_{JIK}\left(\cos\theta_{jik}\right)\cdot P_{JIK}\left(\Delta r_{jik}\right) \\
   \Delta r_{jik} & = r_{ij}-\xi_{IJ}\cdot r_{ik}

where I, J, K represent species of atoms i, j, and k, :math:`i_1, ...,
i_N` represents a list of *i*\ 's neighbors, :math:`\delta_{ij}` is a
Dirac constant (i.e., :math:`\delta_{ij} = 1` when :math:`i = j`, and
:math:`\delta_{ij} = 0` otherwise), :math:`\eta_{ij}` is similar
constant that can be set either to :math:`\eta_{ij} = \delta_{ij}` or
:math:`\eta_{ij} = 1 - \delta_{ij}` depending on the potential type,
:math:`U_{IJ}(r_{ij})`, :math:`V_{IJ}(r_{ij})`, :math:`W_{IK}(r_{ik})`
are pair functions, :math:`G_{JIK}(\cos\theta_{jik})` is an angular
function, :math:`P_{JIK}(\Delta r_{jik})` is a function of atomic
spacing differential :math:`\Delta r_{jik} = r_{ij} - \xi_{IJ} \cdot
r_{ik}` with :math:`\xi_{IJ}` being a pair-dependent parameter, and
:math:`F_{IJ}(X_{ij})` is a function of the local environment variable
:math:`X_{ij}`. This generic potential is fully defined once the
constants :math:`\eta_{ij}` and :math:`\xi_{IJ}`, and the six functions
:math:`U_{IJ}(r_{ij})`, :math:`V_{IJ}(r_{ij})`, :math:`W_{IK}(r_{ik})`,
:math:`G_{JIK}(\cos\theta_{jik})`, :math:`P_{JIK}(\Delta r_{jik})`, and
:math:`F_{IJ}(X_{ij})` are given. Here LAMMPS uses a global parameter
:math:`\eta` to represent :math:`\eta_{ij}`. When :math:`\eta = 1`,
:math:`\eta_{ij} = 1 - \delta_{ij}`, otherwise :math:`\eta_{ij} =
\delta_{ij}`. Additionally, :math:`\eta = 3` indicates that the function
:math:`P_{JIK}(\Delta r)` depends on species I, J and K, otherwise
:math:`P_{JIK}(\Delta r) = P_{IK}(\Delta r)` only depends on species I
and K. Note that these six functions are all one dimensional, and hence
can be provided in a tabular form. This allows users to design different
potentials solely based on a manipulation of these functions. For
instance, the potential reduces to a Stillinger-Weber potential
(:ref:`SW <SW>`) if we set

.. math::

   \eta_{ij} & = \delta_{ij} (\eta = 2~or~\eta = 0),\xi_{IJ}=0 \\
   U_{IJ}\left(r\right) & = A_{IJ}\cdot\epsilon_{IJ}\cdot \left(\frac{\sigma_{IJ}}{r}\right)^q\cdot \left[B_{IJ}\cdot \left(\frac{\sigma_{IJ}}{r}\right)^{p-q}-1\right]\cdot exp\left(\frac{\sigma_{IJ}}{r-a_{IJ}\cdot \sigma_{IJ}}\right) \\
   V_{IJ}\left(r\right) & = \sqrt{\lambda_{IJ}\cdot \epsilon_{IJ}}\cdot exp\left(\frac{\gamma_{IJ}\cdot \sigma_{IJ}}{r-a_{IJ}\cdot \sigma_{IJ}}\right) \\
   F_{IJ}\left(X\right) & = -X \\
   P_{JIK}\left(\Delta r\right) & = P_{IK}\left(\Delta r\right) = 1 \\
   W_{IJ}\left(r\right) & = \sqrt{\lambda_{IJ}\cdot \epsilon_{IJ}}\cdot exp\left(\frac{\gamma_{IJ}\cdot \sigma_{IJ}}{r-a_{IJ}\cdot \sigma_{IJ}}\right) \\
   G_{JIK}\left(\cos\theta\right) & = \left(\cos\theta+\frac{1}{3}\right)^2

The potential reduces to a Tersoff potential (:ref:`Tersoff <Tersoff>`
or :ref:`Albe <poly-Albe>`) if we set

.. math::

   \eta_{ij} & = \delta_{ij} (\eta = 2~or~\eta = 0),\xi_{IJ}=1 \\
   U_{IJ}\left(r\right) & = \frac{D_{e,IJ}}{S_{IJ}-1}\cdot exp\left[-\beta_{IJ}\sqrt{2S_{IJ}}\left(r-r_{e,IJ}\right)\right]\cdot f_{c,IJ}\left(r\right) \\
   V_{IJ}\left(r\right) & = \frac{S_{IJ}\cdot D_{e,IJ}}{S_{IJ}-1}\cdot exp\left[-\beta_{IJ}\sqrt{\frac{2}{S_{IJ}}}\left(r-r_{e,IJ}\right)\right]\cdot f_{c,IJ}\left(r\right) \\
   F_{IJ}\left(X\right) & = \left(1+X\right)^{-\frac{1}{2}} \\
   P_{JIK}\left(\Delta r\right) & = P_{IK}\left(\Delta r\right) = exp\left(2\mu_{IK}\cdot \Delta r\right) \\
   W_{IJ}\left(r\right) & = f_{c,IJ}\left(r\right) \\
   G_{JIK}\left(\cos\theta\right) & = \gamma_{IK}\left[1+\frac{c_{IK}^2}{d_{IK}^2}-\frac{c_{IK}^2}{d_{IK}^2+\left(h_{IK}+\cos\theta\right)^2}\right]

where

.. math::

   f_{c,IJ}\left(r\right)=\left\{\begin{array}{l}
   1, r\leq R_{IJ}-D_{IJ} \\
   \frac{1}{2}+\frac{1}{2}cos\left[\frac{\pi\left(r+D_{IJ}-R_{IJ}\right)}{2D_{IJ}}\right], R_{IJ}-D_{IJ} < r < R_{IJ}+D_{IJ} \\
   0, r \geq R_{IJ}+D_{IJ}
   \end{array}\right.

The potential reduces to a modified Stillinger-Weber potential
(:ref:`Zhou3 <Zhou3>`) if we set

.. math::

   \eta_{ij} & = \delta_{ij} (\eta = 2~or~\eta = 0),\xi_{IJ}=0 \\
   U_{IJ}\left(r\right) & = \varphi_{R,IJ}\left(r\right)-\varphi_{A,IJ}\left(r\right) \\
   V_{IJ}\left(r\right) & = u_{IJ}\left(r\right) \\
   F_{IJ}\left(X\right) & = -X \\
   P_{JIK}\left(\Delta r\right) & = P_{IK}\left(\Delta r\right) = 1 \\
   W_{IJ}\left(r\right) & = u_{IJ}\left(r\right) \\
   G_{JIK}\left(\cos\theta\right) & = g_{JIK}\left(\cos\theta\right)

The potential reduces to a Rockett-Tersoff potential (:ref:`Wang3
<Wang3>`) if we set

.. math::

   \eta_{ij} & = \delta_{ij} (\eta = 2~or~\eta = 0),\xi_{IJ}=1 \\
   U_{IJ}\left(r\right) & = A_{IJ}exp\left(-\lambda_{1,IJ}\cdot r\right)f_{c,IJ}\left(r\right)f_{ca,IJ}\left(r\right) \\
   V_{IJ}\left(r\right) & = \left\{\begin{array}{l}B_{IJ}exp\left(-\lambda_{2,IJ}\cdot r\right)f_{c,IJ}\left(r\right)+ \\ A_{IJ}exp\left(-\lambda_{1,IJ}\cdot r\right)f_{c,IJ}\left(r\right) \left[1-f_{ca,IJ}\left(r\right)\right]\end{array} \right\} \\
   F_{IJ}\left(X\right) & = \left[1+\left(\beta_{IJ}X\right)^{n_{IJ}}\right]^{-\frac{1}{2n_{IJ}}} \\
   P_{JIK}\left(\Delta r\right) & = P_{IK}\left(\Delta r\right) = exp\left(\lambda_{3,IK}\cdot \Delta r^3\right) \\
   W_{IJ}\left(r\right) & = f_{c,IJ}\left(r\right) \\
   G_{JIK}\left(\cos\theta\right) & = 1+\frac{c_{IK}^2}{d_{IK}^2}-\frac{c_{IK}^2}{d_{IK}^2+\left(h_{IK}+\cos\theta\right)^2}

where :math:`f_{ca,IJ}(r)` is similar to the :math:`f_{c,IJ}(r)` defined
above:

.. math::

   f_{ca,IJ}\left(r\right)=\left\{\begin{array}{l}
   1, r\leq R_{a,IJ}-D_{a,IJ} \\
   \frac{1}{2}+\frac{1}{2}cos\left[\frac{\pi\left(r+D_{a,IJ}-R_{a,IJ}\right)}{2D_{a,IJ}}\right], R_{a,IJ}-D_{a,IJ} < r < R_{a,IJ}+D_{a,IJ} \\
   0, r \geq R_{a,IJ}+D_{a,IJ}
   \end{array}\right.

The potential becomes the embedded atom method (:ref:`Daw <poly-Daw>`)
if we set

.. math::

   \eta_{ij} & = 1-\delta_{ij} (\eta = 1),\xi_{IJ}=0 \\
   U_{IJ}\left(r\right) & = \phi_{IJ}\left(r\right) \\
   V_{IJ}\left(r\right) & = 1 \\
   F_{II}\left(X\right) & = -2F_I\left(X\right) \\
   P_{JIK}\left(\Delta r\right) & = P_{IK}\left(\Delta r\right) = 1 \\
   W_{IJ}\left(r\right) & = f_{J}\left(r\right) \\
   G_{JIK}\left(\cos\theta\right) & = 1

In the embedded atom method case, :math:`\phi_{IJ}(r)` is the pair
energy, :math:`F_I(X)` is the embedding energy, *X* is the local
electron density, and :math:`f_J(r)` is the atomic electron density
function.

The potential reduces to another type of Tersoff potential (:ref:`Zhou4
<Zhou4>`) if we set

.. math::

   \eta_{ij} & = \delta_{ij} (\eta = 3),\xi_{IJ}=1 \\
   U_{IJ}\left(r\right) & = \frac{D_{e,IJ}}{S_{IJ}-1}\cdot exp\left[-\beta_{IJ}\sqrt{2S_{IJ}}\left(r-r_{e,IJ}\right)\right]\cdot f_{c,IJ}\left(r\right) \cdot T_{IJ}\left(r\right)+V_{ZBL,IJ}\left(r\right)\left[1-T_{IJ}\left(r\right)\right] \\
   V_{IJ}\left(r\right) & = \frac{S_{IJ}\cdot D_{e,IJ}}{S_{IJ}-1}\cdot exp\left[-\beta_{IJ}\sqrt{\frac{2}{S_{IJ}}}\left(r-r_{e,IJ}\right)\right]\cdot f_{c,IJ}\left(r\right) \cdot T_{IJ}\left(r\right) \\
   F_{IJ}\left(X\right) & = \left(1+X\right)^{-\frac{1}{2}} \\
   P_{JIK}\left(\Delta r\right) & = \omega_{JIK} \cdot exp\left(\alpha_{JIK}\cdot \Delta r\right) \\
   W_{IJ}\left(r\right) & = f_{c,IJ}\left(r\right) \\
   G_{JIK}\left(\cos\theta\right) & = \gamma_{JIK}\left[1+\frac{c_{JIK}^2}{d_{JIK}^2}-\frac{c_{JIK}^2}{d_{JIK}^2+\left(h_{JIK}+\cos\theta\right)^2}\right] \\
   T_{IJ}\left(r\right) & = \frac{1}{1+exp\left[-b_{f,IJ}\left(r-r_{f,IJ}\right)\right]} \\
   V_{ZBL,IJ}\left(r\right) & = 14.4 \cdot \frac{Z_I \cdot Z_J}{r}\sum_{k=1}^{4}\mu_k \cdot exp\left[-\nu_k \left(Z_I^{0.23}+Z_J^{0.23}\right) r\right]

where :math:`f_{c,IJ}(r)` is the same as defined above. This Tersoff
potential differs from the one above because the :math:`P_{JIK}(\Delta
r)` function is now dependent on all three species I, J, and K.

If the tabulated functions are created using the parameters of
Stillinger-Weber, Tersoff, and EAM potentials, the polymorphic pair
style will produce the same global properties (energies and stresses)
and the same forces as the :doc:`sw <pair_sw>`, :doc:`tersoff
<pair_tersoff>`, and :doc:`eam <pair_eam>` pair styles. The polymorphic
pair style also produces the same per-atom properties (energies and
stresses) as the corresponding :doc:`tersoff <pair_tersoff>` and
:doc:`eam <pair_eam>` pair styles. However, due to a different
partitioning of global properties to per-atom properties, the
polymorphic pair style will produce different per-atom properties
(energies and stresses) as the :doc:`sw <pair_sw>` pair style. This does
not mean that polymorphic pair style is different from the sw pair
style. It just means that the definitions of the atom energies and atom
stresses are different.

Only a single pair_coeff command is used with the polymorphic pair style
which specifies a potential file for all needed elements.  These are
mapped to LAMMPS atom types by specifying N additional arguments after
the filename in the pair_coeff command, where N is the number of LAMMPS
atom types:

* filename
* N element names = mapping of polymorphic potential elements to atom types

See the pair_coeff page for alternate ways to specify the path for
the potential file. Several files for polymorphic potentials are
included in the potentials directory of the LAMMPS distribution. They
have a "poly" suffix.

As an example, imagine the GaN_tersoff.poly file has tabulated functions
for Ga-N tersoff potential. If your LAMMPS simulation has 4 atom types and
you want the first 3 to be Ga, and the fourth to be N, you would use the
following pair_coeff command:

.. code-block:: LAMMPS

   pair_coeff * * GaN_tersoff.poly Ga Ga Ga N

The first two arguments must be \* \* to span all pairs of LAMMPS atom
types. The first three Ga arguments map LAMMPS atom types 1,2,3 to the
Ga element in the polymorphic file. The final N argument maps LAMMPS
atom type 4 to the N element in the polymorphic file. If a mapping value
is specified as NULL, the mapping is not performed. This can be used
when an polymorphic potential is used as part of the hybrid pair
style. The NULL values are placeholders for atom types that will be used
with other potentials.

Potential files in the potentials directory of the LAMMPS distribution
have a ".poly" suffix. At the beginning of the files, an unlimited
number of lines starting with '#' are used to describe the potential and
are ignored by LAMMPS. The next line lists two numbers:

.. parsed-literal::

   ntypes eta

Here *ntypes* represent total number of species defined in the potential
file, :math:`\eta = 1` reduces to embedded atom method, :math:`\eta = 3`
assumes a three species dependent :math:`P_{JIK}(\Delta r)` function,
and all other values of :math:`\eta` assume a two species dependent
:math:`P_{JK}(\Delta r)` function. The value of *ntypes* must equal the
total number of different species defined in the pair_coeff command. The
next *ntypes* lines each lists two numbers and a character string
representing atomic number, atomic mass, and name of the species of the
ntypes elements:

.. parsed-literal::

   atomic-number atomic-mass element-name(1)
   atomic-number atomic-mass element-name(2)
   ...
   atomic-number atomic-mass element-name(ntypes)

The next line contains four numbers:

.. parsed-literal::

   nr ntheta nx xmax

Here nr is total number of tabular points for radial functions U, V, W,
P, ntheta is total number of tabular points for the angular function G,
nx is total number of tabular points for the function F, xmax is a
maximum value of the argument of function F. Note that the pair
functions :math:`U_{IJ}(r)`, :math:`V_{IJ}(r)`, :math:`W_{IJ}(r)` are
uniformly tabulated between 0 and cutoff distance of the IJ pair,
:math:`G_{JIK}(\cos\theta)` is uniformly tabulated between -1 and 1,
:math:`P_{JIK}(\Delta r)` is uniformly tabulated between -rcmax and
rcmax where rcmax is the maximum cutoff distance of all pairs, and
:math:`F_{IJ}(X)` is uniformly tabulated between 0 and xmax. Linear
extrapolation is assumed if actual simulations exceed these ranges.

The next ntypes\*(ntypes+1)/2 lines contain two numbers:

.. parsed-literal::

   cut xi(1)
   cut xi(2)
   ...
   cut xi(ntypes\*(ntypes+1)/2)

Here cut means the cutoff distance of the pair functions, "xi" is
:math:`\xi` as defined in the potential functions above. The
ntypes\*(ntypes+1)/2 lines are related to the pairs according to the
sequence of first ii (self) pairs, i = 1, 2, ..., ntypes, and then ij
(cross) pairs, i = 1, 2, ..., ntypes-1, and j = i+1, i+2, ..., ntypes
(i.e., the sequence of the ij pairs follows 11, 22, ..., 12, 13, 14,
..., 23, 24, ...).

In the final blocks of the potential file, U, V, W, P, G, and F
functions are listed sequentially. First, U functions are given for each
of the ntypes\*(ntypes+1)/2 pairs according to the sequence described
above. For each of the pairs, nr values are listed. Next, similar arrays
are given for V and W functions. If P functions depend only on pair
species, i.e., :math:`\eta \neq 3`, then P functions are also listed the
same way the next. If P functions depend on three species, i.e.,
:math:`\eta = 3`, then P functions are listed for all the
ntypes*ntypes*ntypes IJK triplets in a natural sequence I from 1 to
ntypes, J from 1 to ntypes, and K from 1 to ntypes (i.e., IJK = 111,
112, 113, ..., 121, 122, 123 ..., 211, 212, ...). Next, G functions are
listed for all the ntypes*ntypes*ntypes IJK triplets similarly. For each
of the G functions, ntheta values are listed. Finally, F functions are
listed for all the ntypes*(ntypes+1)/2 pairs in the same sequence as
described above.  For each of the F functions, nx values are listed.

Mixing, shift, table, tail correction, restart, rRESPA info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

This pair style does not support the :doc:`pair_modify <pair_modify>`
shift, table, and tail options.

This pair style does not write their information to :doc:`binary restart
files <restart>`, since it is stored in potential files. Thus, you need
to re-specify the pair_style and pair_coeff commands in an input script
that reads a restart file.

----------

Restrictions
""""""""""""

If using create_atoms command, atomic masses must be defined in the
input script. If using read_data, atomic masses must be defined in the
atomic structure data file.

This pair style is part of the MANYBODY package. It is only enabled if
LAMMPS was built with that package. See the :doc:`Build package
<Build_package>` page for more info.

This pair potential requires the :doc:`newton <newton>` setting to be
"on" for pair interactions.

The potential files provided with LAMMPS (see the potentials directory)
are parameterized for metal :doc:`units <units>`. You can use any LAMMPS
units, but you would need to create your own potential files.

Related commands
""""""""""""""""

:doc:`pair_coeff <pair_coeff>`

----------

.. _Zhou3:

**(Zhou3)** X. W. Zhou, M. E. Foster, R. E. Jones, P. Yang, H. Fan, and F. P. Doty, J. Mater. Sci. Res., 4, 15 (2015).

.. _Zhou4:

**(Zhou4)** X. W. Zhou, M. E. Foster, J. A. Ronevich, and C. W. San Marchi, J. Comp. Chem., 41, 1299 (2020).

.. _SW:

**(SW)** F. H. Stillinger, and T. A. Weber, Phys. Rev. B, 31, 5262 (1985).

.. _Tersoff:

**(Tersoff)** J. Tersoff, Phys. Rev. B, 39, 5566 (1989).

.. _poly-Albe:

**(Albe)** K. Albe, K. Nordlund, J. Nord, and A. Kuronen, Phys. Rev. B, 66, 035205 (2002).

.. _Wang3:

**(Wang)** J. Wang, and A. Rockett, Phys. Rev. B, 43, 12571 (1991).

.. _poly-Daw:

**(Daw)** M. S. Daw, and M. I. Baskes, Phys. Rev. B, 29, 6443 (1984).
