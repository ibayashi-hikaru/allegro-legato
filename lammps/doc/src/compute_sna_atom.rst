.. index:: compute sna/atom
.. index:: compute snad/atom
.. index:: compute snav/atom
.. index:: compute snap

compute sna/atom command
========================

compute snad/atom command
=========================

compute snav/atom command
=========================

compute snap command
====================

Syntax
""""""

.. parsed-literal::

   compute ID group-ID sna/atom rcutfac rfac0 twojmax R_1 R_2 ... w_1 w_2 ... keyword values ...
   compute ID group-ID snad/atom rcutfac rfac0 twojmax R_1 R_2 ... w_1 w_2 ... keyword values ...
   compute ID group-ID snav/atom rcutfac rfac0 twojmax R_1 R_2 ... w_1 w_2 ... keyword values ...
   compute ID group-ID snap rcutfac rfac0 twojmax R_1 R_2 ... w_1 w_2 ... keyword values ...

* ID, group-ID are documented in :doc:`compute <compute>` command
* sna/atom = style name of this compute command
* rcutfac = scale factor applied to all cutoff radii (positive real)
* rfac0 = parameter in distance to angle conversion (0 < rcutfac < 1)
* twojmax = band limit for bispectrum components (non-negative integer)
* R_1, R_2,... = list of cutoff radii, one for each type (distance units)
* w_1, w_2,... = list of neighbor weights, one for each type
* zero or more keyword/value pairs may be appended
* keyword = *rmin0* or *switchflag* or *bzeroflag* or *quadraticflag* or *chem* or *bnormflag* or *wselfallflag*

  .. parsed-literal::

       *rmin0* value = parameter in distance to angle conversion (distance units)
       *switchflag* value = *0* or *1*
          *0* = do not use switching function
          *1* = use switching function
       *bzeroflag* value = *0* or *1*
          *0* = do not subtract B0
          *1* = subtract B0
       *quadraticflag* value = *0* or *1*
          *0* = do not generate quadratic terms
          *1* = generate quadratic terms
       *chem* values = *nelements* *elementlist*
          *nelements* = number of SNAP elements
          *elementlist* = *ntypes* integers in range [0, *nelements*)
       *bnormflag* value = *0* or *1*
          *0* = do not normalize
          *1* = normalize bispectrum components
       *wselfallflag* value = *0* or *1*
          *0* = self-contribution only for element of central atom
          *1* = self-contribution for all elements

Examples
""""""""

.. code-block:: LAMMPS

   compute b all sna/atom 1.4 0.99363 6 2.0 2.4 0.75 1.0 rmin0 0.0
   compute db all sna/atom 1.4 0.95 6 2.0 1.0
   compute vb all sna/atom 1.4 0.95 6 2.0 1.0
   compute snap all snap 1.4 0.95 6 2.0 1.0
   compute snap all snap 1.0 0.99363 6 3.81 3.83 1.0 0.93 chem 2 0 1

Description
"""""""""""

Define a computation that calculates a set of quantities related to the
bispectrum components of the atoms in a group. These computes are used
primarily for calculating the dependence of energy, force, and stress
components on the linear coefficients in the :doc:`snap pair_style
<pair_snap>`, which is useful when training a SNAP potential to match
target data.

Bispectrum components of an atom are order parameters characterizing the
radial and angular distribution of neighbor atoms. The detailed
mathematical definition is given in the paper by Thompson et
al. :ref:`(Thompson) <Thompson20141>`

The position of a neighbor atom *i'* relative to a central atom *i* is a
point within the 3D ball of radius :math:`R_{ii'}` = *rcutfac*
:math:`(R_i + R_i')`

Bartok et al. :ref:`(Bartok) <Bartok20101>`, proposed mapping this 3D
ball onto the 3-sphere, the surface of the unit ball in a
four-dimensional space.  The radial distance *r* within *R_ii'* is
mapped on to a third polar angle :math:`\theta_0` defined by,

.. math::

  \theta_0 = {\sf rfac0} \frac{r-r_{min0}}{R_{ii'}-r_{min0}} \pi

In this way, all possible neighbor positions are mapped on to a subset
of the 3-sphere.  Points south of the latitude :math:`\theta_0` =
*rfac0* :math:`\pi` are excluded.

The natural basis for functions on the 3-sphere is formed by the
representatives of *SU(2)*, the matrices :math:`U^j_{m,m'}(\theta, \phi,
\theta_0)`.  These functions are better known as :math:`D^j_{m,m'}`, the
elements of the Wigner *D*\ -matrices :ref:`(Meremianin
<Meremianin2006>`, :ref:`Varshalovich <Varshalovich1987>`, :ref:`Mason)
<Mason2009>` The density of neighbors on the 3-sphere can be written as
a sum of Dirac-delta functions, one for each neighbor, weighted by
species and radial distance. Expanding this density function as a
generalized Fourier series in the basis functions, we can write each
Fourier coefficient as

.. math::

  u^j_{m,m'} = U^j_{m,m'}(0,0,0) + \sum_{r_{ii'} < R_{ii'}}{f_c(r_{ii'}) w_{\mu_{i'}} U^j_{m,m'}(\theta_0,\theta,\phi)}

The :math:`w_{\mu_{i'}}` neighbor weights are dimensionless numbers that
depend on :math:`\mu_{i'}`, the SNAP element of atom *i'*, while the
central atom is arbitrarily assigned a unit weight.  The function
:math:`f_c(r)` ensures that the contribution of each neighbor atom goes
smoothly to zero at :math:`R_{ii'}`:

.. math::

  f_c(r)   = & \frac{1}{2}(\cos(\pi \frac{r-r_{min0}}{R_{ii'}-r_{min0}}) + 1), r \leq R_{ii'} \\
           = & 0,  r > R_{ii'}

The expansion coefficients :math:`u^j_{m,m'}` are complex-valued and
they are not directly useful as descriptors, because they are not
invariant under rotation of the polar coordinate frame. However, the
following scalar triple products of expansion coefficients can be shown
to be real-valued and invariant under rotation :ref:`(Bartok)
<Bartok20101>`.

.. math::

   B_{j_1,j_2,j}  =
   \sum_{m_1,m'_1=-j_1}^{j_1}\sum_{m_2,m'_2=-j_2}^{j_2}\sum_{m,m'=-j}^{j} (u^j_{m,m'})^*
   H {\scriptscriptstyle \begin{array}{l} {j} {m} {m'} \\
        {j_1} {m_1} {m'_1} \\
        {j_2} {m_2} {m'_2} \end{array}}
        u^{j_1}_{m_1,m'_1} u^{j_2}_{m_2,m'_2}

The constants :math:`H^{jmm'}_{j_1 m_1 m_{1'},j_2 m_ 2m_{2'}}` are
coupling coefficients, analogous to Clebsch-Gordan coefficients for
rotations on the 2-sphere. These invariants are the components of the
bispectrum and these are the quantities calculated by the compute
*sna/atom*\ . They characterize the strength of density correlations at
three points on the 3-sphere. The j2=0 subset form the power spectrum,
which characterizes the correlations of two points. The lowest-order
components describe the coarsest features of the density function, while
higher-order components reflect finer detail. Each bispectrum component
contains terms that depend on the positions of up to 4 atoms (3
neighbors and the central atom).

Compute *snad/atom* calculates the derivative of the bispectrum
components summed separately for each LAMMPS atom type:

.. math::

   -\sum_{i' \in I} \frac{\partial {B^{i'}_{j_1,j_2,j}  }}{\partial {\bf r}_i}

The sum is over all atoms *i'* of atom type *I*\ .  For each atom *i*,
this compute evaluates the above expression for each direction, each
atom type, and each bispectrum component.  See section below on output
for a detailed explanation.

Compute *snav/atom* calculates the virial contribution due to the
derivatives:

.. math::

  -{\bf r}_i \otimes \sum_{i' \in I} \frac{\partial {B^{i'}_{j_1,j_2,j}}}{\partial {\bf r}_i}

Again, the sum is over all atoms *i'* of atom type *I*\ .  For each atom
*i*, this compute evaluates the above expression for each of the six
virial components, each atom type, and each bispectrum component.  See
section below on output for a detailed explanation.

Compute *snap* calculates a global array containing information related
to all three of the above per-atom computes *sna/atom*, *snad/atom*,
and *snav/atom*\ . The first row of the array contains the summation of
*sna/atom* over all atoms, but broken out by type. The last six rows of
the array contain the summation of *snav/atom* over all atoms, broken
out by type. In between these are 3\*\ *N* rows containing the same
values computed by *snad/atom* (these are already summed over all atoms
and broken out by type). The element in the last column of each row
contains the potential energy, force, or stress, according to the row.
These quantities correspond to the user-specified reference potential
that must be subtracted from the target data when fitting SNAP.  The
potential energy calculation uses the built in compute *thermo_pe*.  The
stress calculation uses a compute called *snap_press* that is
automatically created behind the scenes, according to the following
command:

.. code-block:: LAMMPS

   compute snap_press all pressure NULL virial

See section below on output for a detailed explanation of the data
layout in the global array.

The value of all bispectrum components will be zero for atoms not in
the group. Neighbor atoms not in the group do not contribute to the
bispectrum of atoms in the group.

The neighbor list needed to compute this quantity is constructed each
time the calculation is performed (i.e. each time a snapshot of atoms
is dumped).  Thus it can be inefficient to compute/dump this quantity
too frequently.

The argument *rcutfac* is a scale factor that controls the ratio of
atomic radius to radial cutoff distance.

The argument *rfac0* and the optional keyword *rmin0* define the
linear mapping from radial distance to polar angle :math:`theta_0` on the
3-sphere, given above.

The argument *twojmax* defines which
bispectrum components are generated. See section below on output for a
detailed explanation of the number of bispectrum components and the
ordered in which they are listed.

The keyword *switchflag* can be used to turn off the switching
function :math:`f_c(r)`.

The keyword *bzeroflag* determines whether or not *B0*, the bispectrum
components of an atom with no neighbors, are subtracted from the
calculated bispectrum components. This optional keyword normally only
affects compute *sna/atom*\ . However, when *quadraticflag* is on, it
also affects *snad/atom* and *snav/atom*\ .

The keyword *quadraticflag* determines whether or not the quadratic
combinations of bispectrum quantities are generated.  These are formed
by taking the outer product of the vector of bispectrum components with
itself.  See section below on output for a detailed explanation of the
number of quadratic terms and the ordered in which they are listed.

The keyword *chem* activates the explicit multi-element variant of the
SNAP bispectrum components. The argument *nelements* specifies the
number of SNAP elements that will be handled.  This is followed by
*elementlist*, a list of integers of length *ntypes*, with values in the
range [0, *nelements* ), which maps each LAMMPS type to one of the SNAP
elements.  Note that multiple LAMMPS types can be mapped to the same
element, and some elements may be mapped by no LAMMPS type. However, in
typical use cases (training SNAP potentials) the mapping from LAMMPS
types to elements is one-to-one.

The explicit multi-element variant invoked by the *chem* keyword
partitions the density of neighbors into partial densities for each
chemical element.  This is described in detail in the paper by
:ref:`Cusentino et al. <Cusentino2020>` The bispectrum components are
indexed on ordered triplets of elements:

.. math::

   B_{j_1,j_2,j}^{\kappa\lambda\mu} =
   \sum_{m_1,m'_1=-j_1}^{j_1}\sum_{m_2,m'_2=-j_2}^{j_2}\sum_{m,m'=-j}^{j} (u^{\mu}_{j,m,m'})^*
   H {\scriptscriptstyle \begin{array}{l} {j} {m} {m'} \\
        {j_1} {m_1} {m'_1} \\
        {j_2} {m_2} {m'_2} \end{array}}
        u^{\kappa}_{j_1,m_1,m'_1} u^{\lambda}_{j_2,m_2,m'_2}

where :math:`u^{\mu}_{j,m,m'}` is an expansion coefficient for the partial density of neighbors
of element :math:`\mu`

.. math::

  u^{\mu}_{j,m,m'} =  w^{self}_{\mu_{i}\mu} U^{j,m,m'}(0,0,0) + \sum_{r_{ii'} < R_{ii'}}{\delta_{\mu\mu_{i'}}f_c(r_{ii'}) w_{\mu_{i'}} U^{j,m,m'}(\theta_0,\theta,\phi)}

where :math:`w^{self}_{\mu_{i}\mu}` is the self-contribution, which is
either 1 or 0 (see keyword *wselfallflag* below),
:math:`\delta_{\mu\mu_{i'}}` indicates that the sum is only over
neighbor atoms of element :math:`\mu`, and all other quantities are the
same as those appearing in the original equation for :math:`u^j_{m,m'}`
given above.

The keyword *wselfallflag* defines the rule used for the
self-contribution.  If *wselfallflag* is on, then
:math:`w^{self}_{\mu_{i}\mu}` = 1. If it is off then
:math:`w^{self}_{\mu_{i}\mu}` = 0, except in the case of
:math:`{\mu_{i}=\mu}`, when :math:`w^{self}_{\mu_{i}\mu}` = 1.  When the
*chem* keyword is not used, this keyword has no effect.

The keyword *bnormflag* determines whether or not the bispectrum
component :math:`B_{j_1,j_2,j}` is divided by a factor of :math:`2j+1`.
This normalization simplifies force calculations because of the
following symmetry relation

.. math::

 \frac{B_{j_1,j_2,j}}{2j+1} = \frac{B_{j,j_2,j_1}}{2j_1+1} = \frac{B_{j_1,j,j_2}}{2j_2+1}

This option is typically used in conjunction with the *chem* keyword,
and LAMMPS will generate a warning if both *chem* and *bnormflag*
are not both set or not both unset.

.. note::

   If you have a bonded system, then the settings of :doc:`special_bonds
   <special_bonds>` command can remove pairwise interactions between
   atoms in the same bond, angle, or dihedral.  This is the default
   setting for the :doc:`special_bonds <special_bonds>` command, and
   means those pairwise interactions do not appear in the neighbor list.
   Because this fix uses the neighbor list, it also means those pairs
   will not be included in the calculation.  One way to get around this,
   is to write a dump file, and use the :doc:`rerun <rerun>` command to
   compute the bispectrum components for snapshots in the dump file.
   The rerun script can use a :doc:`special_bonds <special_bonds>`
   command that includes all pairs in the neighbor list.

----------

Output info
"""""""""""

Compute *sna/atom* calculates a per-atom array, each column
corresponding to a particular bispectrum component.  The total number of
columns and the identity of the bispectrum component contained in each
column depend of the value of *twojmax*, as described by the following
piece of python code:

.. parsed-literal::

   for j1 in range(0,twojmax+1):
       for j2 in range(0,j1+1):
           for j in range(j1-j2,min(twojmax,j1+j2)+1,2):
               if (j>=j1): print j1/2.,j2/2.,j/2.

For even twojmax = 2(*m*\ -1), :math:`K = m(m+1)(2m+1)/6`, the *m*\ -th pyramidal number. For odd twojmax = 2 *m*\ -1, :math:`K = m(m+1)(m+2)/3`, twice the *m*\ -th tetrahedral number.

.. note::

   the *diagonal* keyword allowing other possible choices
   for the number of bispectrum components was removed in 2019,
   since all potentials use the value of 3, corresponding to the
   above set of bispectrum components.

Compute *snad/atom* evaluates a per-atom array. The columns are arranged
into *ntypes* blocks, listed in order of atom type *I*\ .  Each block
contains three sub-blocks corresponding to the *x*, *y*, and *z*
components of the atom position.  Each of these sub-blocks contains *K*
columns for the *K* bispectrum components, the same as for compute
*sna/atom*

Compute *snav/atom* evaluates a per-atom array. The columns are arranged
into *ntypes* blocks, listed in order of atom type *I*\ .  Each block
contains six sub-blocks corresponding to the *xx*, *yy*, *zz*,
*yz*, *xz*, and *xy* components of the virial tensor in Voigt
notation.  Each of these sub-blocks contains *K* columns for the *K*
bispectrum components, the same as for compute *sna/atom*

Compute *snap* evaluates a global array.  The columns are arranged into
*ntypes* blocks, listed in order of atom type *I*\ . Each block contains
one column for each bispectrum component, the same as for compute
*sna/atom*\ . A final column contains the corresponding energy, force
component on an atom, or virial stress component. The rows of the array
appear in the following order:

* 1 row: *sna/atom* quantities summed for all atoms of type *I*
* 3\*\ *N* rows: *snad/atom* quantities, with derivatives w.r.t. x, y, and z coordinate of atom *i* appearing in consecutive rows. The atoms are sorted based on atom ID.
* 6 rows: *snav/atom* quantities summed for all atoms of type *I*

For example, if *K* =30 and ntypes=1, the number of columns in the
per-atom arrays generated by *sna/atom*, *snad/atom*, and
*snav/atom* are 30, 90, and 180, respectively. With *quadratic* value=1,
the numbers of columns are 930, 2790, and 5580, respectively.  The
number of columns in the global array generated by *snap* are 31, and
931, respectively, while the number of rows is 1+3\*\ *N*\ +6, where *N*
is the total number of atoms.

If the *quadratic* keyword value is set to 1, then additional columns
are generated, corresponding to the products of all distinct pairs of
bispectrum components. If the number of bispectrum components is *K*,
then the number of distinct pairs is *K*\ (\ *K*\ +1)/2.  For compute
*sna/atom* these columns are appended to existing *K* columns.  The
ordering of quadratic terms is upper-triangular, (1,1),(1,2)...(1,\ *K*\
),(2,1)...(\ *K*\ -1,\ *K*\ -1),(\ *K*\ -1,\ *K*\ ),(\ *K*,\ *K*\ ).
For computes *snad/atom* and *snav/atom* each set of *K*\ (\ *K*\ +1)/2
additional columns is inserted directly after each of sub-block of
linear terms i.e. linear and quadratic terms are contiguous.  So the
nesting order from inside to outside is bispectrum component, linear
then quadratic, vector/tensor component, type.

If the *chem* keyword is used, then the data is arranged into
:math:`N_{elem}^3` sub-blocks, each sub-block corresponding to a
particular chemical labeling :math:`\kappa\lambda\mu` with the last
label changing fastest.  Each sub-block contains *K* bispectrum
components. For the purposes of handling contributions to force, virial,
and quadratic combinations, these :math:`N_{elem}^3` sub-blocks are
treated as a single block of :math:`K N_{elem}^3` columns.

These values can be accessed by any command that uses per-atom values
from a compute as input.  See the :doc:`Howto output <Howto_output>` doc
page for an overview of LAMMPS output options. To see how this command
can be used within a Python workflow to train SNAP potentials, see the
examples in `FitSNAP <https://github.com/FitSNAP/FitSNAP>`_.

Restrictions
""""""""""""

These computes are part of the ML-SNAP package.  They are only enabled
if LAMMPS was built with that package.  See the :doc:`Build package
<Build_package>` page for more info.

Related commands
""""""""""""""""

:doc:`pair_style snap <pair_snap>`

Default
"""""""

The optional keyword defaults are *rmin0* = 0,
*switchflag* = 1, *bzeroflag* = 1, *quadraticflag* = 0,
*bnormflag* = 0, *wselfallflag* = 0

----------

.. _Thompson20141:

**(Thompson)** Thompson, Swiler, Trott, Foiles, Tucker, under review, preprint
available at `arXiv:1409.3880 <http://arxiv.org/abs/1409.3880>`_

.. _Bartok20101:

**(Bartok)** Bartok, Payne, Risi, Csanyi, Phys Rev Lett, 104, 136403 (2010).

.. _Meremianin2006:

**(Meremianin)** Meremianin, J. Phys. A,  39, 3099 (2006).

.. _Varshalovich1987:

**(Varshalovich)** Varshalovich, Moskalev, Khersonskii, Quantum Theory
of Angular Momentum, World Scientific, Singapore (1987).

.. _Mason2009:

**(Mason)** J. K. Mason, Acta Cryst A65, 259 (2009).

.. _Cusentino2020:

**(Cusentino)** Cusentino, Wood, and Thompson, J Phys Chem A, xxx, xxxxx, (2020)
