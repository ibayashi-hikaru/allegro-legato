.. index:: pair_style oxdna2/excv
.. index:: pair_style oxdna2/stk
.. index:: pair_style oxdna2/hbond
.. index:: pair_style oxdna2/xstk
.. index:: pair_style oxdna2/coaxstk
.. index:: pair_style oxdna2/dh

pair_style oxdna2/excv command
==============================

pair_style oxdna2/stk command
=============================

pair_style oxdna2/hbond command
===============================

pair_style oxdna2/xstk command
==============================

pair_style oxdna2/coaxstk command
=================================

pair_style oxdna2/dh command
============================

Syntax
""""""

.. code-block:: LAMMPS

   pair_style style1

   pair_coeff * * style2 args

* style1 = *hybrid/overlay oxdna2/excv oxdna2/stk oxdna2/hbond oxdna2/xstk oxdna2/coaxstk oxdna2/dh*

* style2 = *oxdna2/excv* or *oxdna2/stk* or *oxdna2/hbond* or *oxdna2/xstk* or *oxdna2/coaxstk* or *oxdna2/dh*
* args = list of arguments for these particular styles

.. parsed-literal::

     *oxdna2/stk* args = seq T xi kappa 6.0 0.4 0.9 0.32 0.75 1.3 0 0.8 0.9 0 0.95 0.9 0 0.95 2.0 0.65 2.0 0.65
       seq = seqav (for average sequence stacking strength) or seqdep (for sequence-dependent stacking strength)
       T = temperature (oxDNA units, 0.1 = 300 K)
       xi = 1.3523 (temperature-independent coefficient in stacking strength)
       kappa = 2.6717 (coefficient of linear temperature dependence in stacking strength)
     *oxdna2/hbond* args = seq eps 8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7 1.5 0 0.7 0.46 3.141592653589793 0.7 4.0 1.5707963267948966 0.45 4.0 1.5707963267948966 0.45
       seq = seqav (for average sequence base-pairing strength) or seqdep (for sequence-dependent base-pairing strength)
       eps = 1.0678 (between base pairs A-T and C-G) or 0 (all other pairs)
     *oxdna2/dh* args = T rhos qeff
       T = temperature (oxDNA units, 0.1 = 300 K)
       rhos = salt concentration (mole per litre)
       qeff = 0.815 (effective charge in elementary charges)

Examples
""""""""

.. code-block:: LAMMPS

   pair_style hybrid/overlay oxdna2/excv oxdna2/stk oxdna2/hbond oxdna2/xstk oxdna2/coaxstk oxdna2/dh
   pair_coeff * * oxdna2/excv    2.0 0.7 0.675 2.0 0.515 0.5 2.0 0.33 0.32
   pair_coeff * * oxdna2/stk     seqdep 0.1 1.3523 2.6717 6.0 0.4 0.9 0.32 0.75 1.3 0 0.8 0.9 0 0.95 0.9 0 0.95 2.0 0.65 2.0 0.65
   pair_coeff * * oxdna2/hbond   seqdep 0.0 8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7 1.5 0 0.7 0.46 3.141592653589793 0.7 4.0 1.5707963267948966 0.45 4.0 1.5707963267948966 0.45
   pair_coeff 1 4 oxdna2/hbond   seqdep 1.0678 8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7 1.5 0 0.7 0.46 3.141592653589793 0.7 4.0 1.5707963267948966 0.45 4.0 1.5707963267948966 0.45
   pair_coeff 2 3 oxdna2/hbond   seqdep 1.0678 8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7 1.5 0 0.7 0.46 3.141592653589793 0.7 4.0 1.5707963267948966 0.45 4.0 1.5707963267948966 0.45
   pair_coeff * * oxdna2/xstk    47.5 0.575 0.675 0.495 0.655 2.25 0.791592653589793 0.58 1.7 1.0 0.68 1.7 1.0 0.68 1.5 0 0.65 1.7 0.875 0.68 1.7 0.875 0.68
   pair_coeff * * oxdna2/coaxstk 58.5 0.4 0.6 0.22 0.58 2.0 2.891592653589793 0.65 1.3 0 0.8 0.9 0 0.95 0.9 0 0.95 40.0 3.116592653589793
   pair_coeff * * oxdna2/dh      0.1 0.5 0.815

Description
"""""""""""

The *oxdna2* pair styles compute the pairwise-additive parts of the oxDNA force field
for coarse-grained modelling of DNA. The effective interaction between the nucleotides consists of potentials for the
excluded volume interaction *oxdna2/excv*, the stacking *oxdna2/stk*, cross-stacking *oxdna2/xstk*
and coaxial stacking interaction *oxdna2/coaxstk*, electrostatic Debye-Hueckel interaction *oxdna2/dh*
as well as the hydrogen-bonding interaction *oxdna2/hbond* between complementary pairs of nucleotides on
opposite strands. Average sequence or sequence-dependent stacking and base-pairing strengths
are supported :ref:`(Sulc) <Sulc2>`. Quasi-unique base-pairing between nucleotides can be achieved by using
more complementary pairs of atom types like 5-8 and 6-7, 9-12 and 10-11, 13-16 and 14-15, etc.
This prevents the hybridization of in principle complementary bases within Ntypes/4 bases
up and down along the backbone.

The exact functional form of the pair styles is rather complex.
The individual potentials consist of products of modulation factors,
which themselves are constructed from a number of more basic potentials
(Morse, Lennard-Jones, harmonic angle and distance) as well as quadratic smoothing and modulation terms.
We refer to :ref:`(Snodin) <Snodin2>` and the original oxDNA publications :ref:`(Ouldridge-DPhil) <Ouldridge-DPhil2>`
and  :ref:`(Ouldridge) <Ouldridge2>` for a detailed description of the oxDNA2 force field.

.. note::

   These pair styles have to be used together with the related oxDNA2 bond style
   *oxdna2/fene* for the connectivity of the phosphate backbone (see also documentation of
   :doc:`bond_style oxdna2/fene <bond_oxdna>`). Most of the coefficients
   in the above example have to be kept fixed and cannot be changed without reparameterizing the entire model.
   Exceptions are the first four coefficients after *oxdna2/stk* (seq=seqdep, T=0.1, xi=1.3523 and kappa=2.6717 in the above example),
   the first coefficient after *oxdna2/hbond* (seq=seqdep in the above example) and the three coefficients
   after *oxdna2/dh* (T=0.1, rhos=0.5, qeff=0.815 in the above example). When using a Langevin thermostat
   e.g. through :doc:`fix langevin <fix_langevin>` or :doc:`fix nve/dotc/langevin <fix_nve_dotc_langevin>`
   the temperature coefficients have to be matched to the one used in the fix.

.. note::

   These pair styles have to be used with the *atom_style hybrid bond ellipsoid oxdna*
   (see documentation of :doc:`atom_style <atom_style>`). The *atom_style oxdna*
   stores the 3'-to-5' polarity of the nucleotide strand, which is set through
   the bond topology in the data file. The first (second) atom in a bond definition
   is understood to point towards the 3'-end (5'-end) of the strand.

Example input and data files for DNA duplexes can be found in examples/PACKAGES/cgdna/examples/oxDNA/ and /oxDNA2/.
A simple python setup tool which creates single straight or helical DNA strands,
DNA duplexes or arrays of DNA duplexes can be found in examples/PACKAGES/cgdna/util/.

Please cite :ref:`(Henrich) <Henrich2>` in any publication that uses
this implementation. An updated documentation that contains general information
on the model, its implementation and performance as well as the structure of
the data and input file can be found `here <PDF/CG-DNA.pdf>`_.

Please cite also the relevant oxDNA2 publications
:ref:`(Snodin) <Snodin2>` and :ref:`(Sulc) <Sulc2>`.

----------

Restrictions
""""""""""""

These pair styles can only be used if LAMMPS was built with the
CG-DNA package and the MOLECULE and ASPHERE package.  See the
:doc:`Build package <Build_package>` page for more info.

Related commands
""""""""""""""""

:doc:`bond_style oxdna2/fene <bond_oxdna>`, :doc:`pair_coeff <pair_coeff>`,
:doc:`bond_style oxdna/fene <bond_oxdna>`, :doc:`pair_style oxdna/excv <pair_oxdna>`,
:doc:`bond_style oxrna2/fene <bond_oxdna>`, :doc:`pair_style oxrna2/excv <pair_oxrna2>`,
:doc:`atom_style oxdna <atom_style>`, :doc:`fix nve/dotc/langevin <fix_nve_dotc_langevin>`

Default
"""""""

none

----------

.. _Henrich2:

**(Henrich)** O. Henrich, Y. A. Gutierrez-Fosado, T. Curk, T. E. Ouldridge, Eur. Phys. J. E 41, 57 (2018).

.. _Snodin2:

**(Snodin)** B.E. Snodin, F. Randisi, M. Mosayebi, et al., J. Chem. Phys. 142, 234901 (2015).

.. _Sulc2:

**(Sulc)** P. Sulc, F. Romano, T.E. Ouldridge, L. Rovigatti, J.P.K. Doye, A.A. Louis, J. Chem. Phys. 137, 135101 (2012).

.. _Ouldridge-DPhil2:

**(Ouldridge-DPhil)** T.E. Ouldridge, Coarse-grained modelling of DNA and DNA self-assembly, DPhil. University of Oxford (2011).

.. _Ouldridge2:

**(Ouldridge)** T.E. Ouldridge, A.A. Louis, J.P.K. Doye, J. Chem. Phys. 134, 085101 (2011).
