.. index:: fix bond/create
.. index:: fix bond/create/angle

fix bond/create command
=======================

fix bond/create/angle command
=============================

Syntax
""""""

.. parsed-literal::

   fix ID group-ID bond/create Nevery itype jtype Rmin bondtype keyword values ...

* ID, group-ID are documented in :doc:`fix <fix>` command
* bond/create = style name of this fix command
* Nevery = attempt bond creation every this many steps
* itype,jtype = atoms of itype can bond to atoms of jtype
* Rmin = 2 atoms separated by less than Rmin can bond (distance units)
* bondtype = type of created bonds
* zero or more keyword/value pairs may be appended to args
* keyword = *iparam* or *jparam* or *prob* or *atype* or *dtype* or *itype* or *aconstrain*

  .. parsed-literal::

       *iparam* values = maxbond, newtype
         maxbond = max # of bonds of bondtype the itype atom can have
         newtype = change the itype atom to this type when maxbonds exist
       *jparam* values = maxbond, newtype
         maxbond = max # of bonds of bondtype the jtype atom can have
         newtype = change the jtype atom to this type when maxbonds exist
       *prob* values = fraction seed
         fraction = create a bond with this probability if otherwise eligible
         seed = random number seed (positive integer)
       *atype* value = angletype
         angletype = type of created angles
       *dtype* value = dihedraltype
         dihedraltype = type of created dihedrals
       *itype* value = impropertype
         impropertype = type of created impropers
       *aconstrain* value = amin amax
         amin = minimal angle at which new bonds can be created
         amax = maximal angle at which new bonds can be created

Examples
""""""""

.. code-block:: LAMMPS

   fix 5 all bond/create 10 1 2 0.8 1
   fix 5 all bond/create 1 3 3 0.8 1 prob 0.5 85784 iparam 2 3
   fix 5 all bond/create 1 3 3 0.8 1 prob 0.5 85784 iparam 2 3 atype 1 dtype 2
   fix 5 all bond/create/angle 10 1 2 1.122 1 aconstrain 120 180 prob 1 4928459 iparam 2 1 jparam 2 2

Description
"""""""""""

Create bonds between pairs of atoms as a simulation runs according to
specified criteria.  This can be used to model cross-linking of
polymers, the formation of a percolation network, etc.  In this
context, a bond means an interaction between a pair of atoms computed
by the :doc:`bond_style <bond_style>` command.  Once the bond is created
it will be permanently in place.  Optionally, the creation of a bond
can also create angle, dihedral, and improper interactions that bond
is part of.  See the discussion of the *atype*, *dtype*, and *itype*
keywords below.

This is different than a :doc:`pairwise <pair_style>` bond-order
potential such as Tersoff or AIREBO which infers bonds and many-body
interactions based on the current geometry of a small cluster of atoms
and effectively creates and destroys bonds and higher-order many-body
interactions from timestep to timestep as atoms move.

A check for possible new bonds is performed every *Nevery* timesteps.
If two atoms I,J are within a distance *Rmin* of each other, if I is
of atom type *itype*, if J is of atom type *jtype*, if both I and J
are in the specified fix group, if a bond does not already exist
between I and J, and if both I and J meet their respective *maxbond*
requirement (explained below), then I,J is labeled as a "possible"
bond pair.

If several atoms are close to an atom, it may have multiple possible
bond partners.  Every atom checks its list of possible bond partners
and labels the closest such partner as its "sole" bond partner.  After
this is done, if atom I has atom J as its sole partner, and atom J has
atom I as its sole partner, then the I,J bond is "eligible" to be
formed.

Note that these rules mean an atom will only be part of at most one
created bond on a given timestep.  It also means that if atom I
chooses atom J as its sole partner, but atom J chooses atom K is its
sole partner (due to Rjk < Rij), then this means atom I will not form
a bond on this timestep, even if it has other possible bond partners.

It is permissible to have *itype* = *jtype*\ .  *Rmin* must be <= the
pairwise cutoff distance between *itype* and *jtype* atoms, as defined
by the :doc:`pair_style <pair_style>` command.

The *iparam* and *jparam* keywords can be used to limit the bonding
functionality of the participating atoms.  Each atom keeps track of
how many bonds of *bondtype* it already has.  If atom I of
itype already has *maxbond* bonds (as set by the *iparam*
keyword), then it will not form any more.  Likewise for atom J.  If
*maxbond* is set to 0, then there is no limit on the number of bonds
that can be formed with that atom.

The *newtype* value for *iparam* and *jparam* can be used to change
the atom type of atom I or J when it reaches *maxbond* number of bonds
of type *bondtype*\ .  This means it can now interact in a pairwise
fashion with other atoms in a different way by specifying different
:doc:`pair_coeff <pair_coeff>` coefficients.  If you do not wish the
atom type to change, simply specify *newtype* as *itype* or *jtype*\ .

The *prob* keyword can also effect whether an eligible bond is
actually created.  The *fraction* setting must be a value between 0.0
and 1.0.  A uniform random number between 0.0 and 1.0 is generated and
the eligible bond is only created if the random number < fraction.

The *aconstrain* keyword is only available with the fix
bond/create/angle command.  It allows to specify a minimal and maximal
angle *amin* and *amax* between the two prospective bonding partners and
a third particle that is already bonded to one of the two partners.
Such a criterion can be important when new angles are defined together
with the formation of a new bond.  Without a restriction on the
permissible angle, and for stiffer angle potentials, very large energies
can arise and lead to uncontrolled behavior.

Any bond that is created is assigned a bond type of *bondtype*.

When a bond is created, data structures within LAMMPS that store bond
topology are updated to reflect the creation.  If the bond is part of
new 3-body (angle) or 4-body (dihedral, improper) interactions, you
can choose to create new angles, dihedrals, impropers as well, using
the *atype*, *dtype*, and *itype* keywords.  All of these changes
typically affect pairwise interactions between atoms that are now part
of new bonds, angles, etc.

.. note::

   One data structure that is not updated when a bond breaks are
   the molecule IDs stored by each atom.  Even though two molecules
   become one molecule due to the created bond, all atoms in the new
   molecule retain their original molecule IDs.

If the *atype* keyword is used and if an angle potential is defined
via the :doc:`angle_style <angle_style>` command, then any new 3-body
interactions inferred by the creation of a bond will create new angles
of type *angletype*, with parameters assigned by the corresponding
:doc:`angle_coeff <angle_coeff>` command.  Likewise, the *dtype* and
*itype* keywords will create new dihedrals and impropers of type
*dihedraltype* and *impropertype*\ .

.. note::

   To create a new bond, the internal LAMMPS data structures that
   store this information must have space for it.  When LAMMPS is
   initialized from a data file, the list of bonds is scanned and the
   maximum number of bonds per atom is tallied.  If some atom will
   acquire more bonds than this limit as this fix operates, then the
   "extra bond per atom" parameter must be set to allow for it.  Ditto
   for "extra angle per atom", "extra dihedral per atom", and "extra
   improper per atom" if angles, dihedrals, or impropers are being added
   when bonds are created.  See the :doc:`read_data <read_data>` or
   :doc:`create_box <create_box>` command for more details.  Note that a
   data file with no atoms can be used if you wish to add non-bonded
   atoms via the :doc:`create atoms <create_atoms>` command, e.g. for a
   percolation simulation.

.. note::

   LAMMPS stores and maintains a data structure with a list of the
   first, second, and third neighbors of each atom (within the bond topology of
   the system) for use in weighting pairwise interactions for bonded
   atoms.  Note that adding a single bond always adds a new first neighbor
   but may also induce \*many\* new second and third neighbors, depending on the
   molecular topology of your system.  The "extra special per atom"
   parameter must typically be set to allow for the new maximum total
   size (first + second + third neighbors) of this per-atom list.  There are 2
   ways to do this.  See the :doc:`read_data <read_data>` or
   :doc:`create_box <create_box>` commands for details.

.. note::

   Even if you do not use the *atype*, *dtype*, or *itype*
   keywords, the list of topological neighbors is updated for atoms
   affected by the new bond.  This in turn affects which neighbors are
   considered for pairwise interactions, using the weighting rules set by
   the :doc:`special_bonds <special_bonds>` command.  Consider a new bond
   created between atoms I,J.  If J has a bonded neighbor K, then K
   becomes a second neighbor of I.  Even if the *atype* keyword is not used
   to create angle I-J-K, the pairwise interaction between I and K will
   be potentially turned off or weighted by the 1-3 weighting specified
   by the :doc:`special_bonds <special_bonds>` command.  This is the case
   even if the "angle yes" option was used with that command.  The same
   is true for third neighbors (1-4 interactions), the *dtype* keyword, and
   the "dihedral yes" option used with the
   :doc:`special_bonds <special_bonds>` command.

Note that even if your simulation starts with no bonds, you must
define a :doc:`bond_style <bond_style>` and use the
:doc:`bond_coeff <bond_coeff>` command to specify coefficients for the
*bondtype*\ .  Similarly, if new atom types are specified by the
*iparam* or *jparam* keywords, they must be within the range of atom
types allowed by the simulation and pairwise coefficients must be
specified for the new types.

Computationally, each timestep this fix operates, it loops over
neighbor lists and computes distances between pairs of atoms in the
list.  It also communicates between neighboring processors to
coordinate which bonds are created.  Moreover, if any bonds are
created, neighbor lists must be immediately updated on the same
timestep.  This is to insure that any pairwise interactions that
should be turned "off" due to a bond creation, because they are now
excluded by the presence of the bond and the settings of the
:doc:`special_bonds <special_bonds>` command, will be immediately
recognized.  All of these operations increase the cost of a timestep.
Thus you should be cautious about invoking this fix too frequently.

You can dump out snapshots of the current bond topology via the :doc:`dump local <dump>` command.

.. note::

   Creating a bond typically alters the energy of a system.  You
   should be careful not to choose bond creation criteria that induce a
   dramatic change in energy.  For example, if you define a very stiff
   harmonic bond and create it when 2 atoms are separated by a distance
   far from the equilibrium bond length, then the 2 atoms will oscillate
   dramatically when the bond is formed.  More generally, you may need to
   thermostat your system to compensate for energy changes resulting from
   created bonds (and angles, dihedrals, impropers).

----------

Restart, fix_modify, output, run start/stop, minimize info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

No information about this fix is written to :doc:`binary restart files
<restart>`.  None of the :doc:`fix_modify <fix_modify>` options are
relevant to this fix.

This fix computes two statistics which it stores in a global vector of
length 2, which can be accessed by various :doc:`output commands
<Howto_output>`.  The vector values calculated by this fix are
"intensive".

These are the 2 quantities:

* (1) # of bonds created on the most recent creation timestep
* (2) cumulative # of bonds created

No parameter of this fix can be used with the *start/stop* keywords of
the :doc:`run <run>` command.  This fix is not invoked during :doc:`energy minimization <minimize>`.

Restrictions
""""""""""""

This fix is part of the MC package.  It is only enabled if LAMMPS was
built with that package.  See the :doc:`Build package <Build_package>`
doc page for more info.

Related commands
""""""""""""""""

:doc:`fix bond/break <fix_bond_break>`, :doc:`fix bond/react <fix_bond_react>`, :doc:`fix bond/swap <fix_bond_swap>`,
:doc:`dump local <dump>`, :doc:`special_bonds <special_bonds>`

Default
"""""""

The option defaults are iparam = (0,itype), jparam = (0,jtype), and
prob = 1.0.
