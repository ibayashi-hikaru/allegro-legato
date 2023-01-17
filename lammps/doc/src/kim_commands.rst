.. index:: kim_commands

kim command
===========

Syntax
""""""

.. code-block:: LAMMPS

   kim sub-command args

* sub-command = :ref:`init <kim_init>` or :ref:`interactions <interactions>` or
  :ref:`query <query>` or :ref:`param <param>` or :ref:`property <property>`

* args = arguments used by a particular sub-command

Examples
""""""""

.. code-block:: LAMMPS

   kim init args
   kim interactions args
   kim query args
   kim param args
   kim property args

Description
"""""""""""

The *kim command* includes a set of sub-commands that allow LAMMPS users to use
interatomic models (IM) (potentials and force fields) and their predictions for
various physical properties archived in the
`Open Knowledgebase of Interatomic Models (OpenKIM) <https://openkim.org>`_
repository.

Using OpenKIM provides LAMMPS users with immediate access to a large number of
verified IMs and their predictions. OpenKIM IMs have multiple benefits including
`reliability, reproducibility and convenience <https://openkim.org/doc/overview/kim-features/>`_.

.. _IM_types:

There are two types of IMs archived in OpenKIM:

1. The first type is called a *KIM Portable Model* (PM).  A KIM PM is an
   independent computer implementation of an IM written in one of the languages
   supported by KIM (C, C++, Fortran) that conforms to the KIM Application
   Programming Interface (`KIM API <https://openkim.org/kim-api/>`_) Portable
   Model Interface (PMI) standard.  A KIM PM will work seamlessly with any
   simulation code that supports the KIM API/PMI standard (including LAMMPS; see
   `complete list of supported codes <https://openkim.org/projects-using-kim/>`_).
2. The second type is called a *KIM Simulator Model* (SM).  A KIM SM is an IM
   that is implemented natively within a simulation code (\ *simulator*\ ) that
   supports the KIM API Simulator Model Interface (SMI); in this case LAMMPS.  A
   separate SM package is archived in OpenKIM for each parameterization of the
   IM, which includes all of the necessary parameter files, LAMMPS commands, and
   metadata (supported species, units, etc.) needed to run the IM in LAMMPS.

With these two IM types, OpenKIM can archive and test almost all IMs that can be
used by LAMMPS. (It is easy to contribute new IMs to OpenKIM, see the
`upload instructions <https://openkim.org/doc/repository/adding-content/>`_.)

OpenKIM IMs are uniquely identified by a
`KIM ID <https://openkim.org/doc/schema/kim-ids/>`_.
The extended KIM ID consists of
a human-readable prefix identifying the type of IM, authors, publication year,
and supported species, separated by two underscores from the KIM ID itself,
which begins with an IM code
(\ *MO* for a KIM Portable Model, and *SM* for a KIM Simulator Model)
followed by a unique 12-digit code and a 3-digit version identifier.
By convention SM prefixes begin with *Sim_* to readily identify them.

.. parsed-literal::

   SW_StillingerWeber_1985_Si__MO_405512056662_005
   Sim_LAMMPS_ReaxFF_StrachanVanDuinChakraborty_2003_CHNO__SM_107643900657_001

Each OpenKIM IM has a dedicated "Model Page" on `OpenKIM <https://openkim.org>`_
providing all the information on the IM including a title, description,
authorship and citation information, test and verification check results,
visualizations of results, a wiki with documentation and user comments, and
access to raw files, and other information.
The URL for the Model Page is constructed from the
`extended KIM ID <https://openkim.org/doc/schema/kim-ids/>`_ of the IM:

.. parsed-literal::

   https://openkim.org/id/extended_KIM_ID

For example, for the Stillinger-Weber potential listed above the Model Page is
located at:

.. parsed-literal::

   `https://openkim.org/id/SW_StillingerWeber_1985_Si__MO_405512056662_005 <https://openkim.org/id/SW_StillingerWeber_1985_Si__MO_405512056662_005>`_

See the
`current list of KIM PMs and SMs archived in OpenKIM <https://openkim.org/browse/models/by-species>`_.
This list is sorted by species and can be filtered to display only IMs for
certain species combinations.

See `Obtaining KIM Models <https://openkim.org/doc/usage/obtaining-models>`_ to
learn how to install a pre-built binary of the OpenKIM Repository of Models.

.. note::
   It is also possible to locally install IMs not archived in OpenKIM,
   in which case their names do not have to conform to the KIM ID format.

Using OpenKIM IMs with LAMMPS (*kim init*, *kim interactions*)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Two sub-commands are employed when using OpenKIM IMs in LAMMPS, one to select
the IM and perform necessary initialization (\ *kim init*\ ), and the second to
set up the IM for use by executing any necessary LAMMPS commands
(\ *kim interactions*\ ).  Both are required.

Syntax
------

.. code-block:: LAMMPS

   kim init model user_units unitarg
   kim interactions typeargs

.. _typeargs_options:

* model = name of the KIM interatomic model (the KIM ID for models archived in
  OpenKIM)
* user_units = the LAMMPS :doc:`units <units>` style assumed in the LAMMPS
  input script
* unitarg = *unit_conversion_mode* (optional)
* typeargs = atom type to species mapping (one entry per atom type) or
  *fixed_types* for models with a preset fixed mapping

Examples
--------

.. code-block:: LAMMPS

   kim init SW_StillingerWeber_1985_Si__MO_405512056662_005 metal
   kim interactions Si

   kim init Sim_LAMMPS_ReaxFF_StrachanVanDuinChakraborty_2003_CHNO__SM_107643900657_001 real
   kim init Sim_LAMMPS_ReaxFF_StrachanVanDuinChakraborty_2003_CHNO__SM_107643900657_001 metal unit_conversion_mode
   kim interactions C H O

   kim init Sim_LAMMPS_IFF_PCFF_HeinzMishraLinEmami_2015Ver1v5_FccmetalsMineralsSolventsPolymers__SM_039297821658_000 real
   kim interactions fixed_types


See the *examples/kim* directory for example input scripts that use KIM PMs
and KIM SMs.

.. _kim_init:

OpenKIM IM Initialization (*kim init*)
--------------------------------------

The *kim* command followed by *init* sub-command must be issued **before**
the simulation box is created (normally at the top of the file).  This command
sets the OpenKIM IM that will be used and may issue additional commands changing
LAMMPS default settings that are required for using the selected IM (such as
:doc:`units <units>` or :doc:`atom_style <atom_style>`).  If needed, those
settings can be overridden, however, typically a script containing a *kim init*
command would not include *units* and *atom_style* commands.

The required arguments of *kim init* are the *model* name of the IM to be used
in the simulation (for an IM archived in OpenKIM this is its
`extended KIM ID <https://openkim.org/doc/schema/kim-ids/>`_, and the
*user_units*, which are the LAMMPS :doc:`units style <units>` used in the input
script. (Any dimensioned numerical values in the input script and values read in
from files are expected to be in the *user_units* system.)

The selected IM can be either a :ref:`KIM PM or a KIM SM <IM_types>`.  For a KIM
SM, the *kim init* command verifies that the SM is designed to work with LAMMPS
(and not another simulation code).  In addition, the LAMMPS version used for
defining the SM and the LAMMPS version being currently run are printed to help
diagnose any incompatible changes to input script or command syntax between the
two LAMMPS versions.

Based on the selected model *kim init* may modify the
:doc:`atom_style <atom_style>`.  Some SMs have requirements for this setting.
If this is the case, then *atom_style* will be set to the required style.
Otherwise, the value is left unchanged (which in the absence of an *atom_style*
command in the input script is the
:doc:`default atom_style value <atom_style>`).

Regarding units, the *kim init* behaves in different ways depending on whether
or not *unit conversion mode* is activated as indicated by the optional
*unitarg* argument.
If unit conversion mode is **not** active, then *user_units* must either match
the required units of the IM or the IM must be able to adjust its units to
match. (The latter is only possible with some KIM PMs; SMs can never adjust
their units.) If a match is possible, the LAMMPS :doc:`units <units>` command is
called to set the units to *user_units*\ .  If the match fails, the simulation
is terminated with an error.  The *kim init* command also sets the
default value for the :doc:`skin <neighbor>` (extra distance beyond force
cutoff) as 2.0 Angstroms and sets the default value for the
:doc:`timestep <timestep>` size as 1.0 femtosecond.

Here is an example of a LAMMPS script to compute the cohesive energy of a
face-centered cubic (fcc) lattice for the MEAM potential by Pascuet and
Fernandez (2015) for Al.


.. code-block:: LAMMPS

   kim          init Sim_LAMMPS_MEAM_PascuetFernandez_2015_Al__SM_811588957187_000 metal
   boundary     p p p
   lattice      fcc 4.049
   region       simbox block 0 1 0 1 0 1 units lattice
   create_box   1 simbox
   create_atoms 1 box
   mass         1 26.981539
   kim          interactions Al
   run          0
   variable     Ec equal (pe/count(all))
   print        "Cohesive Energy = ${Ec} eV"

The above script will end with an error in the *kim init* line if the IM is
changed to another potential for Al that does not work with *metal* units.  To
address this, *kim init* offers the *unit_conversion_mode* as shown below.

If unit conversion mode *is* active, then *kim init* calls the LAMMPS
:doc:`units <units>` command to set the units to the IM's required or preferred
units.  Conversion factors between the IM's units and the *user_units* are
defined for all :doc:`physical quantities <units>` (mass, distance, etc.).
(Note that converting to or from the "lj" unit style is not supported.)
These factors are stored as :doc:`internal style variables <variable>` with the
following standard names:

.. parsed-literal::

   _u_mass
   _u_distance
   _u_time
   _u_energy
   _u_velocity
   _u_force
   _u_torque
   _u_temperature
   _u_pressure
   _u_viscosity
   _u_charge
   _u_dipole
   _u_efield
   _u_density

If desired, the input script can be designed to work with these conversion
factors so that the script will work without change with any OpenKIM IM.
(This approach is used in the
`OpenKIM Testing Framework <https://openkim.org/doc/evaluation/kim-tests/>`_.)

For example, the script given above for the cohesive energy of fcc Al can be
rewritten to work with any IM regardless of units.  The following script
constructs an fcc lattice with a lattice parameter defined in meters, computes
the total energy, and prints the cohesive energy in Joules regardless of the
units of the IM.

.. code-block:: LAMMPS

   kim          init Sim_LAMMPS_MEAM_PascuetFernandez_2015_Al__SM_811588957187_000 si unit_conversion_mode
   boundary     p p p
   lattice      fcc $(4.049e-10*v__u_distance)
   region       simbox block 0 1 0 1 0 1 units lattice
   create_box   1 simbox
   create_atoms 1 box
   mass         1 $(4.480134e-26*v__u_mass)
   kim          interactions Al
   neighbor     $(2e-10*v__u_distance) bin
   run          0
   variable     Ec_in_J equal (pe/count(all))/v__u_energy
   print        "Cohesive Energy = ${Ec_in_J} J"

Note the multiplication by `v__u_distance` and `v__u_mass` to convert from SI
units (specified in the *kim init* command) to whatever units the IM uses (metal
in this case), and the division by `v__u_energy` to convert from the IM's energy
units to SI units (Joule).  This script will work correctly for any IM for Al
(KIM PM or SM) selected by the *kim init* command.

Care must be taken to apply unit conversion to dimensional variables read in
from a file.  For example, if a configuration of atoms is read in from a dump
file using the :doc:`read_dump <read_dump>` command, the following can be done
to convert the box and all atomic positions to the correct units:

.. code-block:: LAMMPS

   change_box all x scale ${_u_distance} &
                  y scale ${_u_distance} &
                  z scale ${_u_distance} &
                  xy final $(xy*v__u_distance) &
                  xz final $(xz*v__u_distance) &
                  yz final $(yz*v__u_distance) &
                  remap

.. note::

   Unit conversion will only work if the conversion factors are placed in all
   appropriate places in the input script.  It is up to the user to do this
   correctly.

.. _interactions:

OpenKIM IM Execution (*kim interactions*)
-----------------------------------------

The second and final step in using an OpenKIM IM is to execute the
*kim interactions* command.  This command must be preceded by a *kim init*
command and a command that defines the number of atom types *N* (such as
:doc:`create_box <create_box>`).
The *kim interactions* command has one argument *typeargs*\ .  This argument
contains either a list of *N* chemical species, which defines a mapping between
atom types in LAMMPS to the available species in the OpenKIM IM, or the keyword
*fixed_types* for models that have a preset fixed mapping (i.e.  the mapping
between LAMMPS atom types and chemical species is defined by the model and
cannot be changed).  In the latter case, the user must consult the model
documentation to see how many atom types there are and how they map to the
chemical species.

For example, consider an OpenKIM IM that supports Si and C species.  If the
LAMMPS simulation has four atom types, where the first three are Si, and the
fourth is C, the following *kim interactions* command would be used:

.. code-block:: LAMMPS

   kim interactions Si Si Si C

Alternatively, for a model with a fixed mapping the command would be:

.. code-block:: LAMMPS

   kim interactions fixed_types

The *kim interactions* command performs all the necessary steps to set up the
OpenKIM IM selected in the *kim init* command.  The specific actions depend on
whether the IM is a KIM PM or a KIM SM.  For a KIM PM, a
:doc:`pair_style kim <pair_kim>` command is executed followed by the appropriate
*pair_coeff* command.  For example, for the Ercolessi and Adams (1994) KIM PM
for Al set by the following commands:

.. code-block:: LAMMPS

   kim init EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005 metal
   ...
   ...  box specification lines skipped
   ...
   kim interactions Al

the *kim interactions* command executes the following LAMMPS input commands:

.. code-block:: LAMMPS

   pair_style kim EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005
   pair_coeff * * Al

For a KIM SM, the generated input commands may be more complex and require that
LAMMPS is built with the required packages included for the type of potential
being used.  The set of commands to be executed is defined in the SM
specification file, which is part of the SM package.  For example, for the
Strachan et al.  (2003) ReaxFF SM set by the following commands:

.. code-block:: LAMMPS

   kim init Sim_LAMMPS_ReaxFF_StrachanVanDuinChakraborty_2003_CHNO__SM_107643900657_000 real
   ...
   ...  box specification lines skipped
   ...
   kim interactions C H N O

the *kim interactions* command executes the following LAMMPS input commands:

.. code-block:: LAMMPS

   pair_style reaxff lmp_control safezone 2.0 mincap 100
   pair_coeff * * ffield.reax.rdx C H N O
   fix reaxqeq all qeq/reaxff 1 0.0 10.0 1.0e-6 param.qeq

.. note::

    The files *lmp_control*, *ffield.reax.rdx* and *param.qeq* are specific to
    the Strachan et al.  (2003) ReaxFF parameterization and are archived as part
    of the SM package in OpenKIM.

.. note::

    Parameters like cutoff radii and charge tolerances, which have an effect on
    IM predictions, are also included in the SM definition ensuring
    reproducibility.

.. note::

   When using *kim init* and *kim interactions* to select and set up an OpenKIM
   IM, other LAMMPS commands for the same functions (such as pair_style,
   pair_coeff, bond_style, bond_coeff, fixes related to charge equilibration,
   etc.) should normally not appear in the input script.

.. note::

   *kim interactions* must be called each time after the
   :doc:`change_box <change_box>` command to provide the correct settings (it
   should be called with the same :ref:`typeargs <typeargs_options>` as the
   first call.)  The reason is that changing a periodic boundary to a
   non-periodic one, or in general, using the *change_box* command after the
   interactions are set via *kim interactions* or *pair_coeff* commands might
   affect some of the settings.  For example, SM models containing Coulombic
   terms in the interactions require different settings if a periodic boundary
   changes to a non-periodic one.  In other cases, the second call to
   *kim interactions* does not affect any other settings.

.. _query:

Using OpenKIM Web Queries in LAMMPS (*kim query*)
"""""""""""""""""""""""""""""""""""""""""""""""""

The *kim query* command performs a web query to retrieve the predictions of an
IM set by *kim init* for material properties archived in
`OpenKIM <https://openkim.org>`_.

Syntax
------

.. code-block:: LAMMPS

   kim query variable formatarg query_function queryargs

* variable(s) = single name or list of names of (string style) LAMMPS
  variable(s) where a query result or parameter get result is stored.  Variables
  that do not exist will be created by the command
* formatarg = *list, split, or index* (optional):

  .. parsed-literal::

     *list* = returns a single string with a list of space separated values
            (e.g. "1.0 2.0 3.0"), which is placed in a LAMMPS variable as
            defined by the *variable* argument. [default]
     *split* = returns the values separately in new variables with names based
            on the prefix specified in *variable* and a number appended to
            indicate which element in the list of values is in the variable
     *index* = returns a variable style index that can be incremented via the
            next command.  This enables the construction of simple loops

* query_function = name of the OpenKIM web API query function to be used
* queryargs = a series of *keyword=value* pairs that represent the web query;
  supported keywords depend on the query function

Examples
--------

.. code-block:: LAMMPS

   kim query a0 get_lattice_constant_cubic crystal=[fcc] species=[Al] units=[angstrom]
   kim query model index get_available_models species=[Al] potential_type=[eam]

The result of the query is stored in one or more
:doc:`string style variables <variable>` as determined by the optional
*formatarg* argument.  For the "list"
setting of *formatarg* (or if *formatarg* is not specified), the result is
returned as a space-separated list of values in *variable*\ .  The *formatarg*
keyword "split" separates the result values into individual variables of the
form *prefix_I*, where *prefix* is set to the *kim query* *variable* argument
and *I* ranges from 1 to the number of returned values.  The number and order of
the returned values is determined by the type of query performed.  The
*formatarg* keyword "index" returns a :doc:`variable style index <variable>`
that can be incremented via the :doc:`next <next>` command.  This enables the
construction of simple loops over the returned values by the type of query
performed.

.. note::

   *kim query* only supports queries that return a single result or an array of
   values.  More complex queries that return a JSON structure are not currently
   supported.  An attempt to use *kim query* in such cases will generate an
   error.

The second required argument *query_function* is the name of the query function
to be called (e.g.  *get_lattice_constant_cubic*\ ).  All following
:doc:`arguments <Commands_parse>` are parameters handed over to the web query in
the format *keyword=value*, where *value* is always an array of one or more
comma-separated items in brackets.  The list of supported keywords and the type
and format of their values depend on the query function used.  The current list
of query functions is available on the OpenKIM webpage at
`https://openkim.org/doc/usage/kim-query <https://openkim.org/doc/usage/kim-query>`_.

.. note::

   All query functions, except *get_available_models*, require the
   *model* keyword, which identifies the IM whose predictions are being queried.
   *kim query* automatically generates the *model* keyword based on the IM set
   in by *kim init*, and it can be overwritten if specified as an argument to
   the *kim query*\ .  Where *kim init* is not specified, the *model* keyword
   must be provided as an argument to the *kim query*\ .

.. note::

   Each *query_function* is associated with a default method (implemented as a
   `KIM Test <https://openkim.org/doc/evaluation/kim-tests/>`_) used to compute
   this property.  In cases where there are multiple methods in OpenKIM for
   computing a property, a *method* keyword can be provided to select the method
   of choice.  See the
   `query documentation <https://openkim.org/doc/usage/kim-query>`_ to see which
   methods are available for a given *query_function*\ .

*kim query* Usage Examples and Further Clarifications
-----------------------------------------------------

The data obtained by *kim query* commands can be used as part of the setup or
analysis phases of LAMMPS simulations.  Some examples are given below.

**Define an equilibrium fcc crystal**

.. code-block:: LAMMPS

   kim      init EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005 metal
   boundary p p p
   kim      query a0 get_lattice_constant_cubic crystal=[fcc] species=[Al] units=[angstrom]
   lattice  fcc ${a0}
   ...

.. code-block:: LAMMPS

   units   metal
   kim     query a0 get_lattice_constant_cubic crystal=[fcc] species=[Al] units=[angstrom] model=[EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005]
   lattice fcc ${a0}
   ...

The *kim query* command retrieves from `OpenKIM <https://openkim.org>`_ the
equilibrium lattice constant predicted by the Ercolessi and Adams (1994)
potential for the fcc structure and places it in variable *a0*\ .  This variable
is then used on the next line to set up the crystal.  By using *kim query*, the
user is saved the trouble and possible error of tracking this value down, or of
having to perform an energy minimization to find the equilibrium lattice
constant.

.. note::

    In *unit_conversion_mode* the results obtained from a *kim query* would need
    to be converted to the appropriate units system.  For example, in the above
    script, the lattice command would need to be changed to:
    "lattice fcc $(v_a0*v__u_distance)".

**Define an equilibrium hcp crystal**

.. code-block:: LAMMPS

   kim      init EAM_Dynamo_MendelevAckland_2007v3_Zr__MO_004835508849_000 metal
   boundary p p p
   kim      query latconst split get_lattice_constant_hexagonal crystal=[hcp] species=[Zr] units=[angstrom]
   lattice  custom ${latconst_1} a1 0.5 -0.866025 0 a2 0.5 0.866025 0 a3 0 0 $(latconst_2/latconst_1) &
            basis 0.333333 0.666666 0.25 basis 0.666666 0.333333 0.75
   ...

In this case the *kim query* returns two arguments (since the hexagonal close
packed (hcp) structure has two independent lattice constants).  The *formatarg*
keyword "split" places the two values into the variables *latconst_1* and
*latconst_2*\ .  (These variables are created if they do not already exist.)

**Define a crystal at finite temperature accounting for thermal expansion**

.. code-block:: LAMMPS

   kim      init EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005 metal
   boundary p p p
   kim      query a0 get_lattice_constant_cubic crystal=[fcc] species=[Al] units=[angstrom]
   kim      query alpha get_linear_thermal_expansion_coefficient_cubic crystal=[fcc] species=[Al] units=[1/K] temperature=[293.15] temperature_units=[K]
   variable DeltaT equal 300
   lattice  fcc $(v_a0*v_alpha*v_DeltaT)
   ...

As in the previous example, the equilibrium lattice constant is obtained for the
Ercolessi and Adams (1994) potential.  However, in this case the crystal is
scaled to the appropriate lattice constant at room temperature (293.15 K) by
using the linear thermal expansion constant predicted by the potential.

.. note::

   When passing numerical values as arguments (as in the case of the temperature
   in the above example) it is also possible to pass a tolerance indicating how
   close to the value is considered a match.  If no tolerance is passed a default
   value is used.  If multiple results are returned (indicating that the
   tolerance is too large), *kim query* will return an error.  See the
   `query documentation <https://openkim.org/doc/usage/kim-query>`_ to see which
   numerical arguments and tolerances are available for a given
   *query_function*\ .

**Compute defect formation energy**

.. code-block:: LAMMPS

   kim      init EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005 metal
   ...
   ... Build fcc crystal containing some defect and compute the total energy
   ... which is stored in the variable *Etot*
   ...
   kim      query Ec get_cohesive_energy_cubic crystal=[fcc] species=[Al] units=[eV]
   variable Eform equal ${Etot} - count(all)*${Ec}
   ...

The defect formation energy *Eform* is computed by subtracting the ideal fcc
cohesive energy of the atoms in the system from *Etot*\ .  The ideal fcc
cohesive energy of the atoms is obtained from
`OpenKIM <https://openkim.org>`_ for the Ercolessi and Adams (1994) potential.

**Retrieve equilibrium fcc crystal of all EAM potentials that support a specific species**

.. code-block:: LAMMPS

   kim   query model index get_available_models species=[Al] potential_type=[eam]
   label model_loop
   kim   query latconst get_lattice_constant_cubic crystal=[fcc] species=[Al] units=[angstrom] model=[${model}]
   print "FCC lattice constant (${model} potential) = ${latconst}"
   ...
   ...   do something with current value of latconst
   ...
   next  model
   jump  SELF model_loop

In this example, the *index* mode of *formatarg* is used.  The first *kim query*
returns the list of all available EAM potentials that support the *Al* species
and archived in `OpenKIM <https://openkim.org>`_.  The result of the query
operation is stored in the LAMMPS variable *model* as an index *variable*\ .
This variable is used later to access the values one at a time within a loop as
shown in the example.  The second *kim query* command retrieves from
`OpenKIM <https://openkim.org>`_ the equilibrium lattice constant predicted by
each potential for the fcc structure and places it in variable *latconst*\ .

.. note::

   *kim query* commands return results archived in
   `OpenKIM <https://openkim.org>`_.  These results are obtained using programs
   for computing material properties (KIM Tests and KIM Test Drivers) that were
   contributed to OpenKIM.  In order to give credit to Test developers, the
   number of times results from these programs are queried is tracked.  No other
   information about the nature of the query or its source is recorded.


.. _param:

Accessing KIM Model Parameters from LAMMPS (*kim param*)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""

All IMs are functional forms containing a set of parameters.  These parameters'
values are typically selected to best reproduce a training set of quantum
mechanical calculations or available experimental data.  For example, a
Lennard-Jones potential intended to model argon might have the values of its two
parameters, epsilon, and sigma, fit to the dimer dissociation energy or
thermodynamic properties at a critical point of the phase diagram.

Normally a user employing an IM should not modify its parameters since, as noted
above, these are selected to reproduce material properties.  However, there are
cases where accessing and modifying IM parameters is desired, such as for
assessing uncertainty, fitting an IM, or working with an ensemble of IMs.  As
explained :ref:`above <IM_types>`, IMs archived in OpenKIM are either Portable
Models (PMs) or Simulator Models (SMs).  KIM PMs are complete independent
implementations of an IM, whereas KIM SMs are wrappers to an IM implemented
within LAMMPS.  Two different mechanisms are provided for accessing IM
parameters in these two cases:

* For a KIM PM, the *kim param* command can be used to *get* and *set* the
  values of the PM's parameters as explained below.
* For a KIM SM, the user should consult the documentation page for the specific
  IM and follow instructions there for how to modify its parameters (if
  possible).

The *kim param get* and *kim param set* commands provide an interface to access
and change the parameters of a KIM PM that "publishes" its parameters and makes
them publicly available (see the
`KIM API documentation <https://kim-api.readthedocs.io/en/devel/features.html>`_
for details).

.. note::

   The *kim param set/get* command must be preceded by a *kim interactions*
   command (or alternatively by a *pair_style kim* and *pair_coeff* commands).
   The *kim param set* command may be used wherever a *pair_coeff* command may
   occur.

Syntax
------

.. code-block:: LAMMPS

   kim param get param_name index_range variable formatarg
   kim param set param_name index_range values

.. _formatarg_options:

* param_name = name of a KIM portable model parameter (which is published by the
  PM and available for access). The specific string used to identify a parameter
  is defined by the PM.  For example, for the
  `Stillinger-Weber (SW) potential in OpenKIM <https://openkim.org/id/SW_StillingerWeber_1985_Si__MO_405512056662_005>`_,
  the parameter names are *A, B, p, q, sigma, gamma, cutoff, lambda, costheta0*
* index_range = KIM portable model parameter index range (an integer for a
  single element, or pair of integers separated by a colon for a range of
  elements)
* variable(s) = single name or list of names of (string style) LAMMPS
  variable(s) where a query result or parameter get result is stored.  Variables
  that do not exist will be created by the command
* formatarg = *list, split, or explicit* (optional):
  .. parsed-literal::

     *list* = returns a single string with a list of space separated values
            (e.g. "1.0 2.0 3.0"), which is placed in a LAMMPS variable as
            defined by the *variable* argument.
     *split* = returns the values separately in new variables with names based
            on the prefix specified in *variable* and a number appended to
            indicate which element in the list of values is in the variable
     *explicit* = returns the values separately in one more more variable names
            provided as arguments that precede *formatarg*\ . [default]
* values = new value(s) to replace the current value(s) of a KIM portable model
  parameter

.. note::

   The list of all the parameters that a PM exposes for access/mutation are
   automatically written to the lammps log file when *kim init* is called.

Each published parameter of a KIM PM takes the form of an array of numerical
values.  The array can contain one element for a single-valued parameter, or a
set of values.  For example, the
`multispecies SW potential for the Zn-Cd-Hg-S-Se-Te system <https://openkim.org/id/SW_ZhouWardMartin_2013_CdTeZnSeHgS__MO_503261197030_002>`_
has the same parameter names as the
`single-species SW potential <https://openkim.org/id/SW_StillingerWeber_1985_Si__MO_405512056662_005>`_,
but each parameter array contains 21 entries that correspond to the parameter
values used for each pairwise combination of the model's six supported species
(this model does not have parameters specific to individual ternary combinations
of its supported species).

The *index_range* argument may either be an integer referring to a specific
element within the array associated with the parameter specified by
*param_name*, or a pair of integers separated by a colon that refer to a slice
of this array.  In both cases, one-based indexing is used to refer to the
entries of the array.

The result of a *get* operation for a specific *index_range* is stored in
one or more :doc:`LAMMPS string style variables <variable>` as determined by the
optional *formatarg* argument :ref:`documented above. <formatarg_options>` If
not specified, the default for *formatarg* is "explicit" for the *kim param*
command.

For the case where the result is an array with multiple values (i.e.
*index_range* contains a range), the optional "split" or "explicit" *formatarg*
keywords can be used to separate the results into multiple variables; see the
examples below.  Multiple parameters can be retrieved with a single call to
*kim param get* by repeating the argument list following *get*\ .

For a *set* operation, the *values* argument contains the new value(s) for the
element(s) of the parameter specified by *index_range*\ .  For the case where
multiple values are being set, *values* contains a set of values separated by
spaces.  Multiple parameters can be set with a single call to *kim param set* by
repeating the argument list following *set*\ .

*kim param* Usage Examples and Further Clarifications
-----------------------------------------------------

Examples of getting and setting KIM PM parameters with further clarifications
are provided below.

**Getting a scalar parameter**

.. code-block:: LAMMPS

   kim init SW_StillingerWeber_1985_Si__MO_405512056662_005 metal
   ...
   kim interactions Si
   kim param get A 1 VARA

or

.. code-block:: LAMMPS

   ...
   pair_style kim SW_StillingerWeber_1985_Si__MO_405512056662_005
   pair_coeff * * Si
   kim param get A 1 VARA

In these cases, the value of the SW *A* parameter is retrieved and placed in the
LAMMPS variable *VARA*\ .  The variable *VARA* can be used in the remainder of
the input script in the same manner as any other LAMMPS variable.

**Getting multiple scalar parameters with a single call**

.. code-block:: LAMMPS

   ...
   kim interactions Si
   kim param get A 1 VARA B 1 VARB

In this example, it is shown how to retrieve the *A* and *B* parameters of the
SW potential and store them in the LAMMPS variables *VARA* and *VARB*\ .

**Getting a range of values from a parameter**

There are several options when getting a range of values from a parameter
determined by the *formatarg* argument.

.. code-block:: LAMMPS

   kim init SW_ZhouWardMartin_2013_CdTeZnSeHgS__MO_503261197030_002 metal
   ...
   kim interactions Te Zn Se
   kim param get lambda 7:9 LAM_TeTe LAM_TeZn LAM_TeSe

In this case, *formatarg* is not specified and therefore the default
"explicit" mode is used. (The behavior would be the same if the word
*explicit* were added after *LAM_TeSe*\ .) Elements 7, 8 and 9 of parameter
lambda retrieved by the *get* operation are placed in the LAMMPS variables
*LAM_TeTe*, *LAM_TeZn* and *LAM_TeSe*, respectively.

.. note::

   In the above example, elements 7-9 of the lambda parameter correspond to
   Te-Te, Te-Zm and Te-Se interactions.  This can be determined by visiting the
   `model page for the specified potential <https://openkim.org/id/SW_ZhouWardMartin_2013_CdTeZnSeHgS__MO_503261197030_002>`_
   and looking at its parameter file linked to at the bottom of the page (file
   with .param ending) and consulting the README documentation provided with the
   driver for the PM being used.  A link to the driver is provided at the top of
   the model page.

.. code-block:: LAMMPS

   ...
   kim      interactions Te Zn Se
   kim      param get lambda 15:17 LAMS list
   variable LAM_VALUE index ${LAMS}
   label    loop_on_lambda
   ...
   ...      do something with the current value of lambda
   ...
   next     LAM_VALUE
   jump     SELF loop_on_lambda

In this case, the "list" mode of *formatarg* is used.  The result of the *get*
operation is stored in the LAMMPS variable *LAMS* as a string containing the
three retrieved values separated by spaces, e.g "1.0 2.0 3.0".  This can be used
in LAMMPS with an *index* variable to access the values one at a time within a
loop as shown in the example.  At each iteration of the loop *LAM_VALUE*
contains the current value of lambda.

.. code-block:: LAMMPS

   ...
   kim interactions Te Zn Se
   kim param get lambda 15:17 LAM split

In this case, the "split" mode of *formatarg* is used.  The three values
retrieved by the *get* operation are stored in the three LAMMPS variables
*LAM_15*, *LAM_16* and *LAM_17*\ .  The provided name "LAM" is used as prefix
and the location in the lambda array is appended to create the variable names.

**Setting a scalar parameter**

.. code-block:: LAMMPS

   kim init SW_StillingerWeber_1985_Si__MO_405512056662_005 metal
   ...
   kim interactions Si
   kim param set gamma 1 2.6

Here, the SW potential's gamma parameter is set to 2.6.  Note that the *get*
and *set* commands work together, so that a *get* following a *set* operation
will return the new value that was set.  For example,

.. code-block:: LAMMPS

   ...
   kim interactions Si
   kim param get gamma 1 ORIG_GAMMA
   kim param set gamma 1 2.6
   kim param get gamma 1 NEW_GAMMA
   ...
   print "original gamma = ${ORIG_GAMMA}, new gamma = ${NEW_GAMMA}"

Here, *ORIG_GAMMA* will contain the original gamma value for the SW potential,
while *NEW_GAMMA* will contain the value 2.6.

**Setting multiple scalar parameters with a single call**

.. code-block:: LAMMPS

   kim      init SW_ZhouWardMartin_2013_CdTeZnSeHgS__MO_503261197030_002 metal
   ...
   kim      interactions Cd Te
   variable VARG equal 2.6
   variable VARS equal 2.0951
   kim      param set gamma 1 ${VARG} sigma 3 ${VARS}

In this case, the first element of the *gamma* parameter and third element of
the *sigma* parameter are set to 2.6 and 2.0951, respectively.  This example
also shows how LAMMPS variables can be used when setting parameters.

**Setting a range of values of a parameter**

.. code-block:: LAMMPS

   kim init SW_ZhouWardMartin_2013_CdTeZnSeHgS__MO_503261197030_002 metal
   ...
   kim interactions Cd Te Zn Se Hg S
   kim param set sigma 2:6 2.35214 2.23869 2.04516 2.43269 1.80415

In this case, elements 2 through 6 of the parameter *sigma* are set to the
values 2.35214, 2.23869, 2.04516, 2.43269 and 1.80415 in order.

.. _property:

Writing material properties in standard KIM Property Instance format (*kim property*)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

The OpenKIM system includes a
collection of Tests (material property calculation codes), Models (interatomic
potentials), Predictions, and Reference Data (DFT or experiments).
Specifically, a KIM Test is a computation that when coupled with a KIM Model
generates the prediction of that model for a specific material property
rigorously defined by a KIM Property Definition (see the
`KIM Properties Framework <https://openkim.org/doc/schema/properties-framework/>`__
for further details).  A prediction of a material property for a given model is
a specific numerical realization of a property definition, referred to as a
"Property Instance." The objective of the *kim property* command is to make it
easy to output material properties in a standardized, machine readable, format
that can be easily ingested by other programs.  Additionally, it aims to make it
as easy as possible to convert a LAMMPS script that computes a material property
into a KIM Test that can then be uploaded to
`openkim.org <https://openkim.org>`_

A developer interested in creating a KIM Test using a LAMMPS script should first
determine whether a property definition that applies to their calculation
already exists in OpenKIM by searching the `properties page
<https://openkim.org/properties>`_.  If none exists, it is possible to use a
locally defined property definition contained in a file until it can be
uploaded to the official repository (see below).  Once one or more applicable
property definitions have been identified, the *kim property create*,
*kim property modify*, *kim property remove*, and *kim property destroy*,
commands provide an interface to create, set, modify, remove, and destroy
instances of them within a LAMMPS script.

Syntax
------

.. code-block:: LAMMPS

   kim property create  instance_id property_id
   kim property modify  instance_id key key_name key_name_key key_name_value
   kim property remove  instance_id key key_name
   kim property destroy instance_id
   kim property dump    file

* instance_id = a positive integer identifying the KIM property instance;
  (note that the results file can contain multiple property instances)
* property_id = identifier of a
  `KIM Property Definition <https://openkim.org/properties>`_, which can be (1)
  a property short name, (2) the full unique ID of the property (including the
  contributor and date), (3) a file name corresponding to a local property
  definition file
* key_name = one of the keys belonging to the specified KIM property definition
* key_name_key = a key belonging to a key-value pair (standardized in the
  `KIM Properties Framework <https://openkim.org/doc/schema/properties-framework>`__)
* key_name_value = value to be associated with a key_name_key in a key-value
  pair
* file = name of a file to write the currently defined set of KIM property
  instances to

Examples of each of the three *property_id* cases are shown below,

.. code-block:: LAMMPS

   kim property create 1 atomic-mass
   kim property create 2 cohesive-energy-relation-cubic-crystal

.. code-block:: LAMMPS

   kim property create 1 tag:brunnels@noreply.openkim.org,2016-05-11:property/atomic-mass
   kim property create 2 tag:staff@noreply.openkim.org,2014-04-15:property/cohesive-energy-relation-cubic-crystal

.. code-block:: LAMMPS

   kim property create 1 new-property.edn
   kim property create 2 /home/mary/marys-kim-properties/dissociation-energy.edn

In the last example, "new-property.edn" and
"/home/mary/marys-kim-properties/dissociation-energy.edn" are the names of files
that contain user-defined (local) property definitions.

A KIM property instance takes the form of a "map," i.e. a set of key-value
pairs akin to Perl's hash, Python's dictionary, or Java's Hashtable.  It
consists of a set of property key names, each of which is referred to here by
the *key_name* argument, that are defined as part of the relevant KIM Property
Definition and include only lowercase alphanumeric characters and dashes.  The
value paired with each property key is itself a map whose possible keys are
defined as part of the `KIM Properties Framework
<https://openkim.org/doc/schema/properties-framework>`__; these keys are
referred to by the *key_name_key* argument and their associated values by the
*key_name_value* argument.  These values may either be scalars or arrays,
as stipulated in the property definition.

.. note::

    Each map assigned to a *key_name* must contain the *key_name_key*
    "source-value" and an associated *key_name_value* of the appropriate
    type (as defined in the relevant KIM Property Definition).  For keys that
    are defined as having physical units, the "source-unit" *key_name_key* must
    also be given a string value recognized by
    `GNU units <https://www.gnu.org/software/units/>`_.

Once a *kim property create* command has been given to instantiate a property
instance, maps associated with the property's keys can be edited using the
*kim property modify* command.  In using this command, the special keyword
"key" should be given, followed by the property key name and the key-value pair
in the map associated with the key that is to be set.  For example, the
`atomic-mass <https://openkim.org/properties/show/2016-05-11/brunnels@noreply.openkim.org/atomic-mass>`_
property definition consists of two property keys named "mass" and "species."
An instance of this property could be created like so:

.. code-block:: LAMMPS

   kim property create 1 atomic-mass
   kim property modify 1 key species source-value Al
   kim property modify 1 key mass    source-value 26.98154
   kim property modify 1 key mass    source-unit amu

or, equivalently,

.. code-block:: LAMMPS

   kim property create 1 atomic-mass
   kim property modify 1 key species source-value Al       &
                         key mass    source-value 26.98154 &
                                     source-unit  amu

*kim property* Usage Examples and Further Clarifications
--------------------------------------------------------

**Create**

.. code-block:: LAMMPS

   kim property create instance_id property_id

The *kim property create* command takes as input a property instance ID and the
property definition name, and creates an initial empty property instance data
structure.  For example,

.. code-block:: LAMMPS

   kim property create 1 atomic-mass
   kim property create 2 cohesive-energy-relation-cubic-crystal

creates an empty property instance of the "atomic-mass" property definition
with instance ID 1 and an empty instance of the
"cohesive-energy-relation-cubic-crystal" property with ID 2.  A list of
published property definitions in OpenKIM can be found on the `properties page
<https://openkim.org/properties>`_.

One can also provide the name of a file in the current working directory or the
path of a file containing a valid property definition.  For example,

.. code-block:: LAMMPS

   kim property create 1 new-property.edn

where "new-property.edn" refers to a file name containing a new property
definition that does not exist in OpenKIM.

If the *property_id* given cannot be found in OpenKIM and no file of this name
containing a valid property definition can be found, this command will produce
an error with an appropriate message.  Calling *kim property create* with the
same instance ID multiple times will also produce an error.

**Modify**

.. code-block:: LAMMPS

   kim property modify instance_id key key_name key_name_key key_name_value

The *kim property modify* command incrementally builds the property instance
by receiving property definition keys along with associated arguments.  Each
*key_name* is associated with a map containing one or more key-value pairs (in
the form of *key_name_key*-*key_name_value* pairs).  For example,

.. code-block:: LAMMPS

   kim property modify 1 key species source-value Al
   kim property modify 1 key mass    source-value 26.98154
   kim property modify 1 key mass    source-unit  amu

where the special keyword "key" is followed by a *key_name* ("species" or
"mass" in the above) and one or more key-value pairs.  These key-value pairs
may continue until either another "key" keyword is given or the end of the
command line is reached.  Thus, the above could equivalently be written as

.. code-block:: LAMMPS

   kim property modify 1 key species source-value Al       &
                         key mass    source-value 26.98154 &
                         key mass    source-unit  amu

As an example of modifying multiple key-value pairs belonging to the map of a
single property key, the following command modifies the map of the
"cohesive-potential-energy" property key to contain the key "source-unit" which
is assigned a value of "eV" and the key "digits" which is assigned a value of
5,

.. code-block:: LAMMPS

   kim property modify 2 key cohesive-potential-energy source-unit eV digits 5

.. note::

    The relevant data types of the values in the map are handled automatically
    based on the specification of the key in the KIM Property Definition.  In
    the example above, this means that the value "eV" will automatically be
    interpreted as a string while the value 5 will be interpreted as an integer.

The values contained in maps can either be scalars, as in all of the examples
above, or arrays depending on which is stipulated in the corresponding Property
Definition.  For one-dimensional arrays, a single one-based index must be
supplied that indicates which element of the array is to be modified.  For
multidimensional arrays, multiple indices must be given depending on the
dimensionality of the array.

.. note::

   All array indexing used by *kim property modify* is one-based, i.e. the
   indices are enumerated 1, 2, 3, ...

.. note::

   The dimensionality of arrays are defined in the the corresponding Property
   Definition.  The extent of each dimension of an array can either be a
   specific finite number or indefinite and determined at run time.  If
   an array has a fixed extent, attempting to modify an out-of-range index will
   fail with an error message.

For example, the "species" property key of the
`cohesive-energy-relation-cubic-crystal
<https://openkim.org/properties/show/2014-04-15/staff@noreply.openkim.org/cohesive-energy-relation-cubic-crystal>`_
property is a one-dimensional array that can contain any number of entries
based on the number of atoms in the unit cell of a given cubic crystal.  To
assign an array containing the string "Al" four times to the "source-value" key
of the "species" property key, we can do so by issuing:

.. code-block:: LAMMPS

   kim property modify 2 key species source-value 1 Al
   kim property modify 2 key species source-value 2 Al
   kim property modify 2 key species source-value 3 Al
   kim property modify 2 key species source-value 4 Al

.. note::

    No declaration of the number of elements in this array was given;
    *kim property modify* will automatically handle memory management to allow
    an arbitrary number of elements to be added to the array.

.. note::

   In the event that *kim property modify* is used to set the value of an
   array index without having set the values of all lesser indices, they will
   be assigned default values based on the data type associated with the key in
   the map:

   .. table_from_list::
      :columns: 2

      * Data type
      * Default value
      * int
      * 0
      * float
      * 0.0
      * string
      * \"\"
      * file
      * \"\"

   For example, doing the following:

   .. code-block:: LAMMPS

      kim property create 2 cohesive-energy-relation-cubic-crystal
      kim property modify 2 key species source-value 4 Al

   will result in the "source-value" key in the map for the property key
   "species" being assigned the array ["", "", "", "Al"].

For convenience, the index argument provided may refer to an inclusive range of
indices by specifying two integers separated by a colon (the first integer must
be less than or equal to the second integer, and no whitespace should be
included).  Thus, the snippet above could equivalently be written:

.. code-block:: LAMMPS

   kim property modify 2 key species source-value 1:4 Al Al Al Al

Calling this command with a non-positive index, e.g.
``kim property modify 2 key species source-value 0 Al``, or an incorrect
number of input arguments, e.g.
``kim property modify 2 key species source-value 1:4 Al Al``, will result in an
error.

As an example of modifying multidimensional arrays, consider the "basis-atoms"
key in the `cohesive-energy-relation-cubic-crystal
<https://openkim.org/properties/show/2014-04-15/staff@noreply.openkim.org/cohesive-energy-relation-cubic-crystal>`_
property definition.  This is a two-dimensional array containing the fractional
coordinates of atoms in the unit cell of the cubic crystal.  In the case of,
e.g. a conventional fcc unit cell, the "source-value" key in the map associated
with this key should be assigned the following value:

.. code-block:: text

   [[0.0, 0.0, 0.0],
    [0.5, 0.5, 0.0],
    [0.5, 0.0, 0.5],
    [0.0, 0.5, 0.5]]

While each of the twelve components could be set individually, we can instead
set each row at a time using colon notation:

.. code-block:: LAMMPS

   kim property modify 2 key basis-atom-coordinates source-value 1 1:3 0.0 0.0 0.0
   kim property modify 2 key basis-atom-coordinates source-value 2 1:3 0.5 0.5 0.0
   kim property modify 2 key basis-atom-coordinates source-value 3 1:3 0.5 0.0 0.5
   kim property modify 2 key basis-atom-coordinates source-value 4 1:3 0.0 0.5 0.5

Where the first index given refers to a row and the second index refers to a
column.  We could, instead, choose to set each column at a time like so:

.. code-block:: LAMMPS

   kim property modify 2 key basis-atom-coordinates source-value 1:4 1 0.0 0.5 0.5 0.0 &
                         key basis-atom-coordinates source-value 1:4 2 0.0 0.5 0.0 0.5 &
                         key basis-atom-coordinates source-value 1:4 3 0.0 0.0 0.5 0.5

.. note::

   Multiple calls of *kim property modify* made for the same instance ID
   can be combined into a single invocation, meaning the following are
   both valid:

   .. code-block:: LAMMPS

      kim property modify 2 key basis-atom-coordinates source-value 1 1:3 0.0 0.0 0.0 &
                            key basis-atom-coordinates source-value 2 1:3 0.5 0.5 0.0 &
                            key basis-atom-coordinates source-value 3 1:3 0.5 0.0 0.5 &
                            key basis-atom-coordinates source-value 4 1:3 0.0 0.5 0.5

   .. code-block:: LAMMPS

      kim property modify 2 key short-name source-value 1 fcc                         &
                            key species source-value 1:4 Al Al Al Al                  &
                            key a source-value 1:5 3.9149 4.0000 4.032 4.0817 4.1602  &
                                  source-unit angstrom                                &
                                  digits 5                                            &
                            key basis-atom-coordinates source-value 1 1:3 0.0 0.0 0.0 &
                            key basis-atom-coordinates source-value 2 1:3 0.5 0.5 0.0 &
                            key basis-atom-coordinates source-value 3 1:3 0.5 0.0 0.5 &
                            key basis-atom-coordinates source-value 4 1:3 0.0 0.5 0.5

.. note::

   For multidimensional arrays, only one colon-separated range is allowed
   in the index listing.  Therefore,

   .. code-block:: LAMMPS

      kim property modify 2 key basis-atom-coordinates 1 1:3 0.0 0.0 0.0

   is valid but

   .. code-block:: LAMMPS

      kim property modify 2 key basis-atom-coordinates 1:2 1:3 0.0 0.0 0.0 0.0 0.0 0.0

   is not.

.. note::

   After one sets a value in a map with the *kim property modify* command,
   additional calls will overwrite the previous value.

**Remove**

.. code-block:: LAMMPS

   kim property remove instance_id key key_name

The *kim property remove* command can be used to remove a property key from a
property instance.  For example,

.. code-block:: LAMMPS

   kim property remove 2 key basis-atom-coordinates

**Destroy**

.. code-block:: LAMMPS

   kim property destroy instance_id

The *kim property destroy* command deletes a previously created property
instance ID.  For example,

.. code-block:: LAMMPS

   kim property destroy 2

.. note::

    If this command is called with an instance ID that does not exist, no
    error is raised.

**Dump**

The *kim property dump*  command can be used to write the content of all
currently defined property instances to a file:

.. code-block:: LAMMPS

   kim property dump file

For example,

.. code-block:: LAMMPS

   kim property dump results.edn

.. note::

    Issuing the *kim property dump* command clears all existing property
    instances from memory.

Citation of OpenKIM IMs
"""""""""""""""""""""""

When publishing results obtained using OpenKIM IMs researchers are requested
to cite the OpenKIM project :ref:`(Tadmor) <kim-mainpaper>`, KIM API
:ref:`(Elliott) <kim-api>`, and the specific IM codes used in the simulations,
in addition to the relevant scientific references for the IM. The citation
format for an IM is displayed on its page on
`OpenKIM <https://openkim.org>`_ along with the corresponding BibTex file, and
is automatically added to the LAMMPS citation reminder.

Citing the IM software (KIM infrastructure and specific PM or SM codes) used in
the simulation gives credit to the researchers who developed them and enables
open source efforts like OpenKIM to function.

Restrictions
""""""""""""

The *kim* command is part of the KIM package.  It is only enabled if LAMMPS is
built with that package.  A requirement for the KIM package, is the KIM API
library that must be downloaded from the
`OpenKIM website <https://openkim.org/kim-api/>`_ and installed before LAMMPS is
compiled.  When installing LAMMPS from binary, the kim-api package is a
dependency that is automatically downloaded and installed.  The *kim query*
command requires the *libcurl* library to be installed.  The *kim property*
command requires *Python* 3.6 or later and the *kim-property* python package to
be installed.  See the KIM section of the
:doc:`Packages details <Packages_details>` for details.

Furthermore, when using *kim* command to run KIM SMs, any packages required by
the native potential being used or other commands or fixes that it invokes must
be installed.

Related commands
""""""""""""""""

:doc:`pair_style kim <pair_kim>`

----------

.. _kim-mainpaper:

**(Tadmor)** Tadmor, Elliott, Sethna, Miller and Becker, JOM, 63, 17 (2011).
doi: `https://doi.org/10.1007/s11837-011-0102-6 <https://doi.org/10.1007/s11837-011-0102-6>`_

.. _kim-api:

**(Elliott)** Elliott, Tadmor and Bernstein, `https://openkim.org/kim-api <https://openkim.org/kim-api>`_ (2011)
doi: `https://doi.org/10.25950/FF8F563A <https://doi.org/10.25950/FF8F563A>`_
