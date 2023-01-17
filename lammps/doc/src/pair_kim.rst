.. index:: pair_style kim

pair_style kim command
======================

Syntax
""""""

.. code-block:: LAMMPS

   pair_style kim model

model = name of a KIM model (the KIM ID for models archived in OpenKIM)

Examples
""""""""

.. code-block:: LAMMPS

   pair_style kim SW_StillingerWeber_1985_Si__MO_405512056662_005
   pair_coeff * * Si

Description
"""""""""""

This pair style is a wrapper on the
`Open Knowledgebase of Interatomic Models (OpenKIM) <https://openkim.org>`_
repository of interatomic potentials to enable their use in LAMMPS scripts.

The preferred interface for using interatomic models archived in
OpenKIM is the :doc:`kim command <kim_commands>` interface. That
interface supports both "KIM Portable Models" (PMs) that conform to the
KIM API Portable Model Interface (PMI) and can be used by any
simulation code that conforms to the KIM API/PMI, and
"KIM Simulator Models" (SMs) that are natively implemented within a single
simulation code (like LAMMPS) and can only be used with it.
The *pair_style kim* command is limited to KIM PMs. It is
used by the :doc:`kim command <kim_commands>` interface as needed.

.. note::

   Since *pair_style kim* is called by *kim interactions* as needed,
   it is not recommended to be directly used in input scripts.

----------

The argument *model* is the name of the KIM PM.
For potentials archived in OpenKIM
this is the extended KIM ID (see :doc:`kim command <kim_commands>`
for details). LAMMPS can invoke any KIM PM, however there can
be incompatibilities (for example due to unit matching issues).
In the event of an incompatibility, the code will terminate with
an error message. Check both the LAMMPS and KIM log files for details.

Only a single *pair_coeff* command is used with the *kim* style, which
specifies the mapping of LAMMPS atom types to the species supported by
the KIM PM.  This is done by specifying *N* additional arguments
after the \* \* in the *pair_coeff* command, where *N* is the number of
LAMMPS atom types:

* N element names = mapping of KIM elements to atom types

For example, consider a KIM PM that supports Si and C species.
If the LAMMPS simulation has four atom types, where the first three are Si,
and the fourth is C, the following *pair_coeff* command would be used:

.. code-block:: LAMMPS

   pair_coeff * * Si Si Si C

The first two arguments must be \* \* so as to span all LAMMPS atom types.
The first three Si arguments map LAMMPS atom types 1, 2, and 3 to Si as
defined within KIM PM.  The final C argument maps LAMMPS atom type 4 to C.

----------

In addition to the usual LAMMPS error messages, the KIM library itself
may generate errors, which should be printed to the screen.  In this
case it is also useful to check the *kim.log* file for additional error
information.  The file *kim.log* should be generated in the same
directory where LAMMPS is running.

To download, build, and install the KIM library on your system, see
the *lib/kim/README* file.  Once you have done this and built LAMMPS
with the KIM package installed you can run the example input scripts
in *examples/kim*\ .

----------

Mixing, shift, table, tail correction, restart, rRESPA info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

This pair style does not support the :doc:`pair_modify <pair_modify>`
mix, shift, table, and tail options.

This pair style does not write its information to :doc:`binary restart files <restart>`, since KIM stores the potential parameters.
Thus, you need to re-specify the pair_style and pair_coeff commands in
an input script that reads a restart file.

This pair style can only be used via the *pair* keyword of the
:doc:`run_style respa <run_style>` command.  It does not support the
*inner*, *middle*, *outer* keywords.

----------

Restrictions
""""""""""""

This pair style is part of the KIM package. See details on
restrictions in :doc:`kim command <kim_commands>`.

This current version of pair_style kim is compatible with the
kim-api package version 2.0.0 and higher.

Related commands
""""""""""""""""

:doc:`pair_coeff <pair_coeff>`, :doc:`kim command <kim_commands>`

Default
"""""""

none
