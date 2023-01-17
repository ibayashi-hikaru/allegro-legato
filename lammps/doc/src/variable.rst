.. index:: variable

variable command
================

Syntax
""""""

.. parsed-literal::

   variable name style args ...

* name = name of variable to define
* style = *delete* or *index* or *loop* or *world* or *universe* or *uloop* or *string* or *format* or *getenv* or *file* or *atomfile* or *python* or *internal* or *equal* or *vector* or *atom*

  .. parsed-literal::

       *delete* = no args
       *index* args = one or more strings
       *loop* args = N
         N = integer size of loop, loop from 1 to N inclusive
       *loop* args = N pad
         N = integer size of loop, loop from 1 to N inclusive
         pad = all values will be same length, e.g. 001, 002, ..., 100
       *loop* args = N1 N2
         N1,N2 = loop from N1 to N2 inclusive
       *loop* args = N1 N2 pad
         N1,N2 = loop from N1 to N2 inclusive
         pad = all values will be same length, e.g. 050, 051, ..., 100
       *world* args = one string for each partition of processors
       *universe* args = one or more strings
       *uloop* args = N
         N = integer size of loop
       *uloop* args = N pad
         N = integer size of loop
         pad = all values will be same length, e.g. 001, 002, ..., 100
       *string* arg = one string
       *format* args = vname fstr
         vname = name of equal-style variable to evaluate
         fstr = C-style format string
       *getenv* arg = one string
       *file* arg = filename
       *atomfile* arg = filename
       *python* arg = function
       *internal* arg = numeric value
       *equal* or *vector* or *atom* args = one formula containing numbers, thermo keywords, math operations, group functions, atom values and vectors, compute/fix/variable references
         numbers = 0.0, 100, -5.4, 2.8e-4, etc
         constants = PI, version, on, off, true, false, yes, no
         thermo keywords = vol, ke, press, etc from :doc:`thermo_style <thermo_style>`
         math operators = (), -x, x+y, x-y, x\*y, x/y, x\^y, x%y,
                          x == y, x != y, x < y, x <= y, x > y, x >= y, x && y, x \|\| y, x \|\^ y, !x
         math functions = sqrt(x), exp(x), ln(x), log(x), abs(x),
                          sin(x), cos(x), tan(x), asin(x), acos(x), atan(x), atan2(y,x),
                          random(x,y,z), normal(x,y,z), ceil(x), floor(x), round(x)
                          ramp(x,y), stagger(x,y), logfreq(x,y,z), logfreq2(x,y,z),
                          logfreq3(x,y,z), stride(x,y,z), stride2(x,y,z,a,b,c),
                          vdisplace(x,y), swiggle(x,y,z), cwiggle(x,y,z)
         group functions = count(group), mass(group), charge(group),
                           xcm(group,dim), vcm(group,dim), fcm(group,dim),
                           bound(group,dir), gyration(group), ke(group),
                           angmom(group,dim), torque(group,dim),
                           inertia(group,dimdim), omega(group,dim)
         region functions = count(group,region), mass(group,region), charge(group,region),
                           xcm(group,dim,region), vcm(group,dim,region), fcm(group,dim,region),
                           bound(group,dir,region), gyration(group,region), ke(group,reigon),
                           angmom(group,dim,region), torque(group,dim,region),
                           inertia(group,dimdim,region), omega(group,dim,region)
         special functions = sum(x), min(x), max(x), ave(x), trap(x), slope(x), gmask(x), rmask(x), grmask(x,y), next(x), is_file(name)
         feature functions = is_available(category,feature), is_active(category,feature), is_defined(category,id)
         atom value = id[i], mass[i], type[i], mol[i], x[i], y[i], z[i], vx[i], vy[i], vz[i], fx[i], fy[i], fz[i], q[i]
         atom vector = id, mass, type, mol, x, y, z, vx, vy, vz, fx, fy, fz, q
         compute references = c_ID, c_ID[i], c_ID[i][j], C_ID, C_ID[i]
         fix references = f_ID, f_ID[i], f_ID[i][j], F_ID, F_ID[i]
         variable references = v_name, v_name[i]

Examples
""""""""

.. code-block:: LAMMPS

   variable x index run1 run2 run3 run4 run5 run6 run7 run8
   variable LoopVar loop $n
   variable beta equal temp/3.0
   variable b1 equal x[234]+0.5*vol
   variable b1 equal "x[234] + 0.5*vol"
   variable b equal xcm(mol1,x)/2.0
   variable b equal c_myTemp
   variable b atom x*y/vol
   variable foo string myfile
   variable foo internal 3.5
   variable myPy python increase
   variable f file values.txt
   variable temp world 300.0 310.0 320.0 ${Tfinal}
   variable x universe 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
   variable x uloop 15 pad
   variable str format x %.6g
   variable x delete

Description
"""""""""""

This command assigns one or more strings to a variable name for
evaluation later in the input script or during a simulation.

Variables can thus be useful in several contexts.  A variable can be
defined and then referenced elsewhere in an input script to become
part of a new input command.  For variable styles that store multiple
strings, the :doc:`next <next>` command can be used to increment which
string is assigned to the variable.  Variables of style *equal* store
a formula which when evaluated produces a single numeric value which
can be output either directly (see the :doc:`print <print>`, :doc:`fix print <fix_print>`, and :doc:`run every <run>` commands) or as part
of thermodynamic output (see the :doc:`thermo_style <thermo_style>`
command), or used as input to an averaging fix (see the :doc:`fix ave/time <fix_ave_time>` command).  Variables of style *vector*
store a formula which produces a vector of such values which can be
used as input to various averaging fixes, or elements of which can be
part of thermodynamic output.  Variables of style *atom* store a
formula which when evaluated produces one numeric value per atom which
can be output to a dump file (see the :doc:`dump custom <dump>` command)
or used as input to an averaging fix (see the :doc:`fix ave/chunk <fix_ave_chunk>` and :doc:`fix ave/atom <fix_ave_atom>`
commands).  Variables of style *atomfile* can be used anywhere in an
input script that atom-style variables are used; they get their
per-atom values from a file rather than from a formula.  Variables of
style *python* can be hooked to Python functions using code you
provide, so that the variable gets its value from the evaluation of
the Python code.  Variables of style *internal* are used by a few
commands which set their value directly.

.. note::

   As discussed on the :doc:`Commands parse <Commands_parse>` doc
   page, an input script can use "immediate" variables, specified as
   $(formula) with parenthesis, where the formula has the same syntax as
   equal-style variables described on this page.  This is a convenient
   way to evaluate a formula immediately without using the variable
   command to define a named variable and then evaluate that
   variable. See below for a more detailed discussion of this feature.

In the discussion that follows, the "name" of the variable is the
arbitrary string that is the first argument in the variable command.
This name can only contain alphanumeric characters and underscores.
The "string" is one or more of the subsequent arguments.  The "string"
can be simple text as in the first example above, it can contain other
variables as in the second example, or it can be a formula as in the third
example.  The "value" is the numeric quantity resulting from
evaluation of the string.  Note that the same string can generate
different values when it is evaluated at different times during a
simulation.

.. note::

   When an input script line is encountered that defines a variable
   of style *equal* or *vector* or *atom* or *python* that contains a
   formula or Python code, the formula is NOT immediately evaluated.  It
   will be evaluated every time when the variable is **used** instead.  If
   you simply want to evaluate a formula in place you can use as
   so-called. See the section below about "Immediate Evaluation of
   Variables" for more details on the topic.  This is also true of a
   *format* style variable since it evaluates another variable when it is
   invoked.

Variables of style *equal* and *vector* and *atom* can be used as
inputs to various other commands which evaluate their formulas as
needed, e.g. at different timesteps during a :doc:`run <run>`.

Variables of style *internal* can be used in place of an equal-style
variable, except by commands that set the value stored by the
internal-style variable.  Thus any command that states it can use an
equal-style variable as an argument, can also use an internal-style
variable.  This means that when the command evaluates the variable, it
will use the value set (internally) by another command.

Variables of style *python* can be used in place of an equal-style
variable so long as the associated Python function, as defined by the
:doc:`python <python>` command, returns a numeric value.  Thus any
command that states it can use an equal-style variable as an argument,
can also use such a python-style variable.  This means that when the
LAMMPS command evaluates the variable, the Python function will be
executed.

.. note::

   When a variable command is encountered in the input script and
   the variable name has already been specified, the command is ignored.
   This means variables can NOT be re-defined in an input script (with
   two exceptions, read further).  This is to allow an input script to be
   processed multiple times without resetting the variables; see the
   :doc:`jump <jump>` or :doc:`include <include>` commands.  It also means
   that using the :doc:`command-line switch <Run_options>` -var will
   override a corresponding index variable setting in the input script.

There are two exceptions to this rule.  First, variables of style
*string*, *getenv*, *internal*, *equal*, *vector*, *atom*, and
*python* ARE redefined each time the command is encountered.  This
allows these style of variables to be redefined multiple times in an
input script.  In a loop, this means the formula associated with an
*equal* or *atom* style variable can change if it contains a
substitution for another variable, e.g. $x or v_x.

Second, as described below, if a variable is iterated on to the end of
its list of strings via the :doc:`next <next>` command, it is removed
from the list of active variables, and is thus available to be
re-defined in a subsequent variable command.  The *delete* style does
the same thing.

----------

The :doc:`Commands parse <Commands_parse>` page explains how
occurrences of a variable name in an input script line are replaced by
the variable's string.  The variable name can be referenced as $x if
the name "x" is a single character, or as ${LoopVar} if the name
"LoopVar" is one or more characters.

As described below, for variable styles *index*, *loop*, *file*,
*universe*, and *uloop*, which string is assigned to a variable can be
incremented via the :doc:`next <next>` command.  When there are no more
strings to assign, the variable is exhausted and a flag is set that
causes the next :doc:`jump <jump>` command encountered in the input
script to be skipped.  This enables the construction of simple loops
in the input script that are iterated over and then exited from.

As explained above, an exhausted variable can be re-used in an input
script.  The *delete* style also removes the variable, the same as if
it were exhausted, allowing it to be redefined later in the input
script or when the input script is looped over.  This can be useful
when breaking out of a loop via the :doc:`if <if>` and :doc:`jump <jump>`
commands before the variable would become exhausted.  For example,

.. code-block:: LAMMPS

   label       loop
   variable    a loop 5
   print       "A = $a"
   if          "$a > 2" then "jump in.script break"
   next        a
   jump        in.script loop
   label       break
   variable    a delete

----------

This section describes how all the various variable styles are defined
and what they store.  Except for the *equal* and *vector* and *atom*
styles, which are explained in the next section.

Many of the styles store one or more strings.  Note that a single
string can contain spaces (multiple words), if it is enclosed in
quotes in the variable command.  When the variable is substituted for
in another input script command, its returned string will then be
interpreted as multiple arguments in the expanded command.

For the *index* style, one or more strings are specified.  Initially,
the first string is assigned to the variable.  Each time a
:doc:`next <next>` command is used with the variable name, the next
string is assigned.  All processors assign the same string to the
variable.

*Index* style variables with a single string value can also be set by
using the :doc:`command-line switch -var <Run_options>`.

The *loop* style is identical to the *index* style except that the
strings are the integers from 1 to N inclusive, if only one argument N
is specified.  This allows generation of a long list of runs
(e.g. 1000) without having to list N strings in the input script.
Initially, the string "1" is assigned to the variable.  Each time a
:doc:`next <next>` command is used with the variable name, the next
string ("2", "3", etc) is assigned.  All processors assign the same
string to the variable.  The *loop* style can also be specified with
two arguments N1 and N2.  In this case the loop runs from N1 to N2
inclusive, and the string N1 is initially assigned to the variable.
N1 <= N2 and N2 >= 0 is required.

For the *world* style, one or more strings are specified.  There must
be one string for each processor partition or "world".  LAMMPS can be
run with multiple partitions via the :doc:`-partition command-line switch <Run_options>`.  This variable command assigns one string to
each world.  All processors in the world are assigned the same string.
The next command cannot be used with *equal* style variables, since
there is only one value per world.  This style of variable is useful
when you wish to run different simulations on different partitions, or
when performing a parallel tempering simulation (see the
:doc:`temper <temper>` command), to assign different temperatures to
different partitions.

For the *universe* style, one or more strings are specified.  There
must be at least as many strings as there are processor partitions or
"worlds".  LAMMPS can be run with multiple partitions via the
:doc:`-partition command-line switch <Run_options>`.  This variable
command initially assigns one string to each world.  When a
:doc:`next <next>` command is encountered using this variable, the first
processor partition to encounter it, is assigned the next available
string.  This continues until all the variable strings are consumed.
Thus, this command can be used to run 50 simulations on 8 processor
partitions.  The simulations will be run one after the other on
whatever partition becomes available, until they are all finished.
*Universe* style variables are incremented using the files
"tmp.lammps.variable" and "tmp.lammps.variable.lock" which you will
see in your directory during such a LAMMPS run.

The *uloop* style is identical to the *universe* style except that the
strings are the integers from 1 to N.  This allows generation of long
list of runs (e.g. 1000) without having to list N strings in the input
script.

For the *string* style, a single string is assigned to the variable.
Two differences between this style and using the *index* style exist:
a variable with *string* style can be redefined, e.g. by another command later
in the input script, or if the script is read again in a loop. The other
difference is that *string* performs variable substitution even if the
string parameter is quoted.

For the *format* style, an equal-style variable is specified along
with a C-style format string, e.g. "%f" or "%.10g", which must be
appropriate for formatting a double-precision floating-point value.
The default format is "%.15g".  This variable style allows an
equal-style variable to be formatted precisely when it is evaluated.

If you simply wish to print a variable value with desired precision to
the screen or logfile via the :doc:`print <print>` or :doc:`fix print <fix_print>` commands, you can also do this by specifying an
"immediate" variable with a trailing colon and format string, as part
of the string argument of those commands.  This is explained on the
:doc:`Commands parse <Commands_parse>` doc page.

For the *getenv* style, a single string is assigned to the variable
which should be the name of an environment variable.  When the
variable is evaluated, it returns the value of the environment
variable, or an empty string if it not defined.  This style of
variable can be used to adapt the behavior of LAMMPS input scripts via
environment variable settings, or to retrieve information that has
been previously stored with the :doc:`shell putenv <shell>` command.
Note that because environment variable settings are stored by the
operating systems, they persist beyond a :doc:`clear <clear>` command.

For the *file* style, a filename is provided which contains a list of
strings to assign to the variable, one per line.  The strings can be
numeric values if desired.  See the discussion of the next() function
below for equal-style variables, which will convert the string of a
file-style variable into a numeric value in a formula.

When a file-style variable is defined, the file is opened and the
string on the first line is read and stored with the variable.  This
means the variable can then be evaluated as many times as desired and
will return that string.  There are two ways to cause the next string
from the file to be read: use the :doc:`next <next>` command or the
next() function in an equal- or atom-style variable, as discussed
below.

The rules for formatting the file are as follows.  A comment character
"#" can be used anywhere on a line; text starting with the comment
character is stripped.  Blank lines are skipped.  The first "word" of
a non-blank line, delimited by white-space, is the "string" assigned
to the variable.

For the *atomfile* style, a filename is provided which contains one or
more sets of values, to assign on a per-atom basis to the variable.
The format of the file is described below.

When an atomfile-style variable is defined, the file is opened and the
first set of per-atom values are read and stored with the variable.
This means the variable can then be evaluated as many times as desired
and will return those values.  There are two ways to cause the next
set of per-atom values from the file to be read: use the
:doc:`next <next>` command or the next() function in an atom-style
variable, as discussed below.

The rules for formatting the file are as follows.  Each time a set of
per-atom values is read, a non-blank line is searched for in the file.
The file is read line by line but only up to 254 characters are used.
The rest are ignored.  A comment character "#" can be used anywhere
on a line and all text following and the "#" character are ignored;
text starting with the comment character is stripped.  Blank lines
are skipped.  The first "word" of a non-blank line, delimited by
white-space, is read as the count N of per-atom lines to immediately
follow.  N can be the total number of atoms in the system, or only a
subset.  The next N lines have the following format

.. parsed-literal::

   ID value

where ID is an atom ID and value is the per-atom numeric value that
will be assigned to that atom.  IDs can be listed in any order.

.. note::

   Every time a set of per-atom lines is read, the value for all
   atoms is first set to 0.0.  Thus values for atoms whose ID does not
   appear in the set, will remain 0.0.

For the *python* style a Python function name is provided.  This needs
to match a function name specified in a :doc:`python <python>` command
which returns a value to this variable as defined by its *return*
keyword.  For example these two commands would be self-consistent:

.. code-block:: LAMMPS

   variable foo python myMultiply
   python myMultiply return v_foo format f file funcs.py

The two commands can appear in either order so long as both are
specified before the Python function is invoked for the first time.

Each time the variable is evaluated, the associated Python function is
invoked, and the value it returns is also returned by the variable.
Since the Python function can use other LAMMPS variables as input, or
query interal LAMMPS quantities to perform its computation, this means
the variable can return a different value each time it is evaluated.

The type of value stored in the variable is determined by the *format*
keyword of the :doc:`python <python>` command.  It can be an integer
(i), floating point (f), or string (s) value.  As mentioned above, if
it is a numeric value (integer or floating point), then the
python-style variable can be used in place of an equal-style variable
anywhere in an input script, e.g. as an argument to another command
that allows for equal-style variables.

For the *internal* style a numeric value is provided.  This value will
be assigned to the variable until a LAMMPS command sets it to a new
value.  There are currently only two LAMMPS commands that require
*internal* variables as inputs, because they reset them:
:doc:`create_atoms <create_atoms>` and :doc:`fix controller <fix_controller>`.  As mentioned above, an
internal-style variable can be used in place of an equal-style
variable anywhere else in an input script, e.g. as an argument to
another command that allows for equal-style variables.

----------

For the *equal* and *vector* and *atom* styles, a single string is
specified which represents a formula that will be evaluated afresh
each time the variable is used.  If you want spaces in the string,
enclose it in double quotes so the parser will treat it as a single
argument.  For *equal*\ -style variables the formula computes a scalar
quantity, which becomes the value of the variable whenever it is
evaluated.  For *vector*\ -style variables the formula must compute a
vector of quantities, which becomes the value of the variable whenever
it is evaluated.  The calculated vector can be of length one, but it
cannot be a simple scalar value like that produced by an equal-style
compute.  I.e. the formula for a vector-style variable must have at
least one quantity in it that refers to a global vector produced by a
compute, fix, or other vector-style variable.  For *atom*\ -style
variables the formula computes one quantity for each atom whenever it
is evaluated.

Note that *equal*, *vector*, and *atom* variables can produce
different values at different stages of the input script or at
different times during a run.  For example, if an *equal* variable is
used in a :doc:`fix print <fix_print>` command, different values could
be printed each timestep it was invoked.  If you want a variable to be
evaluated immediately, so that the result is stored by the variable
instead of the string, see the section below on "Immediate Evaluation
of Variables".

The next command cannot be used with *equal* or *vector* or *atom*
style variables, since there is only one string.

The formula for an *equal*, *vector*, or *atom* variable can contain a
variety of quantities.  The syntax for each kind of quantity is
simple, but multiple quantities can be nested and combined in various
ways to build up formulas of arbitrary complexity.  For example, this
is a valid (though strange) variable formula:

.. code-block:: LAMMPS

   variable x equal "pe + c_MyTemp / vol^(1/3)"

Specifically, a formula can contain numbers, constants, thermo
keywords, math operators, math functions, group functions, region
functions, atom values, atom vectors, compute references, fix
references, and references to other variables.

+--------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Number             | 0.2, 100, 1.0e20, -15.4, etc                                                                                                                                                                                                                                                                                                                              |
+--------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Constant           | PI, version, on, off, true, false, yes, no                                                                                                                                                                                                                                                                                                                |
+--------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Thermo keywords    | vol, pe, ebond, etc                                                                                                                                                                                                                                                                                                                                       |
+--------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Math operators     | (), -x, x+y, x-y, x\*y, x/y, x\^y, x%y,      x == y, x != y, x < y, x <= y, x > y, x >= y, x && y, x \|\| y, x \|\^ y, !x                                                                                                                                                                                                                                 |
+--------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Math functions     | sqrt(x), exp(x), ln(x), log(x), abs(x),      sin(x), cos(x), tan(x), asin(x), acos(x), atan(x), atan2(y,x),      random(x,y,z), normal(x,y,z), ceil(x), floor(x), round(x),      ramp(x,y), stagger(x,y), logfreq(x,y,z), logfreq2(x,y,z),      logfreq3(x,y,z), stride(x,y,z), stride2(x,y,z,a,b,c),      vdisplace(x,y), swiggle(x,y,z), cwiggle(x,y,z) |
+--------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Group functions    | count(ID), mass(ID), charge(ID), xcm(ID,dim),      vcm(ID,dim), fcm(ID,dim), bound(ID,dir),      gyration(ID), ke(ID), angmom(ID,dim), torque(ID,dim),      inertia(ID,dimdim), omega(ID,dim)                                                                                                                                                             |
+--------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Region functions   | count(ID,IDR), mass(ID,IDR), charge(ID,IDR),      xcm(ID,dim,IDR), vcm(ID,dim,IDR), fcm(ID,dim,IDR),      bound(ID,dir,IDR), gyration(ID,IDR), ke(ID,IDR),      angmom(ID,dim,IDR), torque(ID,dim,IDR),      inertia(ID,dimdim,IDR), omega(ID,dim,IDR)                                                                                                    |
+--------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Special functions  | sum(x), min(x), max(x), ave(x), trap(x),      slope(x), gmask(x), rmask(x), grmask(x,y), next(x)                                                                                                                                                                                                                                                          |
+--------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Atom values        | id[i], mass[i], type[i], mol[i], x[i], y[i], z[i],              vx[i], vy[i], vz[i], fx[i], fy[i], fz[i], q[i]                                                                                                                                                                                                                                            |
+--------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Atom vectors       | id, mass, type, mol, x, y, z, vx, vy, vz, fx, fy, fz, q                                                                                                                                                                                                                                                                                                   |
+--------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Compute references | c_ID, c_ID[i], c_ID[i][j], C_ID, C_ID[i]                                                                                                                                                                                                                                                                                                                  |
+--------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Fix references     | f_ID, f_ID[i], f_ID[i][j], F_ID, F_ID[i]                                                                                                                                                                                                                                                                                                                  |
+--------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Other variables    | v_name, v_name[i]                                                                                                                                                                                                                                                                                                                                         |
+--------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Most of the formula elements produce a scalar value.  Some produce a
global or per-atom vector of values.  Global vectors can be produced
by computes or fixes or by other vector-style variables.  Per-atom
vectors are produced by atom vectors, compute references that
represent a per-atom vector, fix references that represent a per-atom
vector, and variables that are atom-style variables.  Math functions
that operate on scalar values produce a scalar value; math function
that operate on global or per-atom vectors do so element-by-element
and produce a global or per-atom vector.

A formula for equal-style variables cannot use any formula element
that produces a global or per-atom vector.  A formula for a
vector-style variable can use formula elements that produce either a
scalar value or a global vector value, but cannot use a formula
element that produces a per-atom vector.  A formula for an atom-style
variable can use formula elements that produce either a scalar value
or a per-atom vector, but not one that produces a global vector.
Atom-style variables are evaluated by other commands that define a
:doc:`group <group>` on which they operate, e.g. a :doc:`dump <dump>` or
:doc:`compute <compute>` or :doc:`fix <fix>` command.  When they invoke
the atom-style variable, only atoms in the group are included in the
formula evaluation.  The variable evaluates to 0.0 for atoms not in
the group.

----------

Numbers, constants, and thermo keywords
---------------------------------------

Numbers can contain digits, scientific notation
(3.0e20,3.0e-20,3.0E20,3.0E-20), and leading minus signs.

Constants are set at compile time and cannot be changed. *PI* will
return the number 3.14159265358979323846; *on*, *true* or *yes* will
return 1.0; *off*, *false* or *no* will return 0.0; *version* will
return a numeric version code of the current LAMMPS version (e.g.
version 2 Sep 2015 will return the number 20150902). The corresponding
value for newer versions of LAMMPS will be larger, for older versions
of LAMMPS will be smaller. This can be used to have input scripts
adapt automatically to LAMMPS versions, when non-backwards compatible
syntax changes are introduced. Here is an illustrative example (which
will not work, since the *version* has been introduced more recently):

.. code-block:: LAMMPS

   if $(version<20140513) then "communicate vel yes" else "comm_modify vel yes"

The thermo keywords allowed in a formula are those defined by the
:doc:`thermo_style custom <thermo_style>` command.  Thermo keywords that
require a :doc:`compute <compute>` to calculate their values such as
"temp" or "press", use computes stored and invoked by the
:doc:`thermo_style <thermo_style>` command.  This means that you can
only use those keywords in a variable if the style you are using with
the thermo_style command (and the thermo keywords associated with that
style) also define and use the needed compute.  Note that some thermo
keywords use a compute indirectly to calculate their value (e.g. the
enthalpy keyword uses temp, pe, and pressure).  If a variable is
evaluated directly in an input script (not during a run), then the
values accessed by the thermo keyword must be current.  See the
discussion below about "Variable Accuracy".

----------

Math Operators
--------------

Math operators are written in the usual way, where the "x" and "y" in
the examples can themselves be arbitrarily complex formulas, as in the
examples above.  In this syntax, "x" and "y" can be scalar values or
per-atom vectors.  For example, "ke/natoms" is the division of two
scalars, where "vy+vz" is the element-by-element sum of two per-atom
vectors of y and z velocities.

Operators are evaluated left to right and have the usual C-style
precedence: unary minus and unary logical NOT operator "!" have the
highest precedence, exponentiation "\^" is next; multiplication and
division and the modulo operator "%" are next; addition and
subtraction are next; the 4 relational operators "<", "<=", ">", and
">=" are next; the two remaining relational operators "==" and "!="
are next; then the logical AND operator "&&"; and finally the logical
OR operator "||" and logical XOR (exclusive or) operator "\|^" have the
lowest precedence.  Parenthesis can be used to group one or more
portions of a formula and/or enforce a different order of evaluation
than what would occur with the default precedence.

.. note::

   Because a unary minus is higher precedence than exponentiation,
   the formula "-2\^2" will evaluate to 4, not -4.  This convention is
   compatible with some programming languages, but not others.  As
   mentioned, this behavior can be easily overridden with parenthesis;
   the formula "-(2\^2)" will evaluate to -4.

The 6 relational operators return either a 1.0 or 0.0 depending on
whether the relationship between x and y is TRUE or FALSE.  For
example the expression x<10.0 in an atom-style variable formula will
return 1.0 for all atoms whose x-coordinate is less than 10.0, and 0.0
for the others.  The logical AND operator will return 1.0 if both its
arguments are non-zero, else it returns 0.0.  The logical OR operator
will return 1.0 if either of its arguments is non-zero, else it
returns 0.0.  The logical XOR operator will return 1.0 if one of its
arguments is zero and the other non-zero, else it returns 0.0.  The
logical NOT operator returns 1.0 if its argument is 0.0, else it
returns 0.0.

These relational and logical operators can be used as a masking or
selection operation in a formula.  For example, the number of atoms
whose properties satisfy one or more criteria could be calculated by
taking the returned per-atom vector of ones and zeroes and passing it
to the :doc:`compute reduce <compute_reduce>` command.

----------

Math Functions
--------------

Math functions are specified as keywords followed by one or more
parenthesized arguments "x", "y", "z", each of which can themselves be
arbitrarily complex formulas.  In this syntax, the arguments can
represent scalar values or global vectors or per-atom vectors.  In the
latter case, the math operation is performed on each element of the
vector.  For example, "sqrt(natoms)" is the sqrt() of a scalar, where
"sqrt(y\*z)" yields a per-atom vector with each element being the
sqrt() of the product of one atom's y and z coordinates.

Most of the math functions perform obvious operations.  The ln() is
the natural log; log() is the base 10 log.

The random(x,y,z) function takes 3 arguments: x = lo, y = hi, and z =
seed.  It generates a uniform random number between lo and hi.  The
normal(x,y,z) function also takes 3 arguments: x = mu, y = sigma, and
z = seed.  It generates a Gaussian variate centered on mu with
variance sigma\^2.  In both cases the seed is used the first time the
internal random number generator is invoked, to initialize it.  For
equal-style and vector-style variables, every processor uses the same
seed so that they each generate the same sequence of random numbers.
For atom-style variables, a unique seed is created for each processor,
based on the specified seed.  This effectively generates a different
random number for each atom being looped over in the atom-style
variable.

.. note::

   Internally, there is just one random number generator for all
   equal-style and vector-style variables and another one for all
   atom-style variables.  If you define multiple variables (of each
   style) which use the random() or normal() math functions, then the
   internal random number generators will only be initialized once, which
   means only one of the specified seeds will determine the sequence of
   generated random numbers.

The ceil(), floor(), and round() functions are those in the C math
library.  Ceil() is the smallest integer not less than its argument.
Floor() if the largest integer not greater than its argument.  Round()
is the nearest integer to its argument.

The ramp(x,y) function uses the current timestep to generate a value
linearly interpolated between the specified x,y values over the course
of a run, according to this formula:

.. parsed-literal::

   value = x + (y-x) \* (timestep-startstep) / (stopstep-startstep)

The run begins on startstep and ends on stopstep.  Startstep and
stopstep can span multiple runs, using the *start* and *stop* keywords
of the :doc:`run <run>` command.  See the :doc:`run <run>` command for
details of how to do this.

The stagger(x,y) function uses the current timestep to generate a new
timestep.  X,y > 0 and x > y are required.  The generated timesteps
increase in a staggered fashion, as the sequence
x,x+y,2x,2x+y,3x,3x+y,etc.  For any current timestep, the next
timestep in the sequence is returned.  Thus if stagger(1000,100) is
used in a variable by the :doc:`dump_modify every <dump_modify>`
command, it will generate the sequence of output timesteps:

.. parsed-literal::

   100,1000,1100,2000,2100,3000,etc

The logfreq(x,y,z) function uses the current timestep to generate a
new timestep.  X,y,z > 0 and y < z are required.  The generated
timesteps are on a base-z logarithmic scale, starting with x, and the
y value is how many of the z-1 possible timesteps within one
logarithmic interval are generated.  I.e. the timesteps follow the
sequence x,2x,3x,...y\*x,x\*z,2x\*z,3x\*z,...y\*x\*z,x\*z\^2,2x\*z\^2,etc.  For
any current timestep, the next timestep in the sequence is returned.
Thus if logfreq(100,4,10) is used in a variable by the :doc:`dump_modify every <dump_modify>` command, it will generate this sequence of
output timesteps:

.. parsed-literal::

   100,200,300,400,1000,2000,3000,4000,10000,20000,etc

The logfreq2(x,y,z) function is similar to logfreq, except a single
logarithmic interval is divided into y equally-spaced timesteps and
all of them are output.  Y < z is not required.  Thus, if
logfreq2(100,18,10) is used in a variable by the :doc:`dump_modify every <dump_modify>` command, then the interval between 100 and
1000 is divided as 900/18 = 50 steps, and it will generate the
sequence of output timesteps:

.. parsed-literal::

   100,150,200,...950,1000,1500,2000,...9500,10000,15000,etc

The logfreq3(x,y,z) function generates y points between x and z (inclusive),
that are separated by a multiplicative ratio: (z/x)\^(1/(y-1)). Constraints
are: x,z > 0, y > 1, z-x >= y-1. For eg., if logfreq3(10,25,1000) is used in
a variable by the :doc:`fix print <fix_print>` command, then the interval
between 10 and 1000 is divided into 24 parts with a multiplicative
separation of ~1.21, and it will generate the following sequence of output
timesteps:

.. parsed-literal::

   10, 13, 15, 18, 22, 27, 32,...384, 465, 563, 682, 826, 1000

The stride(x,y,z) function uses the current timestep to generate a new
timestep.  X,y >= 0 and z > 0 and x <= y are required.  The generated
timesteps increase in increments of z, from x to y, i.e. it generates
the sequence x,x+z,x+2z,...,y.  If y-x is not a multiple of z, then
similar to the way a for loop operates, the last value will be one
that does not exceed y.  For any current timestep, the next timestep
in the sequence is returned.  Thus if stride(1000,2000,100) is used
in a variable by the :doc:`dump_modify every <dump_modify>` command, it
will generate the sequence of output timesteps:

.. parsed-literal::

   1000,1100,1200, ... ,1900,2000

The stride2(x,y,z,a,b,c) function is similar to the stride() function
except it generates two sets of strided timesteps, one at a coarser
level and one at a finer level.  Thus it is useful for debugging,
e.g. to produce output every timestep at the point in simulation when
a problem occurs.  X,y >= 0 and z > 0 and x <= y are required, as are
a,b >= 0 and c > 0 and a < b.  Also, a >= x and b <= y are required so
that the second stride is inside the first.  The generated timesteps
increase in increments of z, starting at x, until a is reached.  At
that point the timestep increases in increments of c, from a to b,
then after b, increments by z are resumed until y is reached.  For any
current timestep, the next timestep in the sequence is returned.  Thus
if stride2(1000,2000,100,1350,1360,1) is used in a variable by the
:doc:`dump_modify every <dump_modify>` command, it will generate the
sequence of output timesteps:

.. parsed-literal::

   1000,1100,1200,1300,1350,1351,1352, ... 1359,1360,1400,1500, ... ,2000

The vdisplace(x,y) function takes 2 arguments: x = value0 and y =
velocity, and uses the elapsed time to change the value by a linear
displacement due to the applied velocity over the course of a run,
according to this formula:

.. parsed-literal::

   value = value0 + velocity\*(timestep-startstep)\*dt

where dt = the timestep size.

The run begins on startstep.  Startstep can span multiple runs, using
the *start* keyword of the :doc:`run <run>` command.  See the
:doc:`run <run>` command for details of how to do this.  Note that the
:doc:`thermo_style <thermo_style>` keyword elaplong =
timestep-startstep.

The swiggle(x,y,z) and cwiggle(x,y,z) functions each take 3 arguments:
x = value0, y = amplitude, z = period.  They use the elapsed time to
oscillate the value by a sin() or cos() function over the course of a
run, according to one of these formulas, where omega = 2 PI / period:

.. parsed-literal::

   value = value0 + Amplitude \* sin(omega\*(timestep-startstep)\*dt)
   value = value0 + Amplitude \* (1 - cos(omega\*(timestep-startstep)\*dt))

where dt = the timestep size.

The run begins on startstep.  Startstep can span multiple runs, using
the *start* keyword of the :doc:`run <run>` command.  See the
:doc:`run <run>` command for details of how to do this.  Note that the
:doc:`thermo_style <thermo_style>` keyword elaplong =
timestep-startstep.

----------

Group and Region Functions
--------------------------

Group functions are specified as keywords followed by one or two
parenthesized arguments.  The first argument *ID* is the group-ID.
The *dim* argument, if it exists, is *x* or *y* or *z*\ .  The *dir*
argument, if it exists, is *xmin*, *xmax*, *ymin*, *ymax*, *zmin*, or
*zmax*\ .  The *dimdim* argument, if it exists, is *xx* or *yy* or *zz*
or *xy* or *yz* or *xz*\ .

The group function count() is the number of atoms in the group.  The
group functions mass() and charge() are the total mass and charge of
the group.  Xcm() and vcm() return components of the position and
velocity of the center of mass of the group.  Fcm() returns a
component of the total force on the group of atoms.  Bound() returns
the min/max of a particular coordinate for all atoms in the group.
Gyration() computes the radius-of-gyration of the group of atoms.  See
the :doc:`compute gyration <compute_gyration>` command for a definition
of the formula.  Angmom() returns components of the angular momentum
of the group of atoms around its center of mass.  Torque() returns
components of the torque on the group of atoms around its center of
mass, based on current forces on the atoms.  Inertia() returns one of
6 components of the symmetric inertia tensor of the group of atoms
around its center of mass, ordered as Ixx,Iyy,Izz,Ixy,Iyz,Ixz.
Omega() returns components of the angular velocity of the group of
atoms around its center of mass.

Region functions are specified exactly the same way as group functions
except they take an extra final argument *IDR* which is the region ID.
The function is computed for all atoms that are in both the group and
the region.  If the group is "all", then the only criteria for atom
inclusion is that it be in the region.

----------

Special Functions
-----------------

Special functions take specific kinds of arguments, meaning their
arguments cannot be formulas themselves.

The is_file(x) function is a test whether 'x' is a (readable) file
and returns 1 in this case, otherwise it returns 0.  For that 'x'
is taken as a literal string and must not have any blanks in it.

The sum(x), min(x), max(x), ave(x), trap(x), and slope(x) functions
each take 1 argument which is of the form "c_ID" or "c_ID[N]" or
"f_ID" or "f_ID[N]" or "v_name".  The first two are computes and the
second two are fixes; the ID in the reference should be replaced by
the ID of a compute or fix defined elsewhere in the input script.  The
compute or fix must produce either a global vector or array.  If it
produces a global vector, then the notation without "[N]" should be
used.  If it produces a global array, then the notation with "[N]"
should be used, when N is an integer, to specify which column of the
global array is being referenced.  The last form of argument "v_name"
is for a vector-style variable where "name" is replaced by the name of
the variable.

These functions operate on a global vector of inputs and reduce it to
a single scalar value.  This is analogous to the operation of the
:doc:`compute reduce <compute_reduce>` command, which performs similar
operations on per-atom and local vectors.

The sum() function calculates the sum of all the vector elements.  The
min() and max() functions find the minimum and maximum element
respectively.  The ave() function is the same as sum() except that it
divides the result by the length of the vector.

The trap() function is the same as sum() except the first and last
elements are multiplied by a weighting factor of 1/2 when performing
the sum.  This effectively implements an integration via the
trapezoidal rule on the global vector of data.  I.e. consider a set of
points, equally spaced by 1 in their x coordinate: (1,V1), (2,V2),
..., (N,VN), where the Vi are the values in the global vector of
length N.  The integral from 1 to N of these points is trap().  When
appropriately normalized by the timestep size, this function is useful
for calculating integrals of time-series data, like that generated by
the :doc:`fix ave/correlate <fix_ave_correlate>` command.

The slope() function uses linear regression to fit a line to the set
of points, equally spaced by 1 in their x coordinate: (1,V1), (2,V2),
..., (N,VN), where the Vi are the values in the global vector of
length N.  The returned value is the slope of the line.  If the line
has a single point or is vertical, it returns 1.0e20.

The gmask(x) function takes 1 argument which is a group ID.  It
can only be used in atom-style variables.  It returns a 1 for
atoms that are in the group, and a 0 for atoms that are not.

The rmask(x) function takes 1 argument which is a region ID.  It can
only be used in atom-style variables.  It returns a 1 for atoms that
are in the geometric region, and a 0 for atoms that are not.

The grmask(x,y) function takes 2 arguments.  The first is a group ID,
and the second is a region ID.  It can only be used in atom-style
variables.  It returns a 1 for atoms that are in both the group and
region, and a 0 for atoms that are not in both.

The next(x) function takes 1 argument which is a variable ID (not
"v_foo", just "foo").  It must be for a file-style or atomfile-style
variable.  Each time the next() function is invoked (i.e. each time
the equal-style or atom-style variable is evaluated), the following
steps occur.

For file-style variables, the current string value stored by the
file-style variable is converted to a numeric value and returned by
the function.  And the next string value in the file is read and
stored.  Note that if the line previously read from the file was not a
numeric string, then it will typically evaluate to 0.0, which is
likely not what you want.

For atomfile-style variables, the current per-atom values stored by
the atomfile-style variable are returned by the function.  And the
next set of per-atom values in the file is read and stored.

Since file-style and atomfile-style variables read and store the first
line of the file or first set of per-atoms values when they are
defined in the input script, these are the value(s) that will be
returned the first time the next() function is invoked.  If next() is
invoked more times than there are lines or sets of lines in the file,
the variable is deleted, similar to how the :doc:`next <next>` command
operates.

----------

Feature Functions
-----------------

Feature functions allow to probe the running LAMMPS executable for
whether specific features are either active, defined, or available.
The functions take two arguments, a *category* and a corresponding
*argument*\ . The arguments are strings thus cannot be formulas
themselves (only $-style immediate variable expansion is possible).
Return value is either 1.0 or 0.0 depending on whether the function
evaluates to true or false, respectively.

The *is_active()* function allows to query for active settings which
are grouped by categories. Currently supported categories and
arguments are:

* *package* (argument = *gpu* or *intel* or *kokkos* or *omp*\ )
* *newton* (argument = *pair* or *bond* or *any*\ )
* *pair* (argument = *single* or *respa* or *manybody* or *tail* or *shift*\ )
* *comm_style* (argument = *brick* or *tiled*\ )
* *min_style* (argument = any of the compiled in minimizer styles)
* *run_style* (argument = any of the compiled in run styles)
* *atom_style* (argument = any of the compiled in atom styles)
* *pair_style* (argument = any of the compiled in pair styles)
* *bond_style* (argument = any of the compiled in bond styles)
* *angle_style* (argument = any of the compiled in angle styles)
* *dihedral_style* (argument = any of the compiled in dihedral styles)
* *improper_style* (argument = any of the compiled in improper styles)
* *kspace_style* (argument = any of the compiled in kspace styles)

Most of the settings are self-explanatory, the *single* argument in the
*pair* category allows to check whether a pair style supports a
Pair::single() function as needed by compute group/group and others
features or LAMMPS, *respa* allows to check whether the inner/middle/outer
mode of r-RESPA is supported. In the various style categories,
the checking is also done using suffix flags, if available and enabled.

Example 1: disable use of suffix for pppm when using GPU package (i.e. run it on the CPU concurrently to running the pair style on the GPU), but do use the suffix otherwise (e.g. with OPENMP).

.. code-block:: LAMMPS

   pair_style lj/cut/coul/long 14.0
   if $(is_active(package,gpu)) then "suffix off"
   kspace_style pppm

Example 2: use r-RESPA with inner/outer cutoff, if supported by pair style, otherwise fall back to using pair and reducing the outer time step

.. code-block:: LAMMPS

   timestep $(2.0*(1.0+2.0*is_active(pair,respa))
   if $(is_active(pair,respa)) then "run_style respa 4 3 2 2  improper 1 inner 2 5.5 7.0 outer 3 kspace 4" else "run_style respa 3 3 2  improper 1 pair 2 kspace 3"

The *is_defined()* function allows to query categories like *compute*,
*dump*, *fix*, *group*, *region*, and *variable* whether an entry
with the provided name or id is defined.

The *is_available(category,name)* function allows to query whether
a specific optional feature is available, i.e. compiled in.
This currently works for the following categories: *command*,
*compute*, *fix*, *pair_style* and *feature*\ . For all categories
except *command* and *feature* also appending active suffixes is
tried before reporting failure.

The *feature* category is used to check the availability of compiled in
features such as GZIP support, PNG support, JPEG support, FFMPEG support,
and C++ exceptions for error handling. Corresponding values for name are
*gzip*, *png*, *jpeg*, *ffmpeg* and *exceptions*\ .

This enables writing input scripts which only dump using a given format if
the compiled binary supports it.

.. code-block:: LAMMPS

   if "$(is_available(feature,png))" then "print 'PNG supported'" else "print 'PNG not supported'"

   if "$(is_available(feature,ffmpeg)" then "dump 3 all movie 25 movie.mp4 type type zoom 1.6 adiam 1.0"

----------

Atom Values and Vectors
-----------------------

Atom values take an integer argument I from 1 to N, where I is the
atom-ID, e.g. x[243], which means use the x coordinate of the atom
with ID = 243.  Or they can take a variable name, specified as v_name,
where name is the name of the variable, like x[v_myIndex].  The
variable can be of any style except *vector* or *atom* or *atomfile*
variables.  The variable is evaluated and the result is expected to be
numeric and is cast to an integer (i.e. 3.4 becomes 3), to use an
index, which must be a value from 1 to N.  Note that a "formula"
cannot be used as the argument between the brackets, e.g. x[243+10]
or x[v_myIndex+1] are not allowed.  To do this a single variable can
be defined that contains the needed formula.

Note that the 0 < atom-ID <= N, where N is the largest atom ID
in the system.  If an ID is specified for an atom that does not
currently exist, then the generated value is 0.0.

Atom vectors generate one value per atom, so that a reference like
"vx" means the x-component of each atom's velocity will be used when
evaluating the variable.

The meaning of the different atom values and vectors is mostly
self-explanatory.  *Mol* refers to the molecule ID of an atom, and is
only defined if an :doc:`atom_style <atom_style>` is being used that
defines molecule IDs.

Note that many other atom attributes can be used as inputs to a
variable by using the :doc:`compute property/atom <compute_property_atom>` command and then specifying
a quantity from that compute.

----------

Compute References
------------------

Compute references access quantities calculated by a
:doc:`compute <compute>`.  The ID in the reference should be replaced by
the ID of a compute defined elsewhere in the input script.  As
discussed in the page for the :doc:`compute <compute>` command,
computes can produce global, per-atom, or local values.  Only global
and per-atom values can be used in a variable.  Computes can also
produce a scalar, vector, or array.

An equal-style variable can only use scalar values, which means a
global scalar, or an element of a global or per-atom vector or array.
A vector-style variable can use scalar values or a global vector of
values, or a column of a global array of values.  Atom-style variables
can use global scalar values.  They can also use per-atom vector
values, or a column of a per-atom array.  See the doc pages for
individual computes to see what kind of values they produce.

Examples of different kinds of compute references are as follows.
There is typically no ambiguity (see exception below) as to what a
reference means, since computes only produce either global or per-atom
quantities, never both.

+-------------+-------------------------------------------------------------------------------------------------------+
| c_ID       | global scalar, or per-atom vector                                                                      |
+-------------+-------------------------------------------------------------------------------------------------------+
| c_ID[I]    | Ith element of global vector, or atom I's value in per-atom vector, or Ith column from per-atom array  |
+-------------+-------------------------------------------------------------------------------------------------------+
| c_ID[I][J] | I,J element of global array, or atom I's Jth value in per-atom array                                   |
+-------------+-------------------------------------------------------------------------------------------------------+

For I and J indices, integers can be specified or a variable name,
specified as v_name, where name is the name of the variable.  The
rules for this syntax are the same as for the "Atom Values and
Vectors" discussion above.

One source of ambiguity for compute references is when a vector-style
variable refers to a compute that produces both a global scalar and a
global vector.  Consider a compute with ID "foo" that does this,
referenced as follows by variable "a", where "myVec" is another
vector-style variable:

.. code-block:: LAMMPS

   variable a vector c_foo*v_myVec

The reference "c_foo" could refer to either the global scalar or
global vector produced by compute "foo".  In this case, "c_foo" will
always refer to the global scalar, and "C_foo" can be used to
reference the global vector.  Similarly if the compute produces both a
global vector and global array, then "c_foo[I]" will always refer to
an element of the global vector, and "C_foo[I]" can be used to
reference the Ith column of the global array.

Note that if a variable containing a compute is evaluated directly in
an input script (not during a run), then the values accessed by the
compute must be current.  See the discussion below about "Variable
Accuracy".

----------

Fix References
--------------

Fix references access quantities calculated by a :doc:`fix <compute>`.
The ID in the reference should be replaced by the ID of a fix defined
elsewhere in the input script.  As discussed in the page for the
:doc:`fix <fix>` command, fixes can produce global, per-atom, or local
values.  Only global and per-atom values can be used in a variable.
Fixes can also produce a scalar, vector, or array.  An equal-style
variable can only use scalar values, which means a global scalar, or
an element of a global or per-atom vector or array.  Atom-style
variables can use the same scalar values.  They can also use per-atom
vector values.  A vector value can be a per-atom vector itself, or a
column of an per-atom array.  See the doc pages for individual fixes
to see what kind of values they produce.

The different kinds of fix references are exactly the same as the
compute references listed in the above table, where "c\_" is replaced
by "f\_".  Again, there is typically no ambiguity (see exception below)
as to what a reference means, since fixes only produce either global
or per-atom quantities, never both.

+-------------+-------------------------------------------------------------------------------------------------------+
| f_ID       | global scalar, or per-atom vector                                                                      |
+-------------+-------------------------------------------------------------------------------------------------------+
| f_ID[I]    | Ith element of global vector, or atom I's value in per-atom vector, or Ith column from per-atom array  |
+-------------+-------------------------------------------------------------------------------------------------------+
| f_ID[I][J] | I,J element of global array, or atom I's Jth value in per-atom array                                   |
+-------------+-------------------------------------------------------------------------------------------------------+

For I and J indices, integers can be specified or a variable name,
specified as v_name, where name is the name of the variable.  The
rules for this syntax are the same as for the "Atom Values and
Vectors" discussion above.

One source of ambiguity for fix references is the same ambiguity
discussed for compute references above.  Namely when a vector-style
variable refers to a fix that produces both a global scalar and a
global vector.  The solution is the same as for compute references.
For a fix with ID "foo", "f_foo" will always refer to the global
scalar, and "F_foo" can be used to reference the global vector.  And
similarly for distinguishing between a fix's global vector versus
global array with "f_foo[I]" versus "F_foo[I]".

Note that if a variable containing a fix is evaluated directly in an
input script (not during a run), then the values accessed by the fix
should be current.  See the discussion below about "Variable
Accuracy".

Note that some fixes only generate quantities on certain timesteps.
If a variable attempts to access the fix on non-allowed timesteps, an
error is generated.  For example, the :doc:`fix ave/time <fix_ave_time>`
command may only generate averaged quantities every 100 steps.  See
the doc pages for individual fix commands for details.

----------

Variable References
-------------------

Variable references access quantities stored or calculated by other
variables, which will cause those variables to be evaluated.  The name
in the reference should be replaced by the name of a variable defined
elsewhere in the input script.

As discussed on this doc page, equal-style variables generate a single
global numeric value, vector-style variables generate a vector of
global numeric values, and atom-style and atomfile-style variables
generate a per-atom vector of numeric values.  All other variables
store one or more strings.

The formula for an equal-style variable can use any style of variable
including a vector_style or atom-style or atomfile-style.  For these
3 styles, a subscript must be used to access a single value from
the vector-, atom-, or atomfile-style variable.  If a string-storing
variable is used, the string is converted to a numeric value.  Note
that this will typically produce a 0.0 if the string is not a numeric
string, which is likely not what you want.

The formula for a vector-style variable can use any style of variable,
including atom-style or atomfile-style variables.  For these 2 styles,
a subscript must be used to access a single value from the atom-, or
atomfile-style variable.

The formula for an atom-style variable can use any style of variable,
including other atom-style or atomfile-style variables.  If it uses a
vector-style variable, a subscript must be used to access a single
value from the vector-style variable.

Examples of different kinds of variable references are as follows.
There is no ambiguity as to what a reference means, since variables
produce only a global scalar or global vector or per-atom vector.

+------------+----------------------------------------------------------------------+
| v_name    | global scalar from equal-style variable                               |
+------------+----------------------------------------------------------------------+
| v_name    | global vector from vector-style variable                              |
+------------+----------------------------------------------------------------------+
| v_name    | per-atom vector from atom-style or atomfile-style variable            |
+------------+----------------------------------------------------------------------+
| v_name[I] | Ith element of a global vector from vector-style variable             |
+------------+----------------------------------------------------------------------+
| v_name[I] | value of atom with ID = I from atom-style or atomfile-style variable  |
+------------+----------------------------------------------------------------------+

For the I index, an integer can be specified or a variable name,
specified as v_name, where name is the name of the variable.  The
rules for this syntax are the same as for the "Atom Values and
Vectors" discussion above.

----------

Immediate Evaluation of Variables
"""""""""""""""""""""""""""""""""

If you want an equal-style variable to be evaluated immediately, it
may be the case that you do not need to define a variable at all.  See
the :doc:`Commands parse <Commands_parse>` page for info on how to
use "immediate" variables in an input script, specified as $(formula)
with parenthesis, where the formula has the same syntax as equal-style
variables described on this page.  This effectively evaluates a
formula immediately without using the variable command to define a
named variable.

More generally, there is a difference between referencing a variable
with a leading $ sign (e.g. $x or ${abc}) versus with a leading "v\_"
(e.g. v_x or v_abc).  The former can be used in any input script
command, including a variable command.  The input script parser
evaluates the reference variable immediately and substitutes its value
into the command.  As explained on the :doc:`Commands parse <Commands_parse>` doc page, you can also use un-named
"immediate" variables for this purpose.  For example, a string like
this $((xlo+xhi)/2+sqrt(v_area)) in an input script command evaluates
the string between the parenthesis as an equal-style variable formula.

Referencing a variable with a leading "v\_" is an optional or required
kind of argument for some commands (e.g. the :doc:`fix ave/chunk <fix_ave_chunk>` or :doc:`dump custom <dump>` or
:doc:`thermo_style <thermo_style>` commands) if you wish it to evaluate
a variable periodically during a run.  It can also be used in a
variable formula if you wish to reference a second variable.  The
second variable will be evaluated whenever the first variable is
evaluated.

As an example, suppose you use this command in your input script to
define the variable "v" as

.. code-block:: LAMMPS

   variable v equal vol

before a run where the simulation box size changes.  You might think
this will assign the initial volume to the variable "v".  That is not
the case.  Rather it assigns a formula which evaluates the volume
(using the thermo_style keyword "vol") to the variable "v".  If you
use the variable "v" in some other command like :doc:`fix ave/time <fix_ave_time>` then the current volume of the box will be
evaluated continuously during the run.

If you want to store the initial volume of the system, you can do it
this way:

.. code-block:: LAMMPS

   variable v equal vol
   variable v0 equal $v

The second command will force "v" to be evaluated (yielding the
initial volume) and assign that value to the variable "v0".  Thus the
command

.. code-block:: LAMMPS

   thermo_style custom step v_v v_v0

would print out both the current and initial volume periodically
during the run.

Note that it is a mistake to enclose a variable formula in double
quotes if it contains variables preceded by $ signs.  For example,

.. code-block:: LAMMPS

   variable vratio equal "${vfinal}/${v0}"

This is because the quotes prevent variable substitution (explained on
the :doc:`Commands parse <Commands_parse>` doc page), and thus an error
will occur when the formula for "vratio" is evaluated later.

----------

Variable Accuracy
"""""""""""""""""

Obviously, LAMMPS attempts to evaluate variables containing formulas
(\ *equal* and *atom* style variables) accurately whenever the
evaluation is performed.  Depending on what is included in the
formula, this may require invoking a :doc:`compute <compute>`, either
directly or indirectly via a thermo keyword, or accessing a value
previously calculated by a compute, or accessing a value calculated
and stored by a :doc:`fix <fix>`.  If the compute is one that calculates
the pressure or energy of the system, then these quantities need to be
tallied during the evaluation of the interatomic potentials (pair,
bond, etc) on timesteps that the variable will need the values.

LAMMPS keeps track of all of this during a :doc:`run <run>` or :doc:`energy minimization <minimize>`.  An error will be generated if you
attempt to evaluate a variable on timesteps when it cannot produce
accurate values.  For example, if a :doc:`thermo_style custom <thermo_style>` command prints a variable which accesses
values stored by a :doc:`fix ave/time <fix_ave_time>` command and the
timesteps on which thermo output is generated are not multiples of the
averaging frequency used in the fix command, then an error will occur.

An input script can also request variables be evaluated before or
after or in between runs, e.g. by including them in a
:doc:`print <print>` command.  In this case, if a compute is needed to
evaluate a variable (either directly or indirectly), LAMMPS will not
invoke the compute, but it will use a value previously calculated by
the compute, and can do this only if it was invoked on the current
timestep.  Fixes will always provide a quantity needed by a variable,
but the quantity may or may not be current.  This leads to one of
three kinds of behavior:

(1) The variable may be evaluated accurately.  If it contains
references to a compute or fix, and these values were calculated on
the last timestep of a preceding run, then they will be accessed and
used by the variable and the result will be accurate.

(2) LAMMPS may not be able to evaluate the variable and will generate
an error message stating so.  For example, if the variable requires a
quantity from a :doc:`compute <compute>` that has not been invoked on
the current timestep, LAMMPS will generate an error.  This means, for
example, that such a variable cannot be evaluated before the first run
has occurred.  Likewise, in between runs, a variable containing a
compute cannot be evaluated unless the compute was invoked on the last
timestep of the preceding run, e.g. by thermodynamic output.

One way to get around this problem is to perform a 0-timestep run
before using the variable.  For example, these commands

.. code-block:: LAMMPS

   variable t equal temp
   print "Initial temperature = $t"
   run 1000

will generate an error if the run is the first run specified in the
input script, because generating a value for the "t" variable requires
a compute for calculating the temperature to be invoked.

However, this sequence of commands would be fine:

.. code-block:: LAMMPS

   run 0
   variable t equal temp
   print "Initial temperature = $t"
   run 1000

The 0-timestep run initializes and invokes various computes, including
the one for temperature, so that the value it stores is current and
can be accessed by the variable "t" after the run has completed.  Note
that a 0-timestep run does not alter the state of the system, so it
does not change the input state for the 1000-timestep run that
follows.  Also note that the 0-timestep run must actually use and
invoke the compute in question (e.g. via :doc:`thermo <thermo_style>` or
:doc:`dump <dump>` output) in order for it to enable the compute to be
used in a variable after the run.  Thus if you are trying to print a
variable that uses a compute you have defined, you can insure it is
invoked on the last timestep of the preceding run by including it in
thermodynamic output.

Unlike computes, :doc:`fixes <fix>` will never generate an error if
their values are accessed by a variable in between runs.  They always
return some value to the variable.  However, the value may not be what
you expect if the fix has not yet calculated the quantity of interest
or it is not current.  For example, the :doc:`fix indent <fix_indent>`
command stores the force on the indenter.  But this is not computed
until a run is performed.  Thus if a variable attempts to print this
value before the first run, zeroes will be output.  Again, performing
a 0-timestep run before printing the variable has the desired effect.

(3) The variable may be evaluated incorrectly and LAMMPS may have no
way to detect this has occurred.  Consider the following sequence of
commands:

.. code-block:: LAMMPS

   pair_coeff 1 1 1.0 1.0
   run 1000
   pair_coeff 1 1 1.5 1.0
   variable e equal pe
   print "Final potential energy = $e"

The first run is performed using one setting for the pairwise
potential defined by the :doc:`pair_style <pair_style>` and
:doc:`pair_coeff <pair_coeff>` commands.  The potential energy is
evaluated on the final timestep and stored by the :doc:`compute pe <compute_pe>` compute (this is done by the
:doc:`thermo_style <thermo_style>` command).  Then a pair coefficient is
changed, altering the potential energy of the system.  When the
potential energy is printed via the "e" variable, LAMMPS will use the
potential energy value stored by the :doc:`compute pe <compute_pe>`
compute, thinking it is current.  There are many other commands which
could alter the state of the system between runs, causing a variable
to evaluate incorrectly.

The solution to this issue is the same as for case (2) above, namely
perform a 0-timestep run before the variable is evaluated to insure
the system is up-to-date.  For example, this sequence of commands
would print a potential energy that reflected the changed pairwise
coefficient:

.. code-block:: LAMMPS

   pair_coeff 1 1 1.0 1.0
   run 1000
   pair_coeff 1 1 1.5 1.0
   run 0
   variable e equal pe
   print "Final potential energy = $e"

----------

Restrictions
""""""""""""

Indexing any formula element by global atom ID, such as an atom value,
requires the :doc:`atom style <atom_style>` to use a global mapping in
order to look up the vector indices.  By default, only atom styles
with molecular information create global maps.  The :doc:`atom_modify map <atom_modify>` command can override the default, e.g. for
atomic-style atom styles.

All *universe*\ - and *uloop*\ -style variables defined in an input script
must have the same number of values.

Related commands
""""""""""""""""

:doc:`next <next>`, :doc:`jump <jump>`, :doc:`include <include>`,
:doc:`temper <temper>`, :doc:`fix print <fix_print>`, :doc:`print <print>`

Default
"""""""

none
