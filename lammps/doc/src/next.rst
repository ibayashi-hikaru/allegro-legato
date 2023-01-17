.. index:: next

next command
============

Syntax
""""""

.. parsed-literal::

   next variables

* variables = one or more variable names

Examples
""""""""

.. code-block:: LAMMPS

   next x
   next a t x myTemp

Description
"""""""""""

This command is used with variables defined by the
:doc:`variable <variable>` command.  It assigns the next value to the
variable from the list of values defined for that variable by the
:doc:`variable <variable>` command.  Thus when that variable is
subsequently substituted for in an input script command, the new value
is used.

See the :doc:`variable <variable>` command for info on how to define and
use different kinds of variables in LAMMPS input scripts.  If a
variable name is a single lower-case character from "a" to "z", it can
be used in an input script command as $a or $z.  If it is multiple
letters, it can be used as ${myTemp}.

If multiple variables are used as arguments to the *next* command,
then all must be of the same variable style: *index*, *loop*, *file*,
*universe*, or *uloop*\ .  An exception is that *universe*\ - and
*uloop*\ -style variables can be mixed in the same *next* command.

All the variables specified with the next command are incremented by
one value from their respective list of values.  A *file*\ -style
variable reads the next line from its associated file.  An
*atomfile*\ -style variable reads the next set of lines (one per atom)
from its associated file.  *String-* or *atom*\ - or *equal*\ - or
*world*\ -style variables cannot be used with the next command,
since they only store a single value.

When any of the variables in the next command has no more values, a
flag is set that causes the input script to skip the next
:doc:`jump <jump>` command encountered.  This enables a loop containing
a next command to exit.  As explained in the :doc:`variable <variable>`
command, the variable that has exhausted its values is also deleted.
This allows it to be used and re-defined later in the input script.
*File*\ -style and *atomfile*\ -style variables are exhausted when the
end-of-file is reached.

When the next command is used with *index*\ - or *loop*\ -style variables,
the next value is assigned to the variable for all processors.  When
the next command is used with *file*\ -style variables, the next line is
read from its file and the string assigned to the variable.  When the
next command is used with *atomfile*\ -style variables, the next set of
per-atom values is read from its file and assigned to the variable.

When the next command is used with *universe*\ - or *uloop*\ -style
variables, all *universe*\ - or *uloop*\ -style variables must be listed
in the next command.  This is because of the manner in which the
incrementing is done, using a single lock file for all variables.  The
next value (for each variable) is assigned to whichever processor
partition executes the command first.  All processors in the partition
are assigned the same value(s).  Running LAMMPS on multiple partitions
of processors via the :doc:`-partition command-line switch <Run_options>`.  *Universe*\ - and *uloop*\ -style variables are
incremented using the files "tmp.lammps.variable" and
"tmp.lammps.variable.lock" which you will see in your directory during
and after such a LAMMPS run.

Here is an example of running a series of simulations using the next
command with an *index*\ -style variable.  If this input script is named
in.polymer, 8 simulations would be run using data files from
directories run1 through run8.

.. code-block:: LAMMPS

   variable d index run1 run2 run3 run4 run5 run6 run7 run8
   shell cd $d
   read_data data.polymer
   run 10000
   shell cd ..
   clear
   next d
   jump in.polymer

If the variable "d" were of style *universe*, and the same in.polymer
input script were run on 3 partitions of processors, then the first 3
simulations would begin, one on each set of processors.  Whichever
partition finished first, it would assign variable "d" the fourth value
and run another simulation, and so forth until all 8 simulations were
finished.

Jump and next commands can also be nested to enable multi-level loops.
For example, this script will run 15 simulations in a double loop.

.. code-block:: LAMMPS

   variable i loop 3
     variable j loop 5
     clear
     ...
     read_data data.polymer.$i$j
     print Running simulation $i.$j
     run 10000
     next j
     jump in.script
   next i
   jump in.script

Here is an example of a double loop which uses the :doc:`if <if>` and
:doc:`jump <jump>` commands to break out of the inner loop when a
condition is met, then continues iterating through the outer loop.

.. code-block:: LAMMPS

   label       loopa
   variable    a loop 5
     label     loopb
     variable  b loop 5
     print     "A,B = $a,$b"
     run       10000
     if        $b > 2 then "jump in.script break"
     next      b
     jump      in.script loopb
   label       break
   variable    b delete

   next        a
   jump        in.script loopa

Restrictions
""""""""""""

As described above.

Related commands
""""""""""""""""

:doc:`jump <jump>`, :doc:`include <include>`, :doc:`shell <shell>`,
:doc:`variable <variable>`,

Default
"""""""

none
