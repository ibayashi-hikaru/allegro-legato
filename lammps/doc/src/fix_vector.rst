.. index:: fix vector

fix vector command
==================

Syntax
""""""

.. parsed-literal::

   fix ID group-ID vector Nevery value1 value2 ...

* ID, group-ID are documented in :doc:`fix <fix>` command
* vector = style name of this fix command
* Nevery = use input values every this many timesteps
* one or more input values can be listed
* value = c_ID, c_ID[N], f_ID, f_ID[N], v_name

  .. parsed-literal::

       c_ID = global scalar calculated by a compute with ID
       c_ID[I] = Ith component of global vector calculated by a compute with ID
       f_ID = global scalar calculated by a fix with ID
       f_ID[I] = Ith component of global vector calculated by a fix with ID
       v_name = value calculated by an equal-style variable with name
       v_name[I] = Ith component of vector-style variable with name

Examples
""""""""

.. code-block:: LAMMPS

   fix 1 all vector 100 c_myTemp
   fix 1 all vector 5 c_myTemp v_integral

Description
"""""""""""

Use one or more global values as inputs every few timesteps, and
simply store them.  For a single specified value, the values are
stored as a global vector of growing length.  For multiple specified
values, they are stored as rows in a global array, whose number of
rows is growing.  The resulting vector or array can be used by other
:doc:`output commands <Howto_output>`.

One way to to use this command is to accumulate a vector that is
time-integrated using the :doc:`variable trap() <variable>` function.
For example the velocity auto-correlation function (VACF) can be
time-integrated, to yield a diffusion coefficient, as follows:

.. code-block:: LAMMPS

   compute         2 all vacf
   fix             5 all vector 1 c_2[4]
   variable        diff equal dt*trap(f_5)
   thermo_style    custom step v_diff

The group specified with this command is ignored.  However, note that
specified values may represent calculations performed by computes and
fixes which store their own "group" definitions.

Each listed value can be the result of a :doc:`compute <compute>` or
:doc:`fix <fix>` or the evaluation of an equal-style or vector-style
:doc:`variable <variable>`.  In each case, the compute, fix, or variable
must produce a global quantity, not a per-atom or local quantity.  And
the global quantity must be a scalar, not a vector or array.

:doc:`Computes <compute>` that produce global quantities are those which
do not have the word *atom* in their style name.  Only a few
:doc:`fixes <fix>` produce global quantities.  See the doc pages for
individual fixes for info on which ones produce such values.
:doc:`Variables <variable>` of style *equal* or *vector* are the only
ones that can be used with this fix.  Variables of style *atom* cannot
be used, since they produce per-atom values.

The *Nevery* argument specifies on what timesteps the input values
will be used in order to be stored.  Only timesteps that are a
multiple of *Nevery*, including timestep 0, will contribute values.

Note that if you perform multiple runs, using the "pre no" option of
the :doc:`run <run>` command to avoid initialization on subsequent runs,
then you need to use the *stop* keyword with the first :doc:`run <run>`
command with a timestep value that encompasses all the runs.  This is
so that the vector or array stored by this fix can be allocated to a
sufficient size.

----------

If a value begins with "c\_", a compute ID must follow which has been
previously defined in the input script.  If no bracketed term is
appended, the global scalar calculated by the compute is used.  If a
bracketed term is appended, the Ith element of the global vector
calculated by the compute is used.

Note that there is a :doc:`compute reduce <compute_reduce>` command
which can sum per-atom quantities into a global scalar or vector which
can thus be accessed by fix vector.  Or it can be a compute defined
not in your input script, but by :doc:`thermodynamic output <thermo_style>` or other fixes such as :doc:`fix nvt <fix_nh>`
or :doc:`fix temp/rescale <fix_temp_rescale>`.  See the doc pages for
these commands which give the IDs of these computes.  Users can also
write code for their own compute styles and :doc:`add them to LAMMPS <Modify>`.

If a value begins with "f\_", a fix ID must follow which has been
previously defined in the input script.  If no bracketed term is
appended, the global scalar calculated by the fix is used.  If a
bracketed term is appended, the Ith element of the global vector
calculated by the fix is used.

Note that some fixes only produce their values on certain timesteps,
which must be compatible with *Nevery*, else an error will result.
Users can also write code for their own fix styles and :doc:`add them to LAMMPS <Modify>`.

If a value begins with "v\_", a variable name must follow which has
been previously defined in the input script.  An equal-style or
vector-style variable can be referenced; the latter requires a
bracketed term to specify the Ith element of the vector calculated by
the variable.  See the :doc:`variable <variable>` command for details.
Note that variables of style *equal* and *vector* define a formula
which can reference individual atom properties or thermodynamic
keywords, or they can invoke other computes, fixes, or variables when
they are evaluated, so this is a very general means of specifying
quantities to be stored by fix vector.

----------

Restart, fix_modify, output, run start/stop, minimize info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

No information about this fix is written to :doc:`binary restart files <restart>`.  None of the :doc:`fix_modify <fix_modify>` options
are relevant to this fix.

This fix produces a global vector or global array which can be
accessed by various :doc:`output commands <Howto_output>`.  The values
can only be accessed on timesteps that are multiples of *Nevery*\ .

A vector is produced if only a single input value is specified.
An array is produced if multiple input values are specified.
The length of the vector or the number of rows in the array grows
by 1 every *Nevery* timesteps.

If the fix produces a vector, then the entire vector will be either
"intensive" or "extensive", depending on whether the values stored in
the vector are "intensive" or "extensive".  If the fix produces an
array, then all elements in the array must be the same, either
"intensive" or "extensive".  If a compute or fix provides the value
stored, then the compute or fix determines whether the value is
intensive or extensive; see the page for that compute or fix for
further info.  Values produced by a variable are treated as intensive.

This fix can allocate storage for stored values accumulated over
multiple runs, using the *start* and *stop* keywords of the
:doc:`run <run>` command.  See the :doc:`run <run>` command for details of
how to do this.  If using the :doc:`run pre no <run>` command option,
this is required to allow the fix to allocate sufficient storage for
stored values.

This fix is not invoked during :doc:`energy minimization <minimize>`.

Restrictions
""""""""""""
 none

Related commands
""""""""""""""""

:doc:`compute <compute>`, :doc:`variable <variable>`

Default
"""""""

none
