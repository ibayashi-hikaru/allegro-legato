.. index:: fix ave/correlate

fix ave/correlate command
=========================

Syntax
""""""

.. parsed-literal::

   fix ID group-ID ave/correlate Nevery Nrepeat Nfreq value1 value2 ... keyword args ...

* ID, group-ID are documented in :doc:`fix <fix>` command
* ave/correlate = style name of this fix command
* Nevery = use input values every this many timesteps
* Nrepeat = # of correlation time windows to accumulate
* Nfreq = calculate time window averages every this many timesteps
* one or more input values can be listed
* value = c_ID, c_ID[N], f_ID, f_ID[N], v_name

  .. parsed-literal::

       c_ID = global scalar calculated by a compute with ID
       c_ID[I] = Ith component of global vector calculated by a compute with ID, I can include wildcard (see below)
       f_ID = global scalar calculated by a fix with ID
       f_ID[I] = Ith component of global vector calculated by a fix with ID, I can include wildcard (see below)
       v_name = global value calculated by an equal-style variable with name
       v_name[I] = Ith component of a vector-style variable with name

* zero or more keyword/arg pairs may be appended
* keyword = *type* or *ave* or *start* or *prefactor* or *file* or *overwrite* or *title1* or *title2* or *title3*

  .. parsed-literal::

       *type* arg = *auto* or *upper* or *lower* or *auto/upper* or *auto/lower* or *full*
         auto = correlate each value with itself
         upper = correlate each value with each succeeding value
         lower = correlate each value with each preceding value
         auto/upper = auto + upper
         auto/lower = auto + lower
         full = correlate each value with every other value, including itself = auto + upper + lower
       *ave* args = *one* or *running*
         one = zero the correlation accumulation every Nfreq steps
         running = accumulate correlations continuously
       *start* args = Nstart
         Nstart = start accumulating correlations on this timestep
       *prefactor* args = value
         value = prefactor to scale all the correlation data by
       *file* arg = filename
         filename = name of file to output correlation data to
       *overwrite* arg = none = overwrite output file with only latest output
       *title1* arg = string
         string = text to print as 1st line of output file
       *title2* arg = string
         string = text to print as 2nd line of output file
       *title3* arg = string
         string = text to print as 3rd line of output file

Examples
""""""""

.. code-block:: LAMMPS

   fix 1 all ave/correlate 5 100 1000 c_myTemp file temp.correlate
   fix 1 all ave/correlate 1 50 10000 &
             c_thermo_press[1] c_thermo_press[2] c_thermo_press[3] &
             type upper ave running title1 "My correlation data"

   fix 1 all ave/correlate 1 50 10000 c_thermo_press[*]

Description
"""""""""""

Use one or more global scalar values as inputs every few timesteps,
calculate time correlations between them at varying time intervals,
and average the correlation data over longer timescales.  The
resulting correlation values can be time integrated by
:doc:`variables <variable>` or used by other :doc:`output commands <Howto_output>` such as :doc:`thermo_style custom <thermo_style>`, and can also be written to a file.  See the
:doc:`fix ave/correlate/long <fix_ave_correlate_long>` command for an
alternate method for computing correlation functions efficiently over
very long time windows.

The group specified with this command is ignored.  However, note that
specified values may represent calculations performed by computes and
fixes which store their own "group" definitions.

Each listed value can be the result of a :doc:`compute <compute>` or
:doc:`fix <fix>` or the evaluation of an equal-style or vector-style
:doc:`variable <variable>`.  In each case, the compute, fix, or variable
must produce a global quantity, not a per-atom or local quantity.  If
you wish to spatial- or time-average or histogram per-atom quantities
from a compute, fix, or variable, then see the :doc:`fix ave/chunk <fix_ave_chunk>`, :doc:`fix ave/atom <fix_ave_atom>`, or
:doc:`fix ave/histo <fix_ave_histo>` commands.  If you wish to convert a
per-atom quantity into a single global value, see the :doc:`compute reduce <compute_reduce>` command.

The input values must be all scalars.  What kinds of
correlations between input values are calculated is determined by the
*type* keyword as discussed below.

:doc:`Computes <compute>` that produce global quantities are those which
do not have the word *atom* in their style name.  Only a few
:doc:`fixes <fix>` produce global quantities.  See the doc pages for
individual fixes for info on which ones produce such values.
:doc:`Variables <variable>` of style *equal* and *vector* are the only
ones that can be used with this fix.  Variables of style *atom* cannot
be used, since they produce per-atom values.

Note that for values from a compute or fix, the bracketed index I can
be specified using a wildcard asterisk with the index to effectively
specify multiple values.  This takes the form "\*" or "\*n" or "n\*" or
"m\*n".  If N = the size of the vector (for *mode* = scalar) or the
number of columns in the array (for *mode* = vector), then an asterisk
with no numeric values means all indices from 1 to N.  A leading
asterisk means all indices from 1 to n (inclusive).  A trailing
asterisk means all indices from n to N (inclusive).  A middle asterisk
means all indices from m to n (inclusive).

Using a wildcard is the same as if the individual elements of the
vector had been listed one by one.  E.g. these 2 fix ave/correlate
commands are equivalent, since the :doc:`compute pressure <compute_pressure>` command creates a global vector with 6
values.

.. code-block:: LAMMPS

   compute myPress all pressure NULL
   fix 1 all ave/correlate 1 50 10000 c_myPress[*]
   fix 1 all ave/correlate 1 50 10000 &
             c_myPress[1] c_myPress[2] c_myPress[3] &
             c_myPress[4] c_myPress[5] c_myPress[6]

----------

The *Nevery*, *Nrepeat*, and *Nfreq* arguments specify on what
timesteps the input values will be used to calculate correlation data.
The input values are sampled every *Nevery* timesteps.  The
correlation data for the preceding samples is computed on timesteps
that are a multiple of *Nfreq*\ .  Consider a set of samples from some
initial time up to an output timestep.  The initial time could be the
beginning of the simulation or the last output time; see the *ave*
keyword for options.  For the set of samples, the correlation value
Cij is calculated as:

.. parsed-literal::

   Cij(delta) = ave(Vi(t)\*Vj(t+delta))

which is the correlation value between input values Vi and Vj,
separated by time delta.  Note that the second value Vj in the pair is
always the one sampled at the later time.  The ave() represents an
average over every pair of samples in the set that are separated by
time delta.  The maximum delta used is of size (\ *Nrepeat*\ -1)\*\ *Nevery*\ .
Thus the correlation between a pair of input values yields *Nrepeat*
correlation datums:

.. parsed-literal::

   Cij(0), Cij(Nevery), Cij(2\*Nevery), ..., Cij((Nrepeat-1)\*Nevery)

For example, if Nevery=5, Nrepeat=6, and Nfreq=100, then values on
timesteps 0,5,10,15,...,100 will be used to compute the final averages
on timestep 100.  Six averages will be computed: Cij(0), Cij(5),
Cij(10), Cij(15), Cij(20), and Cij(25).  Cij(10) on timestep 100 will
be the average of 19 samples, namely Vi(0)\*Vj(10), Vi(5)\*Vj(15),
Vi(10)\*V j20), Vi(15)\*Vj(25), ..., Vi(85)\*Vj(95), Vi(90)\*Vj(100).

*Nfreq* must be a multiple of *Nevery*\ ; *Nevery* and *Nrepeat* must be
non-zero.  Also, if the *ave* keyword is set to *one* which is the
default, then *Nfreq* >= (\ *Nrepeat*\ -1)\*\ *Nevery* is required.

----------

If a value begins with "c\_", a compute ID must follow which has been
previously defined in the input script.  If no bracketed term is
appended, the global scalar calculated by the compute is used.  If a
bracketed term is appended, the Ith element of the global vector
calculated by the compute is used.  See the discussion above for how I
can be specified with a wildcard asterisk to effectively specify
multiple values.

Note that there is a :doc:`compute reduce <compute_reduce>` command
which can sum per-atom quantities into a global scalar or vector which
can thus be accessed by fix ave/correlate.  Or it can be a compute
defined not in your input script, but by :doc:`thermodynamic output <thermo_style>` or other fixes such as :doc:`fix nvt <fix_nh>`
or :doc:`fix temp/rescale <fix_temp_rescale>`.  See the doc pages for
these commands which give the IDs of these computes.  Users can also
write code for their own compute styles and :doc:`add them to LAMMPS <Modify>`.

If a value begins with "f\_", a fix ID must follow which has been
previously defined in the input script.  If no bracketed term is
appended, the global scalar calculated by the fix is used.  If a
bracketed term is appended, the Ith element of the global vector
calculated by the fix is used.  See the discussion above for how I can
be specified with a wildcard asterisk to effectively specify multiple
values.

Note that some fixes only produce their values on certain timesteps,
which must be compatible with *Nevery*, else an error will result.
Users can also write code for their own fix styles and :doc:`add them to LAMMPS <Modify>`.

If a value begins with "v\_", a variable name must follow which has
been previously defined in the input script.  Only equal-style or
vector-style variables can be referenced; the latter requires a
bracketed term to specify the Ith element of the vector calculated by
the variable.  See the :doc:`variable <variable>` command for details.
Note that variables of style *equal* or *vector* define a formula
which can reference individual atom properties or thermodynamic
keywords, or they can invoke other computes, fixes, or variables when
they are evaluated, so this is a very general means of specifying
quantities to time correlate.

----------

Additional optional keywords also affect the operation of this fix.

The *type* keyword determines which pairs of input values are
correlated with each other.  For N input values Vi, for i = 1 to N,
let the number of pairs = Npair.  Note that the second value in the
pair Vi(t)\*Vj(t+delta) is always the one sampled at the later time.

* If *type* is set to *auto* then each input value is correlated with
  itself.  I.e. Cii = Vi\*Vi, for i = 1 to N, so Npair = N.
* If *type* is set
  to *upper* then each input value is correlated with every succeeding
  value.  I.e. Cij = Vi\*Vj, for i < j, so Npair = N\*(N-1)/2.
* If *type* is set
  to *lower* then each input value is correlated with every preceding
  value.  I.e. Cij = Vi\*Vj, for i > j, so Npair = N\*(N-1)/2.
* If *type* is set to *auto/upper* then each input value is correlated
  with itself and every succeeding value.  I.e. Cij = Vi\*Vj, for i >= j,
  so Npair = N\*(N+1)/2.
* If *type* is set to *auto/lower* then each input value is correlated
  with itself and every preceding value.  I.e. Cij = Vi\*Vj, for i <= j,
  so Npair = N\*(N+1)/2.
* If *type* is set to *full* then each input value is correlated with
  itself and every other value.  I.e. Cij = Vi\*Vj, for i,j = 1,N so
  Npair = N\^2.

The *ave* keyword determines what happens to the accumulation of
correlation samples every *Nfreq* timesteps.  If the *ave* setting is
*one*, then the accumulation is restarted or zeroed every *Nfreq*
timesteps.  Thus the outputs on successive *Nfreq* timesteps are
essentially independent of each other.  The exception is that the
Cij(0) = Vi(T)\*Vj(T) value at a timestep T, where T is a multiple of
*Nfreq*, contributes to the correlation output both at time T and at
time T+Nfreq.

If the *ave* setting is *running*, then the accumulation is never
zeroed.  Thus the output of correlation data at any timestep is the
average over samples accumulated every *Nevery* steps since the fix
was defined.  it can only be restarted by deleting the fix via the
:doc:`unfix <unfix>` command, or by re-defining the fix by re-specifying
it.

The *start* keyword specifies what timestep the accumulation of
correlation samples will begin on.  The default is step 0.  Setting it
to a larger value can avoid adding non-equilibrated data to the
correlation averages.

The *prefactor* keyword specifies a constant which will be used as a
multiplier on the correlation data after it is averaged.  It is
effectively a scale factor on Vi\*Vj, which can be used to account for
the size of the time window or other unit conversions.

The *file* keyword allows a filename to be specified.  Every *Nfreq*
steps, an array of correlation data is written to the file.  The
number of rows is *Nrepeat*, as described above.  The number of
columns is the Npair+2, also as described above.  Thus the file ends
up to be a series of these array sections.

The *overwrite* keyword will continuously overwrite the output file
with the latest output, so that it only contains one timestep worth of
output.  This option can only be used with the *ave running* setting.

The *title1* and *title2* and *title3* keywords allow specification of
the strings that will be printed as the first 3 lines of the output
file, assuming the *file* keyword was used.  LAMMPS uses default
values for each of these, so they do not need to be specified.

By default, these header lines are as follows:

.. parsed-literal::

   # Time-correlated data for fix ID
   # TimeStep Number-of-time-windows
   # Index TimeDelta Ncount valueI\*valueJ valueI\*valueJ ...

In the first line, ID is replaced with the fix-ID.  The second line
describes the two values that are printed at the first of each section
of output.  In the third line the value pairs are replaced with the
appropriate fields from the fix ave/correlate command.

----------

Let Sij = a set of time correlation data for input values I and J,
namely the *Nrepeat* values:

.. parsed-literal::

   Sij = Cij(0), Cij(Nevery), Cij(2\*Nevery), ..., Cij(\*Nrepeat-1)\*Nevery)

As explained below, these datums are output as one column of a global
array, which is effectively the correlation matrix.

The *trap* function defined for :doc:`equal-style variables <variable>`
can be used to perform a time integration of this vector of datums,
using a trapezoidal rule.  This is useful for calculating various
quantities which can be derived from time correlation data.  If a
normalization factor is needed for the time integration, it can be
included in the variable formula or via the *prefactor* keyword.

----------

Restart, fix_modify, output, run start/stop, minimize info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

No information about this fix is written to :doc:`binary restart files <restart>`.  None of the :doc:`fix_modify <fix_modify>` options
are relevant to this fix.

This fix computes a global array of values which can be accessed by
various :doc:`output commands <Howto_output>`.  The values can only be
accessed on timesteps that are multiples of *Nfreq* since that is when
averaging is performed.  The global array has # of rows = *Nrepeat*
and # of columns = Npair+2.  The first column has the time delta (in
timesteps) between the pairs of input values used to calculate the
correlation, as described above.  The second column has the number of
samples contributing to the correlation average, as described above.
The remaining Npair columns are for I,J pairs of the N input values,
as determined by the *type* keyword, as described above.

* For *type* = *auto*, the Npair = N columns are ordered: C11, C22, ...,
  CNN.
* For *type* = *upper*, the Npair = N\*(N-1)/2 columns are ordered: C12,
  C13, ..., C1N, C23, ..., C2N, C34, ..., CN-1N.
* For *type* = *lower*, the Npair = N\*(N-1)/2 columns are ordered: C21,
  C31, C32, C41, C42, C43, ..., CN1, CN2, ..., CNN-1.
* For *type* = *auto/upper*, the Npair = N\*(N+1)/2 columns are ordered:
  C11, C12, C13, ..., C1N, C22, C23, ..., C2N, C33, C34, ..., CN-1N,
  CNN.
* For *type* = *auto/lower*, the Npair = N\*(N+1)/2 columns are ordered:
  C11, C21, C22, C31, C32, C33, C41, ..., C44, CN1, CN2, ..., CNN-1,
  CNN.
* For *type* = *full*, the Npair = N\^2 columns are ordered: C11, C12,
  ..., C1N, C21, C22, ..., C2N, C31, ..., C3N, ..., CN1, ..., CNN-1,
  CNN.

The array values calculated by this fix are treated as intensive.  If
you need to divide them by the number of atoms, you must do this in a
later processing step, e.g. when using them in a
:doc:`variable <variable>`.

No parameter of this fix can be used with the *start/stop* keywords of
the :doc:`run <run>` command.  This fix is not invoked during :doc:`energy minimization <minimize>`.

Restrictions
""""""""""""
 none

Related commands
""""""""""""""""

:doc:`fix ave/correlate/long <fix_ave_correlate_long>`,
:doc:`compute <compute>`, :doc:`fix ave/time <fix_ave_time>`, :doc:`fix ave/atom <fix_ave_atom>`, :doc:`fix ave/chunk <fix_ave_chunk>`,
:doc:`fix ave/histo <fix_ave_histo>`, :doc:`variable <variable>`

Default
"""""""

none

The option defaults are ave = one, type = auto, start = 0, no file
output, title 1,2,3 = strings as described above, and prefactor = 1.0.
