.. index:: restart

restart command
===============

Syntax
""""""

.. parsed-literal::

   restart 0
   restart N root keyword value ...
   restart N file1 file2 keyword value ...

* N = write a restart file every this many timesteps
* N can be a variable (see below)
* root = filename to which timestep # is appended
* file1,file2 = two full filenames, toggle between them when writing file
* zero or more keyword/value pairs may be appended
* keyword = *fileper* or *nfile*

  .. parsed-literal::

       *fileper* arg = Np
         Np = write one file for every this many processors
       *nfile* arg = Nf
         Nf = write this many files, one from each of Nf processors

Examples
""""""""

.. code-block:: LAMMPS

   restart 0
   restart 1000 poly.restart
   restart 1000 poly.restart.mpiio
   restart 1000 restart.*.equil
   restart 10000 poly.%.1 poly.%.2 nfile 10
   restart v_mystep poly.restart

Description
"""""""""""

Write out a binary restart file with the current state of the
simulation every so many timesteps, in either or both of two modes, as
a run proceeds.  A value of 0 means do not write out any restart
files.  The two modes are as follows.  If one filename is specified, a
series of filenames will be created which include the timestep in the
filename.  If two filenames are specified, only 2 restart files will
be created, with those names.  LAMMPS will toggle between the 2 names
as it writes successive restart files.

Note that you can specify the restart command twice, once with a
single filename and once with two filenames.  This would allow you,
for example, to write out archival restart files every 100000 steps
using a single filename, and more frequent temporary restart files
every 1000 steps, using two filenames.  Using restart 0 will turn off
both modes of output.

Similar to :doc:`dump <dump>` files, the restart filename(s) can contain
two wild-card characters.

If a "\*" appears in the single filename, it is replaced with the
current timestep value.  This is only recognized when a single
filename is used (not when toggling back and forth).  Thus, the third
example above creates restart files as follows: restart.1000.equil,
restart.2000.equil, etc.  If a single filename is used with no "\*",
then the timestep value is appended.  E.g. the second example above
creates restart files as follows: poly.restart.1000,
poly.restart.2000, etc.

If a "%" character appears in the restart filename(s), then one file
is written for each processor and the "%" character is replaced with
the processor ID from 0 to P-1.  An additional file with the "%"
replaced by "base" is also written, which contains global information.
For example, the files written on step 1000 for filename restart.%
would be restart.base.1000, restart.0.1000, restart.1.1000, ...,
restart.P-1.1000.  This creates smaller files and can be a fast mode
of output and subsequent input on parallel machines that support
parallel I/O.  The optional *fileper* and *nfile* keywords discussed
below can alter the number of files written.

The restart file can also be written in parallel as one large binary
file via the MPI-IO library, which is part of the MPI standard for
versions 2.0 and above.  Using MPI-IO requires two steps.  First,
build LAMMPS with its MPIIO package installed, e.g.

.. code-block:: bash

   make yes-mpiio    # installs the MPIIO package
   make mpi          # build LAMMPS for your platform

Second, use a restart filename which contains ".mpiio".  Note that it
does not have to end in ".mpiio", just contain those characters.
Unlike MPI-IO dump files, a particular restart file must be both
written and read using MPI-IO.

Restart files are written on timesteps that are a multiple of N but
not on the first timestep of a run or minimization.  You can use the
:doc:`write_restart <write_restart>` command to write a restart file
before a run begins.  A restart file is not written on the last
timestep of a run unless it is a multiple of N.  A restart file is
written on the last timestep of a minimization if N > 0 and the
minimization converges.

Instead of a numeric value, N can be specified as an :doc:`equal-style variable <variable>`, which should be specified as v_name, where
name is the variable name.  In this case, the variable is evaluated at
the beginning of a run to determine the next timestep at which a
restart file will be written out.  On that timestep, the variable will
be evaluated again to determine the next timestep, etc.  Thus the
variable should return timestep values.  See the stagger() and
logfreq() and stride() math functions for :doc:`equal-style variables <variable>`, as examples of useful functions to use in
this context.  Other similar math functions could easily be added as
options for :doc:`equal-style variables <variable>`.

For example, the following commands will write restart files
every step from 1100 to 1200, and could be useful for debugging
a simulation where something goes wrong at step 1163:

.. code-block:: LAMMPS

   variable       s equal stride(1100,1200,1)
   restart        v_s tmp.restart

----------

See the :doc:`read_restart <read_restart>` command for information about
what is stored in a restart file.

Restart files can be read by a :doc:`read_restart <read_restart>`
command to restart a simulation from a particular state.  Because the
file is binary (to enable exact restarts), it may not be readable on
another machine.  In this case, you can use the :doc:`-r command-line switch <Run_options>` to convert a restart file to a data file.

.. note::

   Although the purpose of restart files is to enable restarting a
   simulation from where it left off, not all information about a
   simulation is stored in the file.  For example, the list of fixes that
   were specified during the initial run is not stored, which means the
   new input script must specify any fixes you want to use.  Even when
   restart information is stored in the file, as it is for some fixes,
   commands may need to be re-specified in the new input script, in order
   to re-use that information.  See the :doc:`read_restart <read_restart>`
   command for information about what is stored in a restart file.

----------

The optional *nfile* or *fileper* keywords can be used in conjunction
with the "%" wildcard character in the specified restart file name(s).
As explained above, the "%" character causes the restart file to be
written in pieces, one piece for each of P processors.  By default P =
the number of processors the simulation is running on.  The *nfile* or
*fileper* keyword can be used to set P to a smaller value, which can
be more efficient when running on a large number of processors.

The *nfile* keyword sets P to the specified Nf value.  For example, if
Nf = 4, and the simulation is running on 100 processors, 4 files will
be written, by processors 0,25,50,75.  Each will collect information
from itself and the next 24 processors and write it to a restart file.

For the *fileper* keyword, the specified value of Np means write one
file for every Np processors.  For example, if Np = 4, every fourth
processor (0,4,8,12,etc) will collect information from itself and the
next 3 processors and write it to a restart file.

----------

Restrictions
""""""""""""

To write and read restart files in parallel with MPI-IO, the MPIIO
package must be installed.

Related commands
""""""""""""""""

:doc:`write_restart <write_restart>`, :doc:`read_restart <read_restart>`

Default
"""""""

.. code-block:: LAMMPS

   restart 0
