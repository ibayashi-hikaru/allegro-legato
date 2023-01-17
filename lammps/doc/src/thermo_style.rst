.. index:: thermo_style

thermo_style command
====================

Syntax
""""""

.. code-block:: LAMMPS

   thermo_style style args

* style = *one* or *multi* or *custom*
* args = list of arguments for a particular style

  .. parsed-literal::

       *one* args = none
       *multi* args = none
       *custom* args = list of keywords
         possible keywords = step, elapsed, elaplong, dt, time,
                             cpu, tpcpu, spcpu, cpuremain, part, timeremain,
                             atoms, temp, press, pe, ke, etotal,
                             evdwl, ecoul, epair, ebond, eangle, edihed, eimp,
                             emol, elong, etail,
                             enthalpy, ecouple, econserve,
                             vol, density, lx, ly, lz, xlo, xhi, ylo, yhi, zlo, zhi,
                             xy, xz, yz, xlat, ylat, zlat,
                             bonds, angles, dihedrals, impropers,
                             pxx, pyy, pzz, pxy, pxz, pyz,
                             fmax, fnorm, nbuild, ndanger,
                             cella, cellb, cellc, cellalpha, cellbeta, cellgamma,
                             c_ID, c_ID[I], c_ID[I][J],
                             f_ID, f_ID[I], f_ID[I][J],
                             v_name, v_name[I]
           step = timestep
           elapsed = timesteps since start of this run
           elaplong = timesteps since start of initial run in a series of runs
           dt = timestep size
           time = simulation time
           cpu = elapsed CPU time in seconds since start of this run
           tpcpu = time per CPU second
           spcpu = timesteps per CPU second
           cpuremain = estimated CPU time remaining in run
           part = which partition (0 to Npartition-1) this is
           timeremain = remaining time in seconds on timer timeout.
           atoms = # of atoms
           temp = temperature
           press = pressure
           pe = total potential energy
           ke = kinetic energy
           etotal = total energy (pe + ke)
           evdwl = van der Waals pairwise energy (includes etail)
           ecoul = Coulombic pairwise energy
           epair = pairwise energy (evdwl + ecoul + elong)
           ebond = bond energy
           eangle = angle energy
           edihed = dihedral energy
           eimp = improper energy
           emol = molecular energy (ebond + eangle + edihed + eimp)
           elong = long-range kspace energy
           etail = van der Waals energy long-range tail correction
           enthalpy = enthalpy (etotal + press\*vol)
           ecouple = cumulative energy change due to thermo/baro statting fixes
           econserve = pe + ke + ecouple = etotal + ecouple
           vol = volume
           density = mass density of system
           lx,ly,lz = box lengths in x,y,z
           xlo,xhi,ylo,yhi,zlo,zhi = box boundaries
           xy,xz,yz = box tilt for triclinic (non-orthogonal) simulation boxes
           xlat,ylat,zlat = lattice spacings as calculated by :doc:`lattice <lattice>` command
           bonds,angles,dihedrals,impropers = # of these interactions defined
           pxx,pyy,pzz,pxy,pxz,pyz = 6 components of pressure tensor
           fmax = max component of force on any atom in any dimension
           fnorm = length of force vector for all atoms
           nbuild = # of neighbor list builds
           ndanger = # of dangerous neighbor list builds
           cella,cellb,cellc = periodic cell lattice constants a,b,c
           cellalpha, cellbeta, cellgamma = periodic cell angles alpha,beta,gamma
           c_ID = global scalar value calculated by a compute with ID
           c_ID[I] = Ith component of global vector calculated by a compute with ID, I can include wildcard (see below)
           c_ID[I][J] = I,J component of global array calculated by a compute with ID
           f_ID = global scalar value calculated by a fix with ID
           f_ID[I] = Ith component of global vector calculated by a fix with ID, I can include wildcard (see below)
           f_ID[I][J] = I,J component of global array calculated by a fix with ID
           v_name = value calculated by an equal-style variable with name
           v_name[I] = value calculated by a vector-style variable with name

Examples
""""""""

.. code-block:: LAMMPS

   thermo_style multi
   thermo_style custom step temp pe etotal press vol
   thermo_style custom step temp etotal c_myTemp v_abc
   thermo_style custom step temp etotal c_myTemp[*] v_abc

Description
"""""""""""

Set the style and content for printing thermodynamic data to the
screen and log file.

Style *one* prints a one-line summary of thermodynamic info that is
the equivalent of "thermo_style custom step temp epair emol etotal
press".  The line contains only numeric values.

Style *multi* prints a multiple-line listing of thermodynamic info
that is the equivalent of "thermo_style custom etotal ke temp pe ebond
eangle edihed eimp evdwl ecoul elong press".  The listing contains
numeric values and a string ID for each quantity.

Style *custom* is the most general setting and allows you to specify
which of the keywords listed above you want printed on each
thermodynamic timestep.  Note that the keywords c_ID, f_ID, v_name are
references to :doc:`computes <compute>`, :doc:`fixes <fix>`, and
equal-style :doc:`variables <variable>` that have been defined elsewhere
in the input script or can even be new styles which users have added
to LAMMPS.  See the :doc:`Modify <Modify>` page for details on the
latter.  Thus the *custom* style provides a flexible means of
outputting essentially any desired quantity as a simulation proceeds.

All styles except *custom* have *vol* appended to their list of
outputs if the simulation box volume changes during the simulation.

The values printed by the various keywords are instantaneous values,
calculated on the current timestep.  Time-averaged quantities, which
include values from previous timesteps, can be output by using the
f_ID keyword and accessing a fix that does time-averaging such as the
:doc:`fix ave/time <fix_ave_time>` command.

Options invoked by the :doc:`thermo_modify <thermo_modify>` command can
be used to set the one- or multi-line format of the print-out, the
normalization of thermodynamic output (total values versus per-atom
values for extensive quantities (ones which scale with the number of
atoms in the system), and the numeric precision of each printed value.

.. note::

   When you use a "thermo_style" command, all thermodynamic
   settings are restored to their default values, including those
   previously set by a :doc:`thermo_modify <thermo_modify>` command.  Thus
   if your input script specifies a thermo_style command, you should use
   the thermo_modify command after it.

----------

Several of the thermodynamic quantities require a temperature to be
computed: "temp", "press", "ke", "etotal", "enthalpy", "pxx", etc.  By
default this is done by using a *temperature* compute which is created
when LAMMPS starts up, as if this command had been issued:

.. code-block:: LAMMPS

   compute thermo_temp all temp

See the :doc:`compute temp <compute_temp>` command for details.  Note
that the ID of this compute is *thermo_temp* and the group is *all*\ .
You can change the attributes of this temperature (e.g. its
degrees-of-freedom) via the :doc:`compute_modify <compute_modify>`
command.  Alternatively, you can directly assign a new compute (that
calculates temperature) which you have defined, to be used for
calculating any thermodynamic quantity that requires a temperature.
This is done via the :doc:`thermo_modify <thermo_modify>` command.

Several of the thermodynamic quantities require a pressure to be
computed: "press", "enthalpy", "pxx", etc.  By default this is done by
using a *pressure* compute which is created when LAMMPS starts up, as
if this command had been issued:

.. code-block:: LAMMPS

   compute thermo_press all pressure thermo_temp

See the :doc:`compute pressure <compute_pressure>` command for details.
Note that the ID of this compute is *thermo_press* and the group is
*all*\ .  You can change the attributes of this pressure via the
:doc:`compute_modify <compute_modify>` command.  Alternatively, you can
directly assign a new compute (that calculates pressure) which you
have defined, to be used for calculating any thermodynamic quantity
that requires a pressure.  This is done via the
:doc:`thermo_modify <thermo_modify>` command.

Several of the thermodynamic quantities require a potential energy to
be computed: "pe", "etotal", "ebond", etc.  This is done by using a
*pe* compute which is created when LAMMPS starts up, as if this
command had been issued:

.. code-block:: LAMMPS

   compute thermo_pe all pe

See the :doc:`compute pe <compute_pe>` command for details.  Note that
the ID of this compute is *thermo_pe* and the group is *all*\ .  You can
change the attributes of this potential energy via the
:doc:`compute_modify <compute_modify>` command.

----------

The kinetic energy of the system *ke* is inferred from the temperature
of the system with :math:`\frac{1}{2} k_B T` of energy for each degree
of freedom.  Thus, using different :doc:`compute commands <compute>`
for calculating temperature, via the :doc:`thermo_modify temp
<thermo_modify>` command, may yield different kinetic energies, since
different computes that calculate temperature can subtract out
different non-thermal components of velocity and/or include different
degrees of freedom (translational, rotational, etc).

The potential energy of the system *pe* will include contributions
from fixes if the :doc:`fix_modify energy yes <fix_modify>` option is
set for a fix that calculates such a contribution.  For example, the
:doc:`fix wall/lj93 <fix_wall>` fix calculates the energy of atoms
interacting with the wall.  See the doc pages for "individual fixes"
to see which ones contribute and whether their default
:doc:`fix_modify energy <fix_modify>` setting is *yes* or *no*\ .

A long-range tail correction *etail* for the van der Waals pairwise
energy will be non-zero only if the :doc:`pair_modify tail
<pair_modify>` option is turned on.  The *etail* contribution is
included in *evdwl*, *epair*, *pe*, and *etotal*, and the
corresponding tail correction to the pressure is included in *press*
and *pxx*, *pyy*, etc.

----------

Here is more information on other keywords whose meaning may not be
clear:

The *step*, *elapsed*, and *elaplong* keywords refer to timestep
count.  *Step* is the current timestep, or iteration count when a
:doc:`minimization <minimize>` is being performed.  *Elapsed* is the
number of timesteps elapsed since the beginning of this run.
*Elaplong* is the number of timesteps elapsed since the beginning of
an initial run in a series of runs.  See the *start* and *stop*
keywords for the :doc:`run <run>` for info on how to invoke a series of
runs that keep track of an initial starting time.  If these keywords
are not used, then *elapsed* and *elaplong* are the same value.

The *dt* keyword is the current timestep size in time :doc:`units
<units>`.  The *time* keyword is the current elapsed simulation time,
also in time :doc:`units <units>`, which is simply (step\*dt) if the
timestep size has not changed and the timestep has not been reset.  If
the timestep has changed (e.g. via :doc:`fix dt/reset <fix_dt_reset>`)
or the timestep has been reset (e.g. via the "reset_timestep"
command), then the simulation time is effectively a cumulative value
up to the current point.

The *cpu* keyword is elapsed CPU seconds since the beginning of this
run.  The *tpcpu* and *spcpu* keywords are measures of how fast your
simulation is currently running.  The *tpcpu* keyword is simulation
time per CPU second, where simulation time is in time
:doc:`units <units>`.  E.g. for metal units, the *tpcpu* value would be
picoseconds per CPU second.  The *spcpu* keyword is the number of
timesteps per CPU second.  Both quantities are on-the-fly metrics,
measured relative to the last time they were invoked.  Thus if you are
printing out thermodynamic output every 100 timesteps, the two keywords
will continually output the time and timestep rate for the last 100
steps.  The *tpcpu* keyword does not attempt to track any changes in
timestep size, e.g. due to using the :doc:`fix dt/reset <fix_dt_reset>`
command.

The *cpuremain* keyword estimates the CPU time remaining in the
current run, based on the time elapsed thus far.  It will only be a
good estimate if the CPU time/timestep for the rest of the run is
similar to the preceding timesteps.  On the initial timestep the value
will be 0.0 since there is no history to estimate from.  For a
minimization run performed by the "minimize" command, the estimate is
based on the *maxiter* parameter, assuming the minimization will
proceed for the maximum number of allowed iterations.

The *part* keyword is useful for multi-replica or multi-partition
simulations to indicate which partition this output and this file
corresponds to, or for use in a :doc:`variable <variable>` to append to
a filename for output specific to this partition.  See discussion of
the :doc:`-partition command-line switch <Run_options>` for details on
running in multi-partition mode.

The *timeremain* keyword is the seconds remaining when a timeout has
been configured via the :doc:`timer timeout <timer>` command.  If the
timeout timer is inactive, the value of this keyword is 0.0 and if the
timer is expired, it is negative. This allows for example to exit
loops cleanly, if the timeout is expired with:

.. code-block:: LAMMPS

   if "$(timeremain) < 0.0" then "quit 0"

The *ecouple* keyword is cumulative energy change in the system due to
any thermostatting or barostatting fixes that are being used.  A
positive value means that energy has been subtracted from the system
(added to the coupling reservoir).  See the *econserve* keyword for an
explanation of why this sign choice makes sense.

The *econserve* keyword is the sum of the potential and kinetic energy
of the system as well as the energy that has been transferred by
thermostatting or barostatting to their coupling reservoirs.  I.e. it
is *pe* + *ke* + *econserve*\ .  Ideally, for a simulation in the NVE,
NPH, or NPT ensembles, the *econserve* quantity should remain constant
over time.

The *fmax* and *fnorm* keywords are useful for monitoring the progress
of an :doc:`energy minimization <minimize>`.  The *fmax* keyword
calculates the maximum force in any dimension on any atom in the
system, or the infinity-norm of the force vector for the system.  The
*fnorm* keyword calculates the 2-norm or length of the force vector.

The *nbuild* and *ndanger* keywords are useful for monitoring neighbor
list builds during a run.  Note that both these values are also
printed with the end-of-run statistics.  The *nbuild* keyword is the
number of re-builds during the current run.  The *ndanger* keyword is
the number of re-builds that LAMMPS considered potentially
"dangerous".  If atom movement triggered neighbor list rebuilding (see
the :doc:`neigh_modify <neigh_modify>` command), then dangerous
reneighborings are those that were triggered on the first timestep
atom movement was checked for.  If this count is non-zero you may wish
to reduce the delay factor to insure no force interactions are missed
by atoms moving beyond the neighbor skin distance before a rebuild
takes place.

The keywords *cella*, *cellb*, *cellc*, *cellalpha*,
*cellbeta*, *cellgamma*, correspond to the usual crystallographic
quantities that define the periodic unit cell of a crystal.  See the
:doc:`Howto triclinic <Howto_triclinic>` page for a geometric
description of triclinic periodic cells, including a precise
definition of these quantities in terms of the internal LAMMPS cell
dimensions *lx*, *ly*, *lz*, *yz*, *xz*, *xy*\ .

----------

For output values from a compute or fix, the bracketed index I used to
index a vector, as in *c_ID[I]* or *f_ID[I]*, can be specified
using a wildcard asterisk with the index to effectively specify
multiple values.  This takes the form "\*" or "\*n" or "n\*" or "m\*n".
If N = the size of the vector (for *mode* = scalar) or the number of
columns in the array (for *mode* = vector), then an asterisk with no
numeric values means all indices from 1 to N.  A leading asterisk
means all indices from 1 to n (inclusive).  A trailing asterisk means
all indices from n to N (inclusive).  A middle asterisk means all
indices from m to n (inclusive).

Using a wildcard is the same as if the individual elements of the
vector had been listed one by one.  E.g. these 2 thermo_style commands
are equivalent, since the :doc:`compute temp <compute_temp>` command
creates a global vector with 6 values.

.. code-block:: LAMMPS

   compute myTemp all temp
   thermo_style custom step temp etotal c_myTemp[*]
   thermo_style custom step temp etotal &
                c_myTemp[1] c_myTemp[2] c_myTemp[3] &
                c_myTemp[4] c_myTemp[5] c_myTemp[6]

----------

The *c_ID* and *c_ID[I]* and *c_ID[I][J]* keywords allow global values
calculated by a compute to be output.  As discussed on the
:doc:`compute <compute>` doc page, computes can calculate global,
per-atom, or local values.  Only global values can be referenced by
this command.  However, per-atom compute values for an individual atom
can be referenced in a :doc:`variable <variable>` and the variable
referenced by thermo_style custom, as discussed below.  See the
discussion above for how the I in *c_ID[I]* can be specified with a
wildcard asterisk to effectively specify multiple values from a global
compute vector.

The ID in the keyword should be replaced by the actual ID of a compute
that has been defined elsewhere in the input script.  See the
:doc:`compute <compute>` command for details.  If the compute calculates
a global scalar, vector, or array, then the keyword formats with 0, 1,
or 2 brackets will reference a scalar value from the compute.

Note that some computes calculate "intensive" global quantities like
temperature; others calculate "extensive" global quantities like
kinetic energy that are summed over all atoms in the compute group.
Intensive quantities are printed directly without normalization by
thermo_style custom.  Extensive quantities may be normalized by the
total number of atoms in the simulation (NOT the number of atoms in
the compute group) when output, depending on the :doc:`thermo_modify
norm <thermo_modify>` option being used.

The *f_ID* and *f_ID[I]* and *f_ID[I][J]* keywords allow global values
calculated by a fix to be output.  As discussed on the :doc:`fix
<fix>` doc page, fixes can calculate global, per-atom, or local
values.  Only global values can be referenced by this command.
However, per-atom fix values can be referenced for an individual atom
in a :doc:`variable <variable>` and the variable referenced by
thermo_style custom, as discussed below.  See the discussion above for
how the I in *f_ID[I]* can be specified with a wildcard asterisk to
effectively specify multiple values from a global fix vector.

The ID in the keyword should be replaced by the actual ID of a fix
that has been defined elsewhere in the input script.  See the
:doc:`fix <fix>` command for details.  If the fix calculates a global
scalar, vector, or array, then the keyword formats with 0, 1, or 2
brackets will reference a scalar value from the fix.

Note that some fixes calculate "intensive" global quantities like
timestep size; others calculate "extensive" global quantities like
energy that are summed over all atoms in the fix group.  Intensive
quantities are printed directly without normalization by thermo_style
custom.  Extensive quantities may be normalized by the total number of
atoms in the simulation (NOT the number of atoms in the fix group)
when output, depending on the :doc:`thermo_modify norm
<thermo_modify>` option being used.

The *v_name* keyword allow the current value of a variable to be
output.  The name in the keyword should be replaced by the variable
name that has been defined elsewhere in the input script.  Only
equal-style and vector-style variables can be referenced; the latter
requires a bracketed term to specify the Ith element of the vector
calculated by the variable.  However, an atom-style variable can be
referenced for an individual atom by an equal-style variable and that
variable referenced.  See the :doc:`variable <variable>` command for
details.  Variables of style *equal* and *vector* and *atom* define a
formula which can reference per-atom properties or thermodynamic
keywords, or they can invoke other computes, fixes, or variables when
evaluated, so this is a very general means of creating thermodynamic
output.

Note that equal-style and vector-style variables are assumed to
produce "intensive" global quantities, which are thus printed as-is,
without normalization by thermo_style custom.  You can include a
division by "natoms" in the variable formula if this is not the case.

----------

Restrictions
""""""""""""

This command must come after the simulation box is defined by a
:doc:`read_data <read_data>`, :doc:`read_restart <read_restart>`, or
:doc:`create_box <create_box>` command.

Related commands
""""""""""""""""

:doc:`thermo <thermo>`, :doc:`thermo_modify <thermo_modify>`,
:doc:`fix_modify <fix_modify>`, :doc:`compute temp <compute_temp>`,
:doc:`compute pressure <compute_pressure>`

Default
"""""""

.. code-block:: LAMMPS

   thermo_style one
