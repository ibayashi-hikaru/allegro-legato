Auxiliary tools
***************

LAMMPS is designed to be a computational kernel for performing
molecular dynamics computations.  Additional pre- and post-processing
steps are often necessary to setup and analyze a simulation.  A list
of such tools can be found on the `LAMMPS webpage <lws_>`_ at these links:

* `Pre/Post processing <https://www.lammps.org/prepost.html>`_
* `External LAMMPS packages & tools <https://www.lammps.org/external.html>`_
* `Pizza.py toolkit <pizza_>`_

The last link for `Pizza.py <pizza_>`_ is a Python-based tool developed at
Sandia which provides tools for doing setup, analysis, plotting, and
visualization for LAMMPS simulations.

.. _lws: https://www.lammps.org
.. _pizza: https://lammps.github.io/pizza
.. _python: https://www.python.org

Additional tools included in the LAMMPS distribution are described on
this page.

Note that many users write their own setup or analysis tools or use
other existing codes and convert their output to a LAMMPS input format
or vice versa.  The tools listed here are included in the LAMMPS
distribution as examples of auxiliary tools.  Some of them are not
actively supported by the LAMMPS developers, as they were contributed
by LAMMPS users.  If you have problems using them, we can direct you
to the authors.

The source code for each of these codes is in the tools sub-directory
of the LAMMPS distribution.  There is a Makefile (which you may need
to edit for your platform) which will build several of the tools which
reside in that directory.  Most of them are larger packages in their
own sub-directories with their own Makefiles and/or README files.

----------

Pre-processing tools
====================

.. table_from_list::
   :columns: 6

   * :ref:`amber2lmp <amber>`
   * :ref:`ch2lmp <charmm>`
   * :ref:`chain <chain>`
   * :ref:`createatoms <createatoms>`
   * :ref:`drude <drude>`
   * :ref:`eam database <eamdb>`
   * :ref:`eam generate <eamgn>`
   * :ref:`eff <eff>`
   * :ref:`ipp <ipp>`
   * :ref:`micelle2d <micelle>`
   * :ref:`moltemplate <moltemplate>`
   * :ref:`msi2lmp <msi>`
   * :ref:`polybond <polybond>`


Post-processing tools
=====================

.. table_from_list::
   :columns: 6

   * :ref:`amber2lmp <amber>`
   * :ref:`binary2txt <binary>`
   * :ref:`ch2lmp <charmm>`
   * :ref:`colvars <colvars_tools>`
   * :ref:`eff <eff>`
   * :ref:`fep <fep>`
   * :ref:`lmp2arc <arc>`
   * :ref:`lmp2cfg <cfg>`
   * :ref:`matlab <matlab>`
   * :ref:`phonon <phonon>`
   * :ref:`pymol_asphere <pymol>`
   * :ref:`python <pythontools>`
   * :ref:`replica <replica>`
   * :ref:`smd <smd>`
   * :ref:`spin <spin>`
   * :ref:`xmgrace <xmgrace>`

Miscellaneous tools
===================

.. table_from_list::
   :columns: 6

   * :ref:`CMake <cmake>`
   * :ref:`emacs <emacs>`
   * :ref:`i-pi <ipi>`
   * :ref:`kate <kate>`
   * :ref:`LAMMPS shell <lammps_shell>`
   * :ref:`LAMMPS magic patterns for file(1) <magic>`
   * :ref:`Offline build tool <offline>`
   * :ref:`singularity <singularity_tool>`
   * :ref:`SWIG interface <swig>`
   * :ref:`vim <vim>`

----------

Tool descriptions
=================

.. _amber:

amber2lmp tool
--------------------------

The amber2lmp sub-directory contains two Python scripts for converting
files back-and-forth between the AMBER MD code and LAMMPS.  See the
README file in amber2lmp for more information.

These tools were written by Keir Novik while he was at Queen Mary
University of London.  Keir is no longer there and cannot support
these tools which are out-of-date with respect to the current LAMMPS
version (and maybe with respect to AMBER as well).  Since we don't use
these tools at Sandia, you will need to experiment with them and make
necessary modifications yourself.

----------

.. _binary:

binary2txt tool
----------------------------

The file binary2txt.cpp converts one or more binary LAMMPS dump file
into ASCII text files.  The syntax for running the tool is

.. code-block:: bash

   binary2txt file1 file2 ...

which creates file1.txt, file2.txt, etc.  This tool must be compiled
on a platform that can read the binary file created by a LAMMPS run,
since binary files are not compatible across all platforms.

----------

.. _charmm:

ch2lmp tool
------------------------

The ch2lmp sub-directory contains tools for converting files
back-and-forth between the CHARMM MD code and LAMMPS.

They are intended to make it easy to use CHARMM as a builder and as a
post-processor for LAMMPS. Using charmm2lammps.pl, you can convert a
PDB file with associated CHARMM info, including CHARMM force field
data, into its LAMMPS equivalent. Support for the CMAP correction of
CHARMM22 and later is available as an option. This tool can also add
solvent water molecules and Na+ or Cl- ions to the system.
Using lammps2pdb.pl you can convert LAMMPS atom dumps into PDB files.

See the README file in the ch2lmp sub-directory for more information.

These tools were created by Pieter in't Veld (pjintve at sandia.gov)
and Paul Crozier (pscrozi at sandia.gov) at Sandia.

CMAP support added and tested by Xiaohu Hu (hux2 at ornl.gov) and
Robert A. Latour (latourr at clemson.edu), David Hyde-Volpe, and
Tigran Abramyan, (Clemson University) and
Chris Lorenz (chris.lorenz at kcl.ac.uk), King's College London.

----------

.. _chain:

chain tool
----------------------

The file chain.f90 creates a LAMMPS data file containing bead-spring
polymer chains and/or monomer solvent atoms.  It uses a text file
containing chain definition parameters as an input.  The created
chains and solvent atoms can strongly overlap, so LAMMPS needs to run
the system initially with a "soft" pair potential to un-overlap it.
The syntax for running the tool is

.. code-block:: bash

   chain < def.chain > data.file

See the def.chain or def.chain.ab files in the tools directory for
examples of definition files.  This tool was used to create the system
for the :doc:`chain benchmark <Speed_bench>`.

----------

.. _cmake:

CMake tools
-----------

The ``cmbuild`` script is a wrapper around using ``cmake --build <dir>
--target`` and allows compiling LAMMPS in a :ref:`CMake build folder
<cmake_build>` with a make-like syntax regardless of the actual build
tool and the specific name of the program used (e.g. ``ninja-v1.10`` or
``gmake``) when using ``-D CMAKE_MAKE_PROGRAM=<name>``.

.. parsed-literal::

  Usage: cmbuild [-v] [-h] [-C <dir>] [-j <num>] [<target>]

  Options:
    -h                print this message
    -j <NUM>          allow processing of NUM concurrent tasks
    -C DIRECTORY      execute build in folder DIRECTORY
    -v                produce verbose output


----------

.. _colvars_tools:

colvars tools
---------------------------

The colvars directory contains a collection of tools for post-processing
data produced by the colvars collective variable library.
To compile the tools, edit the makefile for your system and run "make".

Please report problems and issues the colvars library and its tools
at: https://github.com/colvars/colvars/issues

abf_integrate:

MC-based integration of multidimensional free energy gradient
Version 20110511

.. parsed-literal::

   ./abf_integrate < filename > [-n < nsteps >] [-t < temp >] [-m [0\|1] (metadynamics)] [-h < hill_height >] [-f < variable_hill_factor >]

The LAMMPS interface to the colvars collective variable library, as
well as these tools, were created by Axel Kohlmeyer (akohlmey at
gmail.com) while at ICTP, Italy.

----------

.. _createatoms:

createatoms tool
----------------------------------

The tools/createatoms directory contains a Fortran program called
createAtoms.f which can generate a variety of interesting crystal
structures and geometries and output the resulting list of atom
coordinates in LAMMPS or other formats.

See the included Manual.pdf for details.

The tool is authored by Xiaowang Zhou (Sandia), xzhou at sandia.gov.

----------

.. _drude:

drude tool
----------------------

The tools/drude directory contains a Python script called
polarizer.py which can add Drude oscillators to a LAMMPS
data file in the required format.

See the header of the polarizer.py file for details.

The tool is authored by Agilio Padua and Alain Dequidt: agilio.padua
at ens-lyon.fr, alain.dequidt at uca.fr

----------

.. _eamdb:

eam database tool
-----------------------------

The tools/eam_database directory contains a Fortran program that will
generate EAM alloy setfl potential files for any combination of 16
elements: Cu, Ag, Au, Ni, Pd, Pt, Al, Pb, Fe, Mo, Ta, W, Mg, Co, Ti,
Zr.  The files can then be used with the :doc:`pair_style eam/alloy <pair_eam>` command.

The tool is authored by Xiaowang Zhou (Sandia), xzhou at sandia.gov,
and is based on his paper:

X. W. Zhou, R. A. Johnson, and H. N. G. Wadley, Phys. Rev. B, 69,
144113 (2004).

----------

.. _eamgn:

eam generate tool
-----------------------------

The tools/eam_generate directory contains several one-file C programs
that convert an analytic formula into a tabulated :doc:`embedded atom method (EAM) <pair_eam>` setfl potential file.  The potentials they
produce are in the potentials directory, and can be used with the
:doc:`pair_style eam/alloy <pair_eam>` command.

The source files and potentials were provided by Gerolf Ziegenhain
(gerolf at ziegenhain.com).

----------

.. _eff:

eff tool
------------------

The tools/eff directory contains various scripts for generating
structures and post-processing output for simulations using the
electron force field (eFF).

These tools were provided by Andres Jaramillo-Botero at CalTech
(ajaramil at wag.caltech.edu).

----------

.. _emacs:

emacs tool
----------------------

The tools/emacs directory contains an Emacs Lisp add-on file for GNU Emacs
that enables a lammps-mode for editing input scripts when using GNU Emacs,
with various highlighting options set up.

These tools were provided by Aidan Thompson at Sandia
(athomps at sandia.gov).

----------

.. _fep:

fep tool
------------------

The tools/fep directory contains Python scripts useful for
post-processing results from performing free-energy perturbation
simulations using the FEP package.

The scripts were contributed by Agilio Padua (ENS de Lyon), agilio.padua at ens-lyon.fr.

See README file in the tools/fep directory.

----------

.. _ipi:

i-pi tool
-------------------

The tools/i-pi directory contains a version of the i-PI package, with
all the LAMMPS-unrelated files removed.  It is provided so that it can
be used with the :doc:`fix ipi <fix_ipi>` command to perform
path-integral molecular dynamics (PIMD).

The i-PI package was created and is maintained by Michele Ceriotti,
michele.ceriotti at gmail.com, to interface to a variety of molecular
dynamics codes.

See the tools/i-pi/manual.pdf file for an overview of i-PI, and the
:doc:`fix ipi <fix_ipi>` page for further details on running PIMD
calculations with LAMMPS.

----------

.. _ipp:

ipp tool
------------------

The tools/ipp directory contains a Perl script ipp which can be used
to facilitate the creation of a complicated file (say, a lammps input
script or tools/createatoms input file) using a template file.

ipp was created and is maintained by Reese Jones (Sandia), rjones at
sandia.gov.

See two examples in the tools/ipp directory.  One of them is for the
tools/createatoms tool's input file.

----------

.. _kate:

kate tool
--------------------

The file in the tools/kate directory is an add-on to the Kate editor
in the KDE suite that allow syntax highlighting of LAMMPS input
scripts.  See the README.txt file for details.

The file was provided by Alessandro Luigi Sellerio
(alessandro.sellerio at ieni.cnr.it).

----------

.. _lammps_shell:

LAMMPS shell
------------

.. versionadded:: 9Oct2020

Overview
^^^^^^^^

The LAMMPS Shell, ``lammps-shell`` is a program that functions very
similar to the regular LAMMPS executable but has several modifications
and additions that make it more powerful for interactive sessions,
i.e. where you type LAMMPS commands from the prompt instead of reading
them from a file.

- It uses the readline and history libraries to provide command line
  editing and context aware TAB-expansion (details on that below).

- When processing an input file with the '-in' or '-i' flag from the
  command line, it does not exit at the end of that input file but
  stops at a prompt, so that additional commands can be issued

- Errors will not abort the shell but return to the prompt.

- It has additional commands aimed at interactive use (details below).

- Interrupting a calculation with CTRL-C will not terminate the
  session but rather enforce a timeout to cleanly stop an ongoing
  run (more info on timeouts is in the :doc:`timer command <timer>`
  documentation).

These enhancements make the LAMMPS shell an attractive choice for
interactive LAMMPS sessions in graphical desktop environments
(e.g. Gnome, KDE, Cinnamon, XFCE, Windows).

TAB-expansion
^^^^^^^^^^^^^

When writing commands interactively at the shell prompt, you can hit
the TAB key at any time to try and complete the text.  This completion
is context aware and will expand any first word only to commands
available in that executable.

- For style commands it will expand to available styles of the
  corresponding category (e.g. pair styles after a
  :doc:`pair_style <pair_style>` command).

- For :doc:`compute <compute>`, :doc:`fix <fix>`, or :doc:`dump <dump>`
  it will also expand only to already defined groups for the group-ID
  keyword.

- For commands like :doc:`compute_modify <compute_modify>`,
  :doc:`fix_modify <fix_modify>`, or :doc:`dump_modify <dump_modify>`
  it will expand to known compute/fix/dump IDs only.

- When typing references to computes, fixes, or variables with a
  "c\_", "f\_", or "v\_" prefix, respectively, then the expansion will
  be to known compute/fix IDs and variable names. Variable name
  expansion is also available for the ${name} variable syntax.

- In all other cases TAB expansion will complete to names of files
  and directories.

Command line editing and history
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When typing commands, command line editing similar to what BASH
provides is available.  Thus it is possible to move around the
currently line and perform various cut and insert and edit operations.
Previous commands can be retrieved by scrolling up (and down)
or searching (e.g. with CTRL-r).

Also history expansion through using the exclamation mark '!'
can be performed.  Examples: '!!' will be replaced with the previous
command, '!-2' will repeat the command before that, '!30' will be
replaced with event number 30 in the command history list, and
'!run' with the last command line that started with "run".  Adding
a ":p" to such a history expansion will result that the expansion is
printed and added to the history list, but NOT executed.
On exit the LAMMPS shell will write the history list to a file
".lammps_history" in the current working directory.  If such a
file exists when the LAMMPS shell is launched it will be read to
populate the history list.

This is realized via the readline library and can thus be customized
with an ``.inputrc`` file in the home directory.  For application
specific customization, the LAMMPS shell uses the name "lammps-shell".
For more information about using and customizing an application using
readline, please see the available documentation at:
`http://www.gnu.org/s/readline/#Documentation
<http://www.gnu.org/s/readline/#Documentation>`_

Additional commands
^^^^^^^^^^^^^^^^^^^

The following commands are added to the LAMMPS shell on top of the
regular LAMMPS commands:

.. parsed-literal::

   help (or ?)    print a brief help message
   history        display the current command history list
   clear_history  wipe out the current command history list
   save_history <range> <file>
                  write commands from the history to file.
                  The range is given as <from>-<to>, where <from> and <to>
                  may be empty. Example: save_history 100- in.recent
   source <file>  read commands from file (same as "include")
   pwd            print current working directory
   cd <directory> change current working directory (same as pwd if no directory)
   mem            print current and maximum memory usage
   \|<command>     execute <command> as a shell command and return to the command prompt
   exit           exit the LAMMPS shell cleanly (unlike the "quit" command)

Please note that some known shell operations are implemented in the
LAMMPS :doc:`shell command <shell>` in a platform neutral fashion,
while using the '\|' character will always pass the following text
to the operating system's shell command.

Compilation
^^^^^^^^^^^

Compilation of the LAMMPS shell can be enabled by setting the CMake
variable ``BUILD_LAMMPS_SHELL`` to "on" or using the makefile in the
``tools/lammps-shell`` folder to compile after building LAMMPS using
the conventional make procedure.  The makefile will likely need
customization depending on the features and settings used for
compiling LAMMPS.

Limitations
^^^^^^^^^^^

The LAMMPS shell was not designed for use with MPI parallelization
via ``mpirun`` or ``mpiexec`` or ``srun``.

Readline customization
^^^^^^^^^^^^^^^^^^^^^^

The behavior of the readline functionality can be customized in the
``${HOME}/.inputrc`` file.  This can be used to alter the default
settings or change the key-bindings.  The LAMMPS Shell sets the
application name ``lammps-shell``, so settings can be either applied
globally or only for the LAMMPS shell by bracketing them between
``$if lammps-shell`` and ``$endif`` like in the following example:

.. code-block:: bash

   $if lammps-shell
   # disable "beep" or "screen flash"
   set bell-style none
   # bind the "Insert" key to toggle overwrite mode
   "\e[2~": overwrite-mode
   $endif

More details about this are in the `readline documentation <https://tiswww.cwru.edu/php/chet/readline/rluserman.html#SEC9>`_.


LAMMPS Shell tips and tricks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Below are some suggestions for how to use and customize the LAMMPS shell.

Enable tilde expansion
""""""""""""""""""""""

Adding ``set expand-tilde on`` to ``${HOME}/.inputrc`` is recommended as
this will change the filename expansion behavior to replace any text
starting with "~" by the full path to the corresponding user's home
directory.  While the expansion of filenames **will** happen on all
arguments where the context is not known (e.g. ``~/compile/lamm<TAB>``
will expand to ``~/compile/lammps/``), it will not replace the tilde by
default.  But since LAMMPS does not do tilde expansion itself (unlike a
shell), this will result in errors.  Instead the tilde-expression should
be expanded into a valid path, where the plain "~/" stands for the
current user's home directory and "~someuser/" stands for
"/home/someuser" or whatever the full path to that user's home directory
is.

File extension association
""""""""""""""""""""""""""

Since the LAMMPS shell (unlike the regular LAMMPS executable) does not
exit when an input file is passed on the command line with the "-in" or
"-i" flag (the behavior is like for ``python -i <filename>``), it makes
the LAMMPS shell suitable for associating it with input files based on
their filename extension (e.g. ".lmp").  Since ``lammps-shell`` is a
console application, you have to run it inside a terminal program with a
command line like this:

.. code-block:: bash

   xterm -title "LAMMPS Shell" -e /path/to/lammps-shell -i in.file.lmp


Use history to create an input file
"""""""""""""""""""""""""""""""""""

When experimenting with commands to interactively to figure out a
suitable choice of settings or simply the correct syntax, you may want
to record part of your commands to a file for later use.  This can be
done with the ``save_history`` commands, which allows to selectively
write a section of the command history to a file (Example:
``save_history 25-30 in.run``).  This file can be further edited
(Example: ``|vim in.run``) and then the file read back in and tried out
(Example: ``source in.run``).  If the input also creates a system box,
you first need to use the :doc:`clear` command.

----------

.. _arc:

lmp2arc tool
------------

The lmp2arc sub-directory contains a tool for converting LAMMPS output
files to the format for Accelrys' Insight MD code (formerly
MSI/Biosym and its Discover MD code).  See the README file for more
information.

This tool was written by John Carpenter (Cray), Michael Peachey
(Cray), and Steve Lustig (Dupont).  John is now at the Mayo Clinic
(jec at mayo.edu), but still fields questions about the tool.

This tool was updated for the current LAMMPS C++ version by Jeff
Greathouse at Sandia (jagreat at sandia.gov).

----------

.. _cfg:

lmp2cfg tool
----------------------

The lmp2cfg sub-directory contains a tool for converting LAMMPS output
files into a series of \*.cfg files which can be read into the
`AtomEye <http://li.mit.edu/Archive/Graphics/A/>`_ visualizer.  See
the README file for more information.

This tool was written by Ara Kooser at Sandia (askoose at sandia.gov).

----------

.. _magic:

Magic patterns for the "file" command
-------------------------------------

.. versionadded:: 10Mar2021

The file ``magic`` contains patterns that are used by the
`file program <https://en.wikipedia.org/wiki/File_(command)>`_
available on most Unix-like operating systems which enables it
to detect various LAMMPS files and print some useful information
about them.  To enable these patterns, append or copy the contents
of the file to either the file ``.magic`` in your home directory
or (as administrator) to ``/etc/magic`` (for a system-wide
installation).  Afterwards the ``file`` command should be able to
detect most LAMMPS restarts, dump, data and log files. Examples:

.. code-block:: bash

   $ file *.*
   dihedral-quadratic.restart:   LAMMPS binary restart file (rev 2), Version 10 Mar 2021, Little Endian
   mol-pair-wf_cut.restart:      LAMMPS binary restart file (rev 2), Version 24 Dec 2020, Little Endian
   atom.bin:                     LAMMPS atom style binary dump (rev 2), Little Endian, First time step: 445570
   custom.bin:                   LAMMPS custom style binary dump (rev 2), Little Endian, First time step: 100
   bn1.lammpstrj:                LAMMPS text mode dump, First time step: 5000
   data.fourmol:                 LAMMPS data file written by LAMMPS
   pnc.data:                     LAMMPS data file written by msi2lmp
   data.spce:                    LAMMPS data file written by TopoTools
   B.data:                       LAMMPS data file written by OVITO
   log.lammps:                   LAMMPS log file written by version 10 Feb 2021

----------

.. _matlab:

matlab tool
------------------------

The matlab sub-directory contains several `MATLAB <matlabhome_>`_ scripts for
post-processing LAMMPS output.  The scripts include readers for log
and dump files, a reader for EAM potential files, and a converter that
reads LAMMPS dump files and produces CFG files that can be visualized
with the `AtomEye <http://li.mit.edu/Archive/Graphics/A/>`_
visualizer.

See the README.pdf file for more information.

These scripts were written by Arun Subramaniyan at Purdue Univ
(asubrama at purdue.edu).

.. _matlabhome: http://www.mathworks.com

----------

.. _micelle:

micelle2d tool
----------------------------

The file micelle2d.f creates a LAMMPS data file containing short lipid
chains in a monomer solution.  It uses a text file containing lipid
definition parameters as an input.  The created molecules and solvent
atoms can strongly overlap, so LAMMPS needs to run the system
initially with a "soft" pair potential to un-overlap it.  The syntax
for running the tool is

.. code-block:: bash

   micelle2d < def.micelle2d > data.file

See the def.micelle2d file in the tools directory for an example of a
definition file.  This tool was used to create the system for the
:doc:`micelle example <Examples>`.

----------

.. _moltemplate:

moltemplate tool
----------------------------------

The moltemplate sub-directory contains instructions for installing
moltemplate, a Python-based tool for building molecular systems based
on a text-file description, and creating LAMMPS data files that encode
their molecular topology as lists of bonds, angles, dihedrals, etc.
See the README.txt file for more information.

This tool was written by Andrew Jewett (jewett.aij at gmail.com), who
supports it.  It has its own WWW page at
`https://moltemplate.org <https://moltemplate.org>`_.
The latest sources can be found `on its GitHub page <https://github.com/jewettaij/moltemplate/releases>`_

----------

.. _msi:

msi2lmp tool
----------------------

The msi2lmp sub-directory contains a tool for creating LAMMPS template
input and data files from BIOVIA's Materias Studio files (formerly
Accelrys' Insight MD code, formerly MSI/Biosym and its Discover MD code).

This tool was written by John Carpenter (Cray), Michael Peachey
(Cray), and Steve Lustig (Dupont). Several people contributed changes
to remove bugs and adapt its output to changes in LAMMPS.

This tool has several known limitations and is no longer under active
development, so there are no changes except for the occasional bug fix.

See the README file in the tools/msi2lmp folder for more information.

----------

.. _offline:

Scripts for building LAMMPS when offline
----------------------------------------

In some situations it might be necessary to build LAMMPS on a system
without direct internet access. The scripts in ``tools/offline`` folder
allow you to pre-load external dependencies for both the documentation
build and for building LAMMPS with CMake.

It does so by

 #. downloading necessary ``pip`` packages,
 #. cloning ``git`` repositories
 #. downloading tarballs

to a designated cache folder.

As of April 2021, all of these downloads make up around 600MB. By
default, the offline scripts will download everything into the
``$HOME/.cache/lammps`` folder, but this can be changed by setting the
``LAMMPS_CACHING_DIR`` environment variable.

Once the caches have been initialized, they can be used for building the
LAMMPS documentation or compiling LAMMPS using CMake on an offline
system.

The ``use_caches.sh`` script must be sourced into the current shell
to initialize the offline build environment. Note that it must use
the same ``LAMMPS_CACHING_DIR``. This script does the following:

 #. Set up environment variables that modify the behavior of both,
    ``pip`` and ``git``
 #. Start a simple local HTTP server using Python to host files for CMake

Afterwards, it will print out instruction on how to modify the CMake
command line to make sure it uses the local HTTP server.

To undo the environment changes and shutdown the local HTTP server,
run the ``deactivate_caches`` command.

Examples
^^^^^^^^

For all of the examples below, you first need to create the cache, which
requires an internet connection.

.. code-block:: bash

   ./tools/offline/init_caches.sh

Afterwards, you can disconnect or copy the contents of the
``LAMMPS_CACHING_DIR`` folder to an offline system.

Documentation Build
^^^^^^^^^^^^^^^^^^^

The documentation build will create a new virtual environment that
typically first installs dependencies from ``pip``. With the offline
environment loaded, these installations will instead grab the necessary
packages from your local cache.

.. code-block:: bash

   # if LAMMPS_CACHING_DIR is different from default, make sure to set it first
   # export LAMMPS_CACHING_DIR=path/to/folder
   source tools/offline/use_caches.sh
   cd doc/
   make html

   deactivate_caches

CMake Build
^^^^^^^^^^^

When compiling certain packages with external dependencies, the CMake
build system will download necessary files or sources from the web. For
more flexibility the CMake configuration allows users to specify the URL
of each of these dependencies.  What the ``init_caches.sh`` script does
is create a CMake "preset" file, which sets the URLs for all of the known
dependencies and redirects the download to the local cache.

.. code-block:: bash

   # if LAMMPS_CACHING_DIR is different from default, make sure to set it first
   # export LAMMPS_CACHING_DIR=path/to/folder
   source tools/offline/use_caches.sh

   mkdir build
   cd build
   cmake -D LAMMPS_DOWNLOADS_URL=${HTTP_CACHE_URL} -C "${LAMMPS_HTTP_CACHE_CONFIG}" -C ../cmake/presets/most.cmake ../cmake
   make -j 8

   deactivate_caches

----------

.. _phonon:

phonon tool
------------------------

The phonon sub-directory contains a post-processing tool useful for
analyzing the output of the :doc:`fix phonon <fix_phonon>` command in
the PHONON package.

See the README file for instruction on building the tool and what
library it needs.  And see the examples/PACKAGES/phonon directory
for example problems that can be post-processed with this tool.

This tool was written by Ling-Ti Kong at Shanghai Jiao Tong
University.

----------

.. _polybond:

polybond tool
----------------------------

The polybond sub-directory contains a Python-based tool useful for
performing "programmable polymer bonding".  The Python file
lmpsdata.py provides a "Lmpsdata" class with various methods which can
be invoked by a user-written Python script to create data files with
complex bonding topologies.

See the Manual.pdf for details and example scripts.

This tool was written by Zachary Kraus at Georgia Tech.

----------

.. _pymol:

pymol_asphere tool
-------------------------------

The pymol_asphere sub-directory contains a tool for converting a
LAMMPS dump file that contains orientation info for ellipsoidal
particles into an input file for the `PyMol visualization package <pymolhome_>`_ or its `open source variant <pymolopen_>`_.

.. _pymolhome: https://www.pymol.org

.. _pymolopen: https://github.com/schrodinger/pymol-open-source

Specifically, the tool triangulates the ellipsoids so they can be
viewed as true ellipsoidal particles within PyMol.  See the README and
examples directory within pymol_asphere for more information.

This tool was written by Mike Brown at Sandia.

----------

.. _pythontools:

python tool
-----------------------------

The python sub-directory contains several Python scripts
that perform common LAMMPS post-processing tasks, such as:

* extract thermodynamic info from a log file as columns of numbers
* plot two columns of thermodynamic info from a log file using GnuPlot
* sort the snapshots in a dump file by atom ID
* convert multiple :doc:`NEB <neb>` dump files into one dump file for viz
* convert dump files into XYZ, CFG, or PDB format for viz by other packages

These are simple scripts built on `Pizza.py <pizza_>`_ modules.  See the
README for more info on Pizza.py and how to use these scripts.

----------

.. _replica:

replica tool
--------------------------

The tools/replica directory contains the reorder_remd_traj python script which
can be used to reorder the replica trajectories (resulting from the use of the
temper command) according to temperature. This will produce discontinuous
trajectories with all frames at the same temperature in each trajectory.
Additional options can be used to calculate the canonical configurational
log-weight for each frame at each temperature using the pymbar package. See
the README.md file for further details. Try out the peptide example provided.

This tool was written by (and is maintained by) Tanmoy Sanyal,
while at the Shell lab at UC Santa Barbara. (tanmoy dot 7989 at gmail.com)

----------

.. _smd:

smd tool
------------------

The smd sub-directory contains a C++ file dump2vtk_tris.cpp and
Makefile which can be compiled and used to convert triangle output
files created by the Smooth-Mach Dynamics (MACHDYN) package into a
VTK-compatible unstructured grid file.  It could then be read in and
visualized by VTK.

See the header of dump2vtk.cpp for more details.

This tool was written by the MACHDYN package author, Georg
Ganzenmuller at the Fraunhofer-Institute for High-Speed Dynamics,
Ernst Mach Institute in Germany (georg.ganzenmueller at emi.fhg.de).

----------

.. _spin:

spin tool
--------------------

The spin sub-directory contains a C file interpolate.c which can
be compiled and used to perform a cubic polynomial interpolation of
the MEP following a GNEB calculation.

See the README file in tools/spin/interpolate_gneb for more details.

This tool was written by the SPIN package author, Julien
Tranchida at Sandia National Labs (jtranch at sandia.gov, and by Aleksei
Ivanov, at University of Iceland (ali5 at hi.is).

----------

.. _singularity_tool:

singularity tool
----------------------------------------

The singularity sub-directory contains container definitions files
that can be used to build container images for building and testing
LAMMPS on specific OS variants using the `Singularity <https://sylabs.io>`_
container software. Contributions for additional variants are welcome.
For more details please see the README.md file in that folder.

----------

.. _swig:

SWIG interface
--------------

The `SWIG tool <http://swig.org>`_ offers a mostly automated way to
incorporate compiled code modules into scripting languages.  It
processes the function prototypes in C and generates wrappers for a wide
variety of scripting languages from it.  Thus it can also be applied to
the :doc:`C language library interface <Library>` of LAMMPS so that
build a wrapper that allows to call LAMMPS from programming languages
like: C#/Mono, Lua, Java, JavaScript, Perl, Python, R, Ruby, Tcl, and
more.

What is included
^^^^^^^^^^^^^^^^

We provide here an "interface file", ``lammps.i``, that has the content
of the ``library.h`` file adapted so SWIG can process it.  That will
create wrappers for all the functions that are present in the LAMMPS C
library interface.  Please note that not all kinds of C functions can be
automatically translated, so you would have to add custom functions to
be able to utilize those where the automatic translation does not work.
A few functions for converting pointers and accessing arrays are
predefined.  We provide the file here on an "as is" basis to help people
getting started, but not as a fully tested and supported feature of the
LAMMPS distribution.  Any contributions to complete this are, of course,
welcome.  Please also note, that for the case of creating a Python wrapper,
a fully supported :doc:`Ctypes based lammps module <Python_module>`
already exists.  That module is designed to be object oriented while
SWIG will generate a 1:1 translation of the functions in the interface file.

Building the wrapper
^^^^^^^^^^^^^^^^^^^^

When using CMake, the build steps for building a wrapper
module are integrated for the languages: Java, Lua,
Perl5, Python, Ruby, and Tcl.  These require that the
LAMMPS library is build as a shared library and all
necessary development headers and libraries are present.

.. code-block:: bash

   -D WITH_SWIG=on         # to enable building any SWIG wrapper
   -D BUILD_SWIG_JAVA=on   # to enable building the Java wrapper
   -D BUILD_SWIG_LUA=on    # to enable building the Lua wrapper
   -D BUILD_SWIG_PERL5=on  # to enable building the Perl 5.x wrapper
   -D BUILD_SWIG_PYTHON=on # to enable building the Python wrapper
   -D BUILD_SWIG_RUBY=on   # to enable building the Ruby wrapper
   -D BUILD_SWIG_TCL=on    # to enable building the Tcl wrapper


Manual building allows a little more flexibility. E.g. one can choose
the name of the module and build and use a dynamically loaded object
for Tcl with:

.. code-block:: bash

   $ swig -tcl -module tcllammps lammps.i
   $ gcc -fPIC -shared $(pkgconf --cflags tcl) -o tcllammps.so \
               lammps_wrap.c -L ../src/ -llammps
   $ tclsh

Or one can build an extended Tcl shell command with the wrapped
functions included with:

.. code-block:: bash

   $ swig -tcl -module tcllmps lammps_shell.i
   $ gcc -o tcllmpsh lammps_wrap.c -Xlinker -export-dynamic \
            -DHAVE_CONFIG_H $(pkgconf --cflags tcl) \
            $(pkgconf --libs tcl) -L ../src -llammps

In both cases it is assumed that the LAMMPS library was compiled
as a shared library in the ``src`` folder. Otherwise the last
part of the commands needs to be adjusted.

Utility functions
^^^^^^^^^^^^^^^^^

Definitions for several utility functions required to manage and access
data passed or returned as pointers are included in the ``lammps.i``
file.  So most of the functionality of the library interface should be
accessible.  What works and what does not depends a bit on the
individual language for which the wrappers are built and how well SWIG
supports those.  The `SWIG documentation <http://swig.org/doc.html>`_
has very detailed instructions and recommendations.

Usage examples
^^^^^^^^^^^^^^

The ``tools/swig`` folder has multiple shell scripts, ``run_<name>_example.sh``
that will create a small example script and demonstrate how to load
the wrapper and run LAMMPS through it in the corresponding programming
language.

For illustration purposes below is a part of the Tcl example script.

.. code-block:: tcl

   % load ./tcllammps.so
   % set lmp [lammps_open_no_mpi 0 NULL NULL]
   % lammps_command $lmp "units real"
   % lammps_command $lmp "lattice fcc 2.5"
   % lammps_command $lmp "region box block -5 5 -5 5 -5 5"
   % lammps_command $lmp "create_box 1 box"
   % lammps_command $lmp "create_atoms 1 box"
   %
   % set dt [doublep_value [voidp_to_doublep [lammps_extract_global $lmp dt]]]
   % puts "LAMMPS version $ver"
   % puts [format "Number of created atoms: %g" [lammps_get_natoms $lmp]]
   % puts "Current size of timestep: $dt"
   % puts "LAMMPS version: [lammps_version $lmp]"
   % lammps_close $lmp

----------

.. _vim:

vim tool
------------------

The files in the tools/vim directory are add-ons to the VIM editor
that allow easier editing of LAMMPS input scripts.  See the README.txt
file for details.

These files were provided by Gerolf Ziegenhain (gerolf at
ziegenhain.com)

----------

.. _xmgrace:

xmgrace tool
--------------------------

The files in the tools/xmgrace directory can be used to plot the
thermodynamic data in LAMMPS log files via the xmgrace plotting
package.  There are several tools in the directory that can be used in
post-processing mode.  The lammpsplot.cpp file can be compiled and
used to create plots from the current state of a running LAMMPS
simulation.

See the README file for details.

These files were provided by Vikas Varshney (vv0210 at gmail.com)
