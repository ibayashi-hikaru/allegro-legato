Install LAMMPS
==============

You can download LAMMPS as an executable or as source code.

With source code, you also have to :doc:`build LAMMPS <Build>`.  But you
have more flexibility as to what features to include or exclude in the
build.  If you plan to :doc:`modify or extend LAMMPS <Modify>`, then you
need the source code.

.. toctree::
   :maxdepth: 1

   Install_linux
   Install_mac
   Install_windows
   Install_conda

   Install_tarball
   Install_git

These are the files and sub-directories in the LAMMPS distribution:

+------------+-------------------------------------------+
| README     | Short description of the LAMMPS package   |
+------------+-------------------------------------------+
| LICENSE    | GNU General Public License (GPL)          |
+------------+-------------------------------------------+
| SECURITY.md| Security Policy for the LAMMPS package    |
+------------+-------------------------------------------+
| bench      | benchmark problems                        |
+------------+-------------------------------------------+
| cmake      | CMake build files                         |
+------------+-------------------------------------------+
| doc        | documentation                             |
+------------+-------------------------------------------+
| examples   | simple test problems                      |
+------------+-------------------------------------------+
| fortran    | Fortran wrapper for LAMMPS                |
+------------+-------------------------------------------+
| lib        | additional provided or external libraries |
+------------+-------------------------------------------+
| potentials | interatomic potential files               |
+------------+-------------------------------------------+
| python     | Python wrappers for LAMMPS                |
+------------+-------------------------------------------+
| src        | source files                              |
+------------+-------------------------------------------+
| tools      | pre- and post-processing tools            |
+------------+-------------------------------------------+
| unittest   | sources and inputs for testing LAMMPS     |
+------------+-------------------------------------------+

You will have all of these if you download source.  You will only have
some of them if you download executables, as explained on the pages
listed above.
