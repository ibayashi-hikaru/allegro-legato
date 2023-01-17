Bond, angle, dihedral, improper styles
======================================

Classes that compute molecular interactions are derived from the Bond,
Angle, Dihedral, and Improper classes.  New styles can be created to
add new potentials to LAMMPS.

Bond_harmonic.cpp is the simplest example of a bond style.  Ditto for
the harmonic forms of the angle, dihedral, and improper style
commands.

Here is a brief description of common methods you define in your
new derived class.  See bond.h, angle.h, dihedral.h, and improper.h
for details and specific additional methods.

+-----------------------+---------------------------------------------------------------------------------------+
| init                  | check if all coefficients are set, calls *init_style* (optional)                      |
+-----------------------+---------------------------------------------------------------------------------------+
| init_style            | check if style specific conditions are met (optional)                                 |
+-----------------------+---------------------------------------------------------------------------------------+
| compute               | compute the molecular interactions (required)                                         |
+-----------------------+---------------------------------------------------------------------------------------+
| settings              | apply global settings for all types (optional)                                        |
+-----------------------+---------------------------------------------------------------------------------------+
| coeff                 | set coefficients for one type (required)                                              |
+-----------------------+---------------------------------------------------------------------------------------+
| equilibrium_distance  | length of bond, used by SHAKE (required, bond only)                                   |
+-----------------------+---------------------------------------------------------------------------------------+
| equilibrium_angle     | opening of angle, used by SHAKE (required, angle only)                                |
+-----------------------+---------------------------------------------------------------------------------------+
| write & read_restart  | writes/reads coeffs to restart files (required)                                       |
+-----------------------+---------------------------------------------------------------------------------------+
| single                | force (bond only) and energy of a single bond or angle (required, bond or angle only) |
+-----------------------+---------------------------------------------------------------------------------------+
| memory_usage          | tally memory allocated by the style (optional)                                        |
+-----------------------+---------------------------------------------------------------------------------------+
