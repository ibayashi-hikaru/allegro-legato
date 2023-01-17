Body styles
===========

Classes that define body particles are derived from the Body class.
Body particles can represent complex entities, such as surface meshes
of discrete points, collections of sub-particles, deformable objects,
etc.

See the :doc:`Howto body <Howto_body>` page for an overview of using
body particles and the various body styles LAMMPS supports.  New
styles can be created to add new kinds of body particles to LAMMPS.

Body_nparticle.cpp is an example of a body particle that is treated as
a rigid body containing N sub-particles.

Here is a brief description of methods you define in your new derived
class.  See body.h for details.

+----------------------+-----------------------------------------------------------+
| data_body            | process a line from the Bodies section of a data file     |
+----------------------+-----------------------------------------------------------+
| noutrow              | number of sub-particles output is generated for           |
+----------------------+-----------------------------------------------------------+
| noutcol              | number of values per-sub-particle output is generated for |
+----------------------+-----------------------------------------------------------+
| output               | output values for the Mth sub-particle                    |
+----------------------+-----------------------------------------------------------+
| pack_comm_body       | body attributes to communicate every timestep             |
+----------------------+-----------------------------------------------------------+
| unpack_comm_body     | unpacking of those attributes                             |
+----------------------+-----------------------------------------------------------+
| pack_border_body     | body attributes to communicate when reneighboring is done |
+----------------------+-----------------------------------------------------------+
| unpack_border_body   | unpacking of those attributes                             |
+----------------------+-----------------------------------------------------------+
