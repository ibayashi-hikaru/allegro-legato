.. index:: fix atc

fix atc command
===============

Syntax
""""""

.. parsed-literal::

   fix <fixID> <group> atc <type> <parameter_file>

* fixID = name of fix
* group = name of group fix is to be applied
* type = *thermal* or *two_temperature* or *hardy* or *field*

.. parsed-literal::

    *thermal* = thermal coupling with fields: temperature
    *two_temperature* = electron-phonon coupling with field: temperature and electron_temperature
    *hardy* = on-the-fly post-processing using kernel localization functions
    *field* = on-the-fly post-processing using mesh-based localization functions

* parameter_file = name of the file with material parameters. Note: Neither hardy nor field requires a parameter file

Examples
""""""""

.. code-block:: LAMMPS

   fix AtC internal atc thermal Ar_thermal.dat
   fix AtC internal atc two_temperature Ar_ttm.mat
   fix AtC internal atc hardy
   fix AtC internal atc field

Description
"""""""""""

This fix is the beginning to creating a coupled FE/MD simulation and/or
an on-the-fly estimation of continuum fields. The coupled versions of
this fix do Verlet integration and the post-processing does not. After
instantiating this fix, several other fix_modify commands will be
needed to set up the problem, e.g. define the finite element mesh and
prescribe initial and boundary conditions.

.. image:: JPG/atc_nanotube.jpg
   :align: center

The following coupling example is typical, but non-exhaustive:

.. code-block:: LAMMPS

    # ... commands to create and initialize the MD system

    # initial fix to designate coupling type and group to apply it to
    # tag group physics material_file
    fix AtC internal atc thermal Ar_thermal.mat

    # create a uniform 12 x 2 x 2 mesh that covers region contain the group
    # nx ny nz region periodicity
    fix_modify AtC mesh create 12 2 2 mdRegion f p p

    # specify the control method for the type of coupling
    # physics control_type
    fix_modify AtC thermal control flux

    # specify the initial values for the empirical field "temperature"
    # field node_group value
    fix_modify AtC initial temperature all 30

    # create an output stream for nodal fields
    # filename output_frequency
    fix_modify AtC output atc_fe_output 100

    run 1000

likewise for this post-processing example:

.. code-block:: LAMMPS

    # ... commands to create and initialize the MD system

    # initial fix to designate post-processing and the group to apply it to
    # no material file is allowed nor required
    fix AtC internal atc hardy

    # for hardy fix, specific kernel function (function type and range) to # be used as a localization function
    fix AtC kernel quartic_sphere 10.0

    # create a uniform 1 x 1 x 1 mesh that covers region contain the group
    # with periodicity this effectively creates a system average
    fix_modify AtC mesh create 1 1 1 box p p p

    # change from default lagrangian map to eulerian
    # refreshed every 100 steps
    fix_modify AtC atom_element_map eulerian 100

    # start with no field defined
    # add mass density, potential energy density, stress and temperature
    fix_modify AtC fields add density energy stress temperature

    # create an output stream for nodal fields
    # filename output_frequency
    fix_modify AtC output nvtFE 100 text

    run 1000

the mesh's linear interpolation functions can be used as the localization function
by using the field option:

.. code-block:: LAMMPS

    fix AtC internal atc field
    fix_modify AtC mesh create 1 1 1 box p p p
    ...

Note coupling and post-processing can be combined in the same simulations using separate fixes.

----------

Restart, fix_modify, output, run start/stop, minimize info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

No information about this fix is written to :doc:`binary restart files
<restart>`.

The :doc:`fix_modify <fix_modify>` *energy* option is not supported by
this fix, but this fix does add the kinetic energy imparted to atoms
by the momentum coupling mode of the AtC package to the global
potential energy of the system as part of :doc:`thermodynamic output
<thermo_style>`.

Additional :doc:`fix_modify <fix_modify>` options relevant to this
fix are listed below.

This fix computes a global scalar which can be accessed by various
:doc:`output commands <Howto_output>`.  The scalar is the energy
discussed in the previous paragraph.  The scalar value is "extensive".

No parameter of this fix can be used with the
*start/stop* keywords of the :doc:`run <run>` command.  This fix is not
invoked during :doc:`energy minimization <minimize>`.

Restrictions
""""""""""""

Thermal and two_temperature (coupling) types use a Verlet
time-integration algorithm. The hardy type does not contain its own
time-integrator and must be used with a separate fix that does contain
one, e.g. nve, nvt, etc. In addition, currently:

* the coupling is restricted to thermal physics
* the FE computations are done in serial on each processor.

.. _atc_fix_modify:

Related commands
""""""""""""""""

After specifying this fix in your input script, several
:doc:`fix_modify AtC <fix_modify>` commands are used to setup the
problem, e.g. define the finite element mesh and prescribe initial and
boundary conditions.  Each of these options has its own doc page.

*fix_modify* commands for setup:

* :doc:`fix_modify AtC mesh create <atc_mesh_create>`
* :doc:`fix_modify AtC mesh quadrature <atc_mesh_quadrature>`
* :doc:`fix_modify AtC mesh read <atc_mesh_read>`
* :doc:`fix_modify AtC mesh write <atc_mesh_write>`
* :doc:`fix_modify AtC mesh create_nodeset <atc_mesh_create_nodeset>`
* :doc:`fix_modify AtC mesh add_to_nodeset <atc_mesh_add_to_nodeset>`
* :doc:`fix_modify AtC mesh create_faceset box <atc_mesh_create_faceset_box>`
* :doc:`fix_modify AtC mesh create_faceset plane <atc_mesh_create_faceset_plane>`
* :doc:`fix_modify AtC mesh create_elementset <atc_mesh_create_elementset>`
* :doc:`fix_modify AtC mesh delete_elements <atc_mesh_delete_elements>`
* :doc:`fix_modify AtC mesh nodeset_to_elementset <atc_mesh_nodeset_to_elementset>`
* :doc:`fix_modify AtC boundary type <atc_boundary_type>`
* :doc:`fix_modify AtC internal_quadrature <atc_internal_quadrature>`
* :doc:`fix_modify AtC time_integration <atc_time_integration>`
* :doc:`fix_modify AtC extrinsic electron_integration <atc_electron_integration>`
* :doc:`fix_modify AtC internal_element_set <atc_internal_element_set>`
* :doc:`fix_modify AtC decomposition <atc_decomposition>`

*fix_modify* commands for boundary and initial conditions:

* :doc:`fix_modify AtC initial <atc_initial>`
* :doc:`fix_modify AtC fix <atc_fix>`
* :doc:`fix_modify AtC unfix <atc_unfix>`
* :doc:`fix_modify AtC fix_flux <atc_fix_flux>`
* :doc:`fix_modify AtC unfix_flux <atc_unfix_flux>`
* :doc:`fix_modify AtC source <atc_source>`
* :doc:`fix_modify AtC remove_source <atc_remove_source>`

*fix_modify* commands for control and filtering:

* :doc:`fix_modify AtC control thermal <atc_control_thermal>`
* :doc:`fix_modify AtC control momentum <atc_control_momentum>`
* :doc:`fix_modify AtC control localized_lambda <atc_control_localized_lambda>`
* :doc:`fix_modify AtC control lumped_lambda_solve <atc_lumped_lambda_solve>`
* :doc:`fix_modify AtC control mask_direction <atc_mask_direction>`
* :doc:`fix_modify AtC filter <atc_time_filter>`
* :doc:`fix_modify AtC filter scale <atc_filter_scale>`
* :doc:`fix_modify AtC filter type <atc_filter_type>`
* :doc:`fix_modify AtC equilibrium_start <atc_equilibrium_start>`
* :doc:`fix_modify AtC extrinsic exchange <atc_extrinsic_exchange>`
* :doc:`fix_modify AtC poisson_solver <atc_poisson_solver>`

*fix_modify* commands for output:

* :doc:`fix_modify AtC output <atc_output>`
* :doc:`fix_modify AtC output nodeset <atc_output_nodeset>`
* :doc:`fix_modify AtC output volume_integral <atc_output_volume_integral>`
* :doc:`fix_modify AtC output boundary_integral <atc_output_boundary_integral>`
* :doc:`fix_modify AtC output contour_integral <atc_output_contour_integral>`
* :doc:`fix_modify AtC mesh output <atc_mesh_output>`
* :doc:`fix_modify AtC write_restart <atc_write_restart>`
* :doc:`fix_modify AtC read_restart <atc_read_restart>`

*fix_modify* commands for post-processing:

* :doc:`fix_modify AtC kernel <atc_hardy_kernel>`
* :doc:`fix_modify AtC fields <atc_hardy_fields>`
* :doc:`fix_modify AtC gradients <atc_hardy_gradients>`
* :doc:`fix_modify AtC rates <atc_hardy_rates>`
* :doc:`fix_modify AtC computes <atc_hardy_computes>`
* :doc:`fix_modify AtC on_the_fly <atc_hardy_on_the_fly>`
* :doc:`fix_modify AtC pair/bond_interactions <atc_pair_interactions>`
* :doc:`fix_modify AtC sample_frequency <atc_sample_frequency>`
* :doc:`fix_modify AtC set <atc_set_reference_pe>`

miscellaneous *fix_modify* commands:

* :doc:`fix_modify AtC atom_element_map <atc_atom_element_map>`
* :doc:`fix_modify AtC atom_weight <atc_atom_weight>`
* :doc:`fix_modify AtC write_atom_weights <atc_write_atom_weights>`
* :doc:`fix_modify AtC kernel_bandwidth <atc_kernel_bandwidth>`
* :doc:`fix_modify AtC reset_time <atc_reset_time>`
* :doc:`fix_modify AtC reset_atomic_reference_positions <atc_reset_atomic_reference>`
* :doc:`fix_modify AtC fe_md_boundary <atc_fe_md_boundary>`
* :doc:`fix_modify AtC boundary_faceset <atc_boundary_faceset>`
* :doc:`fix_modify AtC consistent_fe_initialization <atc_consistent_fe_initialization>`
* :doc:`fix_modify AtC mass_matrix <atc_mass_matrix>`
* :doc:`fix_modify AtC material <atc_material>`
* :doc:`fix_modify AtC atomic_charge <atc_atomic_charge>`
* :doc:`fix_modify AtC source_integration <atc_source_integration>`
* :doc:`fix_modify AtC temperature_definition <atc_temperature_definition>`
* :doc:`fix_modify AtC track_displacement <atc_track_displacement>`
* :doc:`fix_modify AtC boundary_dynamics <atc_boundary_dynamics>`
* :doc:`fix_modify AtC add_species <atc_add_species>`
* :doc:`fix_modify AtC add_molecule <atc_add_molecule>`
* :doc:`fix_modify AtC remove_species <atc_remove_species>`
* :doc:`fix_modify AtC remove_molecule <atc_remove_molecule>`

Note: a set of example input files with the attendant material files
are included in the ``examples/PACKAGES/atc`` folders.

Default
"""""""
None

----------

For detailed exposition of the theory and algorithms please see:

.. _Wagner:

**(Wagner)** Wagner, GJ; Jones, RE; Templeton, JA; Parks, MA, "An
 atomistic-to-continuum coupling method for heat transfer in solids."
 Special Issue of Computer Methods and Applied Mechanics (2008)
 197:3351.

.. _Zimmeman2004:

**(Zimmerman2004)** Zimmerman, JA; Webb, EB; Hoyt, JJ;. Jones, RE;
 Klein, PA; Bammann, DJ, "Calculation of stress in atomistic
 simulation." Special Issue of Modelling and Simulation in Materials
 Science and Engineering (2004), 12:S319.

.. _Zimmerman2010:

**(Zimmerman2010)** Zimmerman, JA; Jones, RE; Templeton, JA, "A
 material frame approach for evaluating continuum variables in
 atomistic simulations." Journal of Computational Physics (2010),
 229:2364.

.. _Templeton2010:

**(Templeton2010)** Templeton, JA; Jones, RE; Wagner, GJ, "Application
 of a field-based method to spatially varying thermal transport
 problems in molecular dynamics." Modelling and Simulation in
 Materials Science and Engineering (2010), 18:085007.

.. _Jones:

**(Jones)** Jones, RE; Templeton, JA; Wagner, GJ; Olmsted, D; Modine,
 JA, "Electron transport enhanced molecular dynamics for metals and
 semi-metals." International Journal for Numerical Methods in
 Engineering (2010), 83:940.

.. _Templeton2011:

**(Templeton2011)** Templeton, JA; Jones, RE; Lee, JW; Zimmerman, JA;
 Wong, BM, "A long-range electric field solver for molecular dynamics
 based on atomistic-to-continuum modeling." Journal of Chemical Theory
 and Computation (2011), 7:1736.

.. _Mandadapu:

**(Mandadapu)** Mandadapu, KK; Templeton, JA; Lee, JW, "Polarization
 as a field variable from molecular dynamics simulations." Journal of
 Chemical Physics (2013), 139:054115.

Please refer to the standard finite element (FE) texts, e.g. T.J.R
Hughes " The finite element method ", Dover 2003, for the basics of FE
simulation.
