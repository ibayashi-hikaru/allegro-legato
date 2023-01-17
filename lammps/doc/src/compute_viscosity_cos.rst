.. index:: compute viscosity/cos

compute viscosity/cos command
=============================

Syntax
""""""


.. parsed-literal::

   compute ID group-ID viscosity/cos

* ID, group-ID are documented in :doc:`compute <compute>` command
* viscosity/cos = style name of this compute command


Examples
""""""""


.. code-block:: LAMMPS

   compute  cos all viscosity/cos
   variable V equal c_cos[7]
   variable A equal 0.02E-5
   variable density equal density
   variable lz equal lz
   variable reciprocalViscosity equal v_V/${A}/v_density*39.4784/v_lz/v_lz*100

Description
"""""""""""

Define a computation that calculates the velocity amplitude of a group of atoms
with an cosine-shaped velocity profile and the temperature of them
after subtracting out the velocity profile before computing the kinetic energy.
A compute of this style can be used by any command that computes a temperature,
e.g. :doc:`thermo_modify <thermo_modify>`, :doc:`fix npt <fix_nh>`, etc.

This command together with :doc:`fix_accelerate/cos<fix_accelerate_cos>`
enables viscosity calculation with periodic perturbation method,
as described by :ref:`Hess<Hess1>`.
An acceleration along the x-direction is applied to the simulation system
by using :doc:`fix_accelerate/cos<fix_accelerate_cos>` command.
The acceleration is a periodic function along the z-direction:

.. math::

   a_{x}(z) = A \cos \left(\frac{2 \pi z}{l_{z}}\right)

where :math:`A` is the acceleration amplitude, :math:`l_z` is the z-length
of the simulation box. At steady state, the acceleration generates
a velocity profile:

.. math::

   v_{x}(z) = V \cos \left(\frac{2 \pi z}{l_{z}}\right)

The generated velocity amplitude :math:`V` is related to the
shear viscosity :math:`\eta` by:

.. math::

   V = \frac{A \rho}{\eta}\left(\frac{l_{z}}{2 \pi}\right)^{2}


and it can be obtained from ensemble average of the velocity profile:

.. math::

   V = \frac{\sum_i 2 m_{i} v_{i, x} \cos \left(\frac{2 \pi z_i}{l_{z}}\right)}{\sum_i m_{i}}


where :math:`m_i`, :math:`v_{i,x}` and :math:`z_i` are the mass,
x-component velocity and z coordinate of a particle.

After the cosine-shaped collective velocity in :math:`x` direction
has been subtracted for each atom, the temperature is calculated by the formula
KE = dim/2 N k T, where KE = total kinetic energy of the group of
atoms (sum of 1/2 m v\^2), dim = 2 or 3 = dimensionality of the
simulation, N = number of atoms in the group, k = Boltzmann constant,
and T = temperature.

A kinetic energy tensor, stored as a 6-element vector, is also
calculated by this compute for use in the computation of a pressure
tensor. The formula for the components of the tensor is the same as
the above formula, except that v\^2 is replaced by vx\*vy for the xy
component, etc. The 6 components of the vector are ordered xx, yy,
zz, xy, xz, yz.

The number of atoms contributing to the temperature is assumed to be
constant for the duration of the run; use the *dynamic* option of the
:doc:`compute_modify <compute_modify>` command if this is not the case.
However, in order to get meaningful result, the group ID of this compute should be all.

The removal of the cosine-shaped velocity component by this command is
essentially computing the temperature after a "bias" has been removed
from the velocity of the atoms.  If this compute is used with a fix
command that performs thermostatting then this bias will be subtracted
from each atom, thermostatting of the remaining thermal velocity will
be performed, and the bias will be added back in.  Thermostatting
fixes that work in this way include :doc:`fix nvt <fix_nh>`, :doc:`fix temp/rescale <fix_temp_rescale>`, :doc:`fix temp/berendsen <fix_temp_berendsen>`, and :doc:`fix langevin <fix_langevin>`.

This compute subtracts out degrees-of-freedom due to fixes that
constrain molecular motion, such as :doc:`fix shake <fix_shake>` and
:doc:`fix rigid <fix_rigid>`.  This means the temperature of groups of
atoms that include these constraints will be computed correctly.  If
needed, the subtracted degrees-of-freedom can be altered using the
*extra* option of the :doc:`compute_modify <compute_modify>` command.

See the :doc:`Howto thermostat <Howto_thermostat>` page for a
discussion of different ways to compute temperature and perform
thermostatting.

----------

Output info
"""""""""""

This compute calculates a global scalar (the temperature) and a global
vector of length 7, which can be accessed by indices 1-7.
The first 6 elements of the vector are the KE tensor,
and the 7-th is the cosine-shaped velocity amplitude :math:`V`,
which can be used to calculate the reciprocal viscosity, as shown in the example.
These values can be used by any command that uses global scalar or
vector values from a compute as input.
See the :doc:`Howto output <Howto_output>` page for an overview of LAMMPS output options.

The scalar value calculated by this compute is "intensive".  The
first 6 elements of vector values are "extensive",
and the 7-th element of vector values is "intensive".

The scalar value will be in temperature :doc:`units <units>`.  The
first 6 elements of vector values will be in energy :doc:`units <units>`.
The 7-th element of vector value will be in velocity :doc:`units <units>`.

Restrictions
""""""""""""

This command is only available when LAMMPS was built with the MISC package.
Since this compute depends on :doc:`fix accelerate/cos <fix_accelerate_cos>` which can
only work for 3d systems, it cannot be used for 2d systems.

Related commands
""""""""""""""""

:doc:`fix accelerate/cos <fix_accelerate_cos>`

Default
"""""""
 none

----------

.. _Hess1:

**(Hess)** Hess, B. The Journal of Chemical Physics 2002, 116 (1), 209-217.
