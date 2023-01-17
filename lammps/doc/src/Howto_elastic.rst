Calculate elastic constants
===========================

Elastic constants characterize the stiffness of a material. The formal
definition is provided by the linear relation that holds between the
stress and strain tensors in the limit of infinitesimal deformation.
In tensor notation, this is expressed as s_ij = C_ijkl \* e_kl, where
the repeated indices imply summation. s_ij are the elements of the
symmetric stress tensor. e_kl are the elements of the symmetric strain
tensor. C_ijkl are the elements of the fourth rank tensor of elastic
constants. In three dimensions, this tensor has 3\^4=81 elements. Using
Voigt notation, the tensor can be written as a 6x6 matrix, where C_ij
is now the derivative of s_i w.r.t. e_j. Because s_i is itself a
derivative w.r.t. e_i, it follows that C_ij is also symmetric, with at
most 7\*6/2 = 21 distinct elements.

At zero temperature, it is easy to estimate these derivatives by
deforming the simulation box in one of the six directions using the
:doc:`change_box <change_box>` command and measuring the change in the
stress tensor. A general-purpose script that does this is given in the
examples/elastic directory described on the :doc:`Examples <Examples>`
doc page.

Calculating elastic constants at finite temperature is more
challenging, because it is necessary to run a simulation that performs
time averages of differential properties. One way to do this is to
measure the change in average stress tensor in an NVT simulations when
the cell volume undergoes a finite deformation. In order to balance
the systematic and statistical errors in this method, the magnitude of
the deformation must be chosen judiciously, and care must be taken to
fully equilibrate the deformed cell before sampling the stress
tensor. Another approach is to sample the triclinic cell fluctuations
that occur in an NPT simulation. This method can also be slow to
converge and requires careful post-processing :ref:`(Shinoda) <Shinoda1>`

----------

.. _Shinoda1:

**(Shinoda)** Shinoda, Shiga, and Mikami, Phys Rev B, 69, 134103 (2004).
