.. index:: compute angle

compute angle command
=====================

Syntax
""""""

.. parsed-literal::

   compute ID group-ID angle

* ID, group-ID are documented in :doc:`compute <compute>` command
* angle = style name of this compute command

Examples
""""""""

.. code-block:: LAMMPS

   compute 1 all angle

Description
"""""""""""

Define a computation that extracts the angle energy calculated by each of the
angle sub-styles used in the :doc:`angle_style hybrid <angle_hybrid>` command.
These values are made accessible for output or further processing by other
commands.  The group specified for this command is ignored.

This compute is useful when using :doc:`angle_style hybrid <angle_hybrid>` if
you want to know the portion of the total energy contributed by one or more of
the hybrid sub-styles.

Output info
"""""""""""

This compute calculates a global vector of length N where N is the number of
sub_styles defined by the :doc:`angle_style hybrid <angle_style>` command,
which can be accessed by indices 1-N.  These values can be used by any command
that uses global scalar or vector values from a compute as input.  See the
:doc:`Howto output <Howto_output>` page for an overview of LAMMPS output
options.

The vector values are "extensive" and will be in energy
:doc:`units <units>`.

Restrictions
""""""""""""

none

Related commands
""""""""""""""""

:doc:`compute pe <compute_pe>`, :doc:`compute pair <compute_pair>`

Default
"""""""

none
