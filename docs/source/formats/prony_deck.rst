The prony deck
==============

A prony deck is a plain-text table of a spherically symmetric
viscoelastic body whose shear relaxes as a generalised Maxwell body:
density and the unrelaxed elastic velocities, as in a
:doc:`maxwell_deck`, and in place of its viscosity one pair of columns
for each Prony branch, one row per knot, radius first.  Branch k
relaxes the fraction ``w_k`` of the unrelaxed shear modulus ``mu`` with
the time ``tau_k``, so that the shear modulus at angular frequency
``omega`` is

.. code-block:: text

   mu(omega) = mu [1 - sum_k w_k / (1 + i omega tau_k)]

and ``(1 - sum_k w_k) mu`` is left at long times: strengths summing to
one give a body that flows, less a relaxed modulus.  The relaxation is
in shear alone; the bulk modulus is elastic.  A Maxwell deck is the case
of one branch, ``w_1 = 1`` and ``tau_1`` the viscosity over the shear
modulus, and a :doc:`burgers_deck` that of two summing to one.  The
elastic values are the unrelaxed ones, and the shear modulus ``mu`` is
their Voigt average, ``(A + C - 2F + 5N + 6L) / 15`` with ``A = rho
vph^2``, ``C = rho vpv^2``, ``L = rho vsv^2``, ``N = rho vsh^2`` and
``F = eta (A - 2L)``.

The Python writes one with ``planetmodel.write_deck(path, deck, PRONY)``,
a deck from any model being ``conversion.prony.prony_deck(model)``, which
writes a layer holding a viscosity as its one branch and a layer holding
Prony branches as they are, and reads one with ``read_deck(path,
PRONY)``, the numbers and the header, or as a model with
``PronyDeckModel(path)``.  In C++, planetmodel's own library
(:doc:`../cpp`) reads one with ``ReadDeck(path, PRONY())``, gives the
number of branches with ``PronyBranches(deck)`` and each layer's columns
as functions of the radius with ``DeckLayers(deck, "cubic",
PRONY_KINDS)``, and writes one with ``WriteDeck``, the same text the
Python writes, character for character.

Layout
------

.. code-block:: text

   PREM with a Burgers asthenosphere
   r rho vpv vsv vph vsh eta relaxation_strength_1 relaxation_time_1 relaxation_strength_2 relaxation_time_2
   # layers inner_core outer_core lower_mantle transition_zone asthenosphere lid crust
   # interfaces icb cmb d670 d220 d80 moho surface
            0.000         13088.5  ...               1             nan             nan             nan             nan
   ...
      6151000.000        3435.775  ...               1               1    1.349614e+10               0    1.349614e+10
      6151000.000       3359.4904  ...      0.97646208      0.52496881         7431693      0.47503119   3.1269765e+08
      6179000.000       3362.5334  ...      0.96557236      0.52496881       7392249.5      0.47503119   3.1103802e+08

Below 220 km the mantle is a Maxwell body, one branch of strength one,
the second written with a strength of 0 and the first's time; above it,
a Burgers body of two branches summing to one.

1. **The title**, a line of free text, the model's name.
2. **The column line**, the words ``r rho vpv vsv vph vsh eta`` and then
   ``relaxation_strength_k relaxation_time_k`` for k = 1, 2, ..., N in
   that order, and nothing else, except a ``#`` comment after them.  N,
   the number of branches, is at least one and is read from this line;
   it is the same on every layer.  A line whose pairs are incomplete,
   out of order or skip an index is refused, saying which word stands
   where another belongs, as is a table whose rows hold another number
   of values.
3. **The naming lines**, optional, as in a :doc:`maxwell_deck`: ``#
   layers`` followed by one name per layer and ``# interfaces`` followed
   by one name per interface, centre outward, ``-`` for one that has no
   name.
4. **The table**, one row per knot of 7 + 2N numbers separated by
   whitespace: the radius, the six elastic columns, then the branches.
   Blank lines and ``#`` comments are skipped.  The writers put the
   radius as ``%14.3f`` and each value as ``%16.8g``.

Columns
-------

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - column
     - unit
     - meaning
   * - ``r``
     - m
     - the radius from the centre, non-decreasing down the table
   * - ``rho``
     - kg/m^3
     - density
   * - ``vpv``, ``vph``
     - m/s
     - P velocities, vertical and horizontal, unrelaxed
   * - ``vsv``, ``vsh``
     - m/s
     - S velocities, vertical and horizontal, unrelaxed; zero in a fluid
   * - ``eta``
     - 1
     - the anisotropy parameter F / (A - 2L), 1 where isotropic
   * - ``relaxation_strength_k``
     - 1
     - the fraction ``w_k`` of the unrelaxed shear modulus that relaxes in
       branch k, in [0, 1]
   * - ``relaxation_time_k``
     - s
     - the relaxation time ``tau_k`` of branch k, positive and finite

The layers
----------

A radius that appears twice marks a boundary, and every layer has at
least two knots, as in a :doc:`maxwell_deck`.

A layer is either elastic, ``nan`` in every branch column, or
viscoelastic, a number in every branch column at every knot; a layer
holding ``nan`` in some branch columns and numbers in others is refused.
A viscoelastic layer that uses fewer than N branches holds a strength of
0 in each branch it does not use, and a positive time there: the
writers give such a branch the time of the layer's last branch at the
same knot, so that it brings no new time into the layer, and a reader
takes any positive time.  At every knot each strength lies in [0, 1],
the strengths sum to at most one, and each time is positive and
finite; the sum may pass one by 1e-8 for each branch, the rounding of
eight significant digits.  The readers and the writers both hold a deck
to these rules, naming the layer, the column and the radius of the
first value that breaks them.

The readers join each layer's knots by a piecewise polynomial: the
velocities and the density by a not-a-knot cubic spline (the parabola
through three knots, the line through two), every branch column by
straight lines.  Straight lines keep every strength in [0, 1], their sum
at most one and every time positive between knots, and join the times,
which can change by orders of magnitude within a layer, as a Maxwell
deck's viscosity is joined.  A one-branch deck made from a Maxwell deck
is the same body at the knots; between them its time is the line
through the knots' viscosity over shear modulus, where the Maxwell deck
divides the line of the viscosity by the cubic of the modulus, and the
two differ at second order in the spacing of the knots.  The Python and
the C++ give the same functions.

``PronyDeckModel`` makes each layer's branches its fields
``relaxation_strength_k`` and ``relaxation_time_k``, which the library's
rheology reads, leaving out of a layer the branches after the last whose
strength is not zero at every knot, so that a model whose layers hold
different numbers of branches is read back as it was written.
