The Maxwell deck
================

A Maxwell deck is a plain-text table of a spherically symmetric
viscoelastic body, of the kind used for glacial isostatic adjustment:
density and the elastic velocities beside a viscosity, one row per
knot, radius first.  A layer holding a viscosity is a Maxwell body in
shear, its relaxation time the viscosity over the shear modulus; the
elastic values are the unrelaxed ones, and the deck holds no quality
factors, since a constant-Q dispersion about a seismic reference period
would otherwise reach glacial periods.  A :doc:`prony_deck` holds any
number of Prony branches in place of the viscosity, and a
:doc:`burgers_deck` a Kelvin element beside it.

A deck is a table dense enough in knots that the choice of
interpolation between them does not matter in practice: the kinds the
readers join the knots by (below) are a reasonable default, not part of
the model's definition.  A model that is a particular function in each
layer (a given polynomial, say) belongs in a format that states that
function, not in a deck.

The Python writes one with ``planetmodel.write_deck(path, deck,
MAXWELL)``, a deck from any model being ``sample_deck(model,
MAXWELL.columns)``, and reads one with ``read_deck(path, MAXWELL)``, the
numbers and the header, or as a model with ``MaxwellDeckModel(path)``.
In C++, planetmodel's own library (:doc:`../cpp`) reads one with
``ReadDeck(path, MAXWELL())`` and writes one with ``WriteDeck``, the
same text the Python writes, character for character.

Layout
------

.. code-block:: text

   PREM with a Maxwell mantle
   r rho vpv vsv vph vsh eta viscosity
   # layers inner_core outer_core lower_mantle upper_mantle lid crust
   # interfaces icb cmb d670 lab moho surface
            0.000         13088.5         11262.2          3667.8  ...
       174500.000        13081.87       11257.426       3664.4635  ...

1. **The title**, a line of free text, the model's name.
2. **The column line**, the words ``r rho vpv vsv vph vsh eta
   viscosity`` and nothing else, except a ``#`` comment after them.
3. **The naming lines**, optional, each straight after the column line
   or the other: ``# layers`` followed by one name per layer and
   ``# interfaces`` followed by one name per interface, centre outward,
   ``-`` for one that has no name.  A name is one word without ``#``.
   There is one interface per layer, the outermost being the surface,
   and one more where the body is hollow (its first radius above zero),
   its inner boundary; a line naming another number is refused.
   Without a ``# layers`` line the layers are unnamed, and without a
   ``# interfaces`` line the interfaces are too, but for ``surface``,
   the outermost.  Any other ``#`` line is a comment.
4. **The table**, one row per knot of eight numbers separated by
   whitespace: the radius, then the seven columns.  Blank lines and
   ``#`` comments are skipped.  The writers put the radius as
   ``%14.3f`` and each value as ``%16.8g``.

Columns
-------

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

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
     - P velocities, vertical and horizontal
   * - ``vsv``, ``vsh``
     - m/s
     - S velocities, vertical and horizontal; zero in a fluid
   * - ``eta``
     - 1
     - the anisotropy parameter F / (A - 2L), 1 where isotropic
   * - ``viscosity``
     - Pa s
     - the dynamic viscosity of the Maxwell body; ``nan`` in a layer
       that is elastic or fluid

The layers
----------

A radius that appears twice marks a boundary: the first of the two
rows belongs to the layer below, the second to the layer above, so a
boundary is recorded even where the values agree across it.  A radius
appears at most twice, and every layer has at least two knots.

A column that is ``nan`` on every knot of a layer is absent from that
layer: that is how the table carries a viscosity that some layers do
not have.  A column that is ``nan`` on some knots of a layer and not
others is refused.

The readers join each layer's knots: the velocities and the density by a
not-a-knot cubic spline (the parabola through three knots, the line
through two), and the viscosity in its logarithm by straight lines,
``log-linear``, so that between two knots of viscosities ``eta_1`` and
``eta_2`` it is geometric, ``eta_1 (eta_2 / eta_1)^t`` a fraction ``t``
of the way from one to the other.  A viscosity changes by orders of
magnitude within a layer: a cubic through the values overshoots, below
zero where it falls steeply, and straight lines through them lie above
the geometric values between knots by the ratio of the two knots'
arithmetic and geometric means, 1.74 between knots a decade apart.  A
viscosity in a layer holding one is therefore positive and finite at
every knot; a viscosity constant through a layer is that constant.  The
Python's ``MaxwellDeckModel(path, kinds={"viscosity": "linear"})``, or
``DeckLayers(deck, "cubic", {{"viscosity", "linear"}})`` in C++, joins
the values themselves by straight lines instead, and any column may be
given any kind, through the values (``cubic``, ``natural``, ``linear``,
and in the Python ``pchip`` and ``akima``) or through their logarithms
(the same after ``log-``).  The Python and the C++ give the same
functions, values and derivatives, to rounding.  A model read from a
deck and written to a :doc:`model_file` comes back as the same
functions, a column joined in its logarithm by the file's radial rules
in the logarithm.
