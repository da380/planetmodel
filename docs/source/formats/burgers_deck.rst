The burgers deck
================

A burgers deck is a plain-text table of a spherically symmetric
viscoelastic body whose layers are Maxwell or Burgers bodies in shear:
the columns of a :doc:`maxwell_deck`, density, the unrelaxed elastic
velocities and a viscosity, and two more for the Kelvin element of a
Burgers body, one row per knot, radius first.  A Burgers body is a
Maxwell element, of the layer's elastic shear modulus ``mu`` and the
viscosity ``eta``, in series with a Kelvin-Voigt element of viscosity
``eta_K`` and shear modulus ``f mu``; its compliance in the Laplace
variable ``s`` is

.. code-block:: text

   1 / M(s) = 1 / mu + 1 / (eta s) + 1 / (f mu + eta_K s)

so that it responds elastically at once, creeps in a transient over
times near ``eta_K / (f mu)``, and flows with the steady viscosity
``eta``.  It is exactly the generalised Maxwell body of two Prony
branches whose strengths sum to one, which a :doc:`prony_deck` holds:
the burgers deck is that deck's reduced form, as a Maxwell deck is of
one branch.  The relaxation is in shear alone; the bulk modulus is
elastic.  The shear modulus ``mu`` is the Voigt average of the elastic
values, ``(A + C - 2F + 5N + 6L) / 15`` with ``A = rho vph^2``, ``C =
rho vpv^2``, ``L = rho vsv^2``, ``N = rho vsh^2`` and ``F = eta (A -
2L)``.

The Python writes one with ``planetmodel.write_deck(path, deck,
BURGERS)``, a deck from any model whose layers are Maxwell or Burgers
bodies being ``conversion.prony.burgers_deck(model)``, and reads one with
``read_deck(path, BURGERS)``, the numbers and the header, or as a model
with ``BurgersDeckModel(path)``, whose Burgers layers hold their two
Prony branches (``conversion.prony.burgers_terms``).  In C++,
planetmodel's own library (:doc:`../cpp`) reads one with ``ReadDeck(path,
BURGERS())`` and each layer's columns as functions of the radius with
``DeckLayers(deck, "cubic", BURGERS_KINDS)``, and writes one with
``WriteDeck``, the same text the Python writes, character for character.

Layout
------

.. code-block:: text

   PREM with a Burgers asthenosphere
   r rho vpv vsv vph vsh eta viscosity kelvin_viscosity kelvin_modulus_fraction
   # layers inner_core outer_core lower_mantle transition_zone asthenosphere lid crust
   # interfaces icb cmb d670 d220 d80 moho surface
            0.000         13088.5  ...               1             nan             nan             nan
   ...
      6151000.000        3435.775  ...               1           1e+21             nan             nan
      6151000.000       3359.4904  ...      0.97646208           1e+19           1e+18               1
      6179000.000       3362.5334  ...      0.96557236           1e+19           1e+18               1

Below 220 km the mantle is a Maxwell body of 1e21 Pa s; above it, a
Burgers body of steady viscosity 1e19 Pa s whose Kelvin element has a
tenth of that viscosity and the layer's shear modulus.

1. **The title**, a line of free text, the model's name.
2. **The column line**, the words ``r rho vpv vsv vph vsh eta viscosity
   kelvin_viscosity kelvin_modulus_fraction`` and nothing else, except a
   ``#`` comment after them.
3. **The naming lines**, optional, as in a :doc:`maxwell_deck`: ``#
   layers`` followed by one name per layer and ``# interfaces`` followed
   by one name per interface, centre outward, ``-`` for one that has no
   name.
4. **The table**, one row per knot of ten numbers separated by
   whitespace: the radius, then the nine columns.  Blank lines and ``#``
   comments are skipped.  The writers put the radius as ``%14.3f`` and
   each value as ``%16.8g``.

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
   * - ``viscosity``
     - Pa s
     - the Maxwell element's viscosity, the body's steady viscosity, as a
       Maxwell deck's
   * - ``kelvin_viscosity``
     - Pa s
     - the Kelvin element's viscosity ``eta_K``
   * - ``kelvin_modulus_fraction``
     - 1
     - the Kelvin element's shear modulus as a fraction ``f`` of the
       layer's elastic shear modulus ``mu``

The layers
----------

A radius that appears twice marks a boundary, and every layer has at
least two knots, as in a :doc:`maxwell_deck`.

Each of the last three columns is a number at every knot of a layer or
``nan`` at every one, and a layer holds one of three combinations: none
of them, an elastic or fluid layer; the viscosity alone, a Maxwell body,
as in a Maxwell deck; or all three, a Burgers body.  Any other
combination is refused, as is a value that is not positive and finite.
The readers and the writers both hold a deck to these rules, naming the
layer, the column and the radius of the first value that breaks them.

The readers join each layer's knots by a piecewise polynomial: the
velocities and the density by a not-a-knot cubic spline (the parabola
through three knots, the line through two), the three rheological
columns by straight lines, which keep them positive between knots, as a
Maxwell deck's viscosity is joined.  The Python and the C++ give the same
functions.

``BurgersDeckModel`` gives a Maxwell layer its viscosity and a Burgers
layer its two Prony branches at every point, of the joined columns and
the layer's shear modulus, which the library's rheology reads.
``conversion.prony.burgers_deck`` writes a model back, finding a Burgers
layer's viscosity and Kelvin element from its two branches
(``conversion.prony.burgers_parameters``): with ``s = w_1 tau_1 + w_2
tau_2`` and ``d = w_1 w_2 (tau_2 - tau_1)^2``, ``eta = mu s``, ``eta_K =
mu s tau_1 tau_2 / d`` and ``f = s^2 / d``.  Any two branches whose
strengths sum to one and whose times differ are a Burgers body, and one
branch of strength one is a Maxwell body; a layer of other branches is
no layer of a burgers deck, and is refused.
