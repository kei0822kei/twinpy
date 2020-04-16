==========
properties
==========

.. contents::
   :depth: 2
   :local:


hexagonal close-packed
======================

Hexagonal lattice is already defined in the lattice page.
There are two options how each atom set. First,
two atoms :math:`\boldsymbol{X}_{atm}` are placed Wyckoff 'c' (No. 194) as

.. math::

  \boldsymbol{X}_{atm}
  =
  \begin{pmatrix}
    \frac{1}{3} & -\frac{1}{3} \\
   -\frac{1}{3} &  \frac{1}{3} \\
    \frac{1}{4} & -\frac{1}{4} \\
  \end{pmatrix}

and the other is Wyckoff 'd' as

.. math::

  \boldsymbol{X}_{atm}
  =
  \begin{pmatrix}
    \frac{1}{3} & -\frac{1}{3} \\
   -\frac{1}{3} &  \frac{1}{3} \\
   -\frac{1}{4} &  \frac{1}{4} \\
  \end{pmatrix}

with the :math:`[\boldsymbol{a}_1, \boldsymbol{a}_2, \boldsymbol{c}]` basis.


twin mode
=========

Currently, {10-11}, {10-12}, {11-21} and {11-22} twins are supported.


shear strain function
---------------------

The shear strain function :math:`\gamma` are provided in (Yoo. 1981).
When :math:`r (= \frac{c}{a})` are given,

in {10-12} twin,

.. math::
   \gamma = \frac{r^2-3}{r\sqrt{3}}

in {10-11} twin,

.. math::
   \gamma = \frac{4r^2-9}{4r\sqrt{3}}

in {11-22} twin,

.. math::
   \gamma = \frac{2(r^2-2)}{3r}

in {11-21} twin,

.. math::
   \gamma = \frac{1}{r}.



FUTURE EDITED
=============

twin indices
------------

You can find customs how to set the four indices
in the paper 'DEFORMATION TWINNING' by Christian (1995).
Abstruct is as bellow:

1. Vector :math:`\boldsymbol{m}` (, which is normal to shear plane),
   :math:`\boldsymbol{\eta}_1` and :math:`\boldsymbol{\eta}_2`
   form right hand system.
2. The angle between :math:`\boldsymbol{\eta}_1`
   and :math:`\boldsymbol{\eta}_2` are obtuse.
3. The angle between :math:`\boldsymbol{\eta}_1`
   and :math:`\boldsymbol{k}_2` are acute.
4. The angle between :math:`\boldsymbol{\eta}_2`
   and :math:`\boldsymbol{k}_1` are acute.

In this algorithm, indices are set in order of:

- :math:`\boldsymbol{k}_1`  set first
- :math:`\boldsymbol{\eta}_2`  set using condition 4
- :math:`\boldsymbol{\eta}_1`  set using condition 2
- :math:`\boldsymbol{k}_2`  set using condition 3
- :math:`\boldsymbol{m}`  set using condition 1.

where :math:`\boldsymbol{k}_1`, :math:`\boldsymbol{k}_2`
and :math:`\boldsymbol{m}` are normal to
:math:`\boldsymbol{K}_1`, :math:`\boldsymbol{K}_2`
and :math:`\boldsymbol{S}` plane.
They all are positive direction toward each plane.


algorothm
^^^^^^^^^

First, default indices are given which you can find in Yoo. 1981.

======================== ======================== =========================== =========================== ======================
:math:`\boldsymbol{K}_1` :math:`\boldsymbol{K}_2` :math:`\boldsymbol{\eta}_1` :math:`\boldsymbol{\eta}_2` :math:`\boldsymbol{S}`
======================== ======================== =========================== =========================== ======================
(1 0 -1 2)               ( 1 0 -1 -2)                [ 1  0 -1 -1]               [-1 0  1 -1]             ( 1 1 -2 0)
(1 0 -1 1)               ( 1 0 -1 -3)                [ 1  0 -1 -2]               [ 3 0 -3  2]             ( 1 1 -2 0)
(1 1 -2 1)               ( 0 0  0  2)             1/3[-1 -1  2  6]            1/3[ 1 1 -2  0]             ( 1 0 -1 0)
(1 1 -2 2)               ( 1 1 -2 -4)             1/3[ 1  1 -2 -3]            1/3[ 2 2 -4  3]             ( 1 0 -1 0)
======================== ======================== =========================== =========================== ======================

Then, each plane and direction is reset as follows.

- set :math:`\boldsymbol{K}_1` and calculate :math:`\boldsymbol{k}_1`
- check whether condition 4. is fulfilled
- if not, reset :math:`\boldsymbol{\eta}_2` to :math:`-\boldsymbol{\eta}_2`
- check whether condition 2. is fulfilled
- if not, reset :math:`\boldsymbol{\eta}_1` to :math:`-\boldsymbol{\eta}_1`
- check whether condition 3. is fulfilled
- if not, reset :math:`\boldsymbol{k}_2` to :math:`-\boldsymbol{k}_2`
  and :math:`\boldsymbol{K}_2` to :math:`-\boldsymbol{K}_2`
- look for the direction :math:`\boldsymbol{m}` which fulfill condition 1.

In the last step, first enumerate every equivalent plane
with original plane :math:`\boldsymbol{S}`.
When :math:`\boldsymbol{S} = (h,k,i,l)`, all candidates are
(h, k, i, l), (h, i, k, l), (k, h, i, l),
(k, i, h, l), (i, h, k, l), and (i, k, i, l).
In each candidate, :math:`\boldsymbol{m}` is calculated
and check :math:`\boldsymbol{m}` normal to
:math:`\boldsymbol{k}_1` :math:`\boldsymbol{k}_2`
:math:`\boldsymbol{\eta}_1` and :math:`\boldsymbol{\eta}_2`.
Then, if abstructed :math:`\boldsymbol{m}` does not fulfill condition 4.,
reset :math:`\boldsymbol{m}` to :math:`-\boldsymbol{m}`.
