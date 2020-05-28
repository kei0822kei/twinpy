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
   \gamma = \frac{|r^2-3|}{r\sqrt{3}}

in {10-11} twin,

.. math::
   \gamma = \frac{|4r^2-9|}{4r\sqrt{3}}

in {11-22} twin,

.. math::
   \gamma = \frac{2|r^2-2|}{3r}

in {11-21} twin,

.. math::
   \gamma = \frac{1}{r}.


twin indices
------------

You can find twin indices in the paper by [Yoo]_.

======================== ======================== =========================== =========================== ======================
:math:`\boldsymbol{K}_1` :math:`\boldsymbol{K}_2` :math:`\boldsymbol{\eta}_1` :math:`\boldsymbol{\eta}_2` :math:`\boldsymbol{S}`
======================== ======================== =========================== =========================== ======================
{ 1  0 -1  2 }           { 1  0 -1 -2 }               < 1  0 -1 -1 >              <-1  0  1 -1>           { 1  1 -2  0 }
{ 1  0 -1  1 }           { 1  0 -1 -3 }               < 1  0 -1 -2 >              < 3  0 -3  2>           { 1  1 -2  0 }
{ 1  1 -2  1 }           { 0  0  0  2 }           1/3 <-1 -1  2  6 >          1/3 < 1  1 -2  0>           { 1  0 -1  0 }
{ 1  1 -2  2 }           { 1  1 -2 -4 }           1/3 < 1  1 -2 -3 >          1/3 < 2  2 -4  3>           { 1  0 -1  0 }
======================== ======================== =========================== =========================== ======================

In actual, one has to determine specific plane and direction for each twin mode.
[Christian]_ proposed the method for determining them and
``twinpy`` follows this convention which is

  1. Vector :math:`\boldsymbol{m}` (, which is normal to shear plane),
     :math:`\boldsymbol{\eta}_1` and
     :math:`\boldsymbol{\eta}_2` form right hand system.
  2. The angle between :math:`\boldsymbol{\eta}_1`
     and :math:`\boldsymbol{\eta}_2` are obtuse.
  3. The angle between :math:`\boldsymbol{\eta}_1`
     and :math:`\boldsymbol{k}_2` are acute.
  4. The angle between :math:`\boldsymbol{\eta}_2`
     and :math:`\boldsymbol{k}_1` are acute.

In addition, ``twinpy`` add the following role as

  5. K1 plane is determined to meet the condition that
     the distance from K1 plane to the nearest atoms is minimum.
     This condition depends on wyckoff position.

In twinpy code, indices are set in order of

   a. k1    condition 5.
   b. eta2  condition 4.
   c. eta1  condition 2.
   d. k2    condition 3.
   e. m     condition 1.

References
----------
.. [Yoo] M. H. Yoo, MTA **12**, 409-418 (1981).
.. [Christian] W. Christian, et al., Progress in Progress in Materials Science. **39**, 1 (1995).
