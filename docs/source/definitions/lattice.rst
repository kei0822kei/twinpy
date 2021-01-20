=======
lattice
=======

In this page, some formulations are provided used in hexagonal.py file.

.. contents::
   :depth: 2
   :local:


Lattice Matrix
==============

Inner Product and Metric Tensor
-------------------------------

Given lattice matrix :math:`\boldsymbol{L}`
and two vectors :math:`(x_1, y_1, z_1)^{T}` and :math:`(x_2, y_2, z_2)^{T}`,
inner product can calculated as

.. math::

   [\boldsymbol{a},
    \boldsymbol{b},
    \boldsymbol{c}]
   =
   [\boldsymbol{e}_x,
    \boldsymbol{e}_y,
    \boldsymbol{e}_z] \boldsymbol{L}

.. math::

   (
   \boldsymbol{L}
   \begin{pmatrix}
   x_1 \\
   y_1 \\
   z_1 \\
   \end{pmatrix}
   )^{T}
   \boldsymbol{L}
   \begin{pmatrix}
   x_2 \\
   y_2 \\
   z_2 \\
   \end{pmatrix}
   =
   (x_1, y_1, z_1) \boldsymbol{G}
   \begin{pmatrix}
   x_2 \\
   y_2 \\
   z_2 \\
   \end{pmatrix}

where metric tensor :math:`\boldsymbol{G}`

.. math::

   \boldsymbol{G} = \boldsymbol{L}^{T} \boldsymbol{L}


Hexagonal Direction
===================

Hexagonal Lattice
-----------------

To create this object, hexagonal lattice :math:`\boldsymbol{H}`
and plane indices (three or four) are needed.

.. math::

   [\boldsymbol{a}_1,
    \boldsymbol{a}_2,
    \boldsymbol{c}]
   =
   [\boldsymbol{e}_x,
    \boldsymbol{e}_y,
    \boldsymbol{e}_z] \boldsymbol{H}

where

.. math::

   \boldsymbol{H}
   =
   \begin{pmatrix}
   a & -\frac{1}{2}a & 0 \\
   0 & \frac{\sqrt{3}}{2}a & 0 \\
   0 & 0 & c \\
   \end{pmatrix}.


indices
-------

Let input direction :math:`\boldsymbol{k}` is :math:`(UVW)`
in three indices and :math:`(uvtw)` in four indices.
There are the relation between two kinds of direction indices as

.. math::

  \begin{pmatrix}
    U \\
    V \\
    W \\
  \end{pmatrix}
  =
  \begin{pmatrix}
    u \\
    v \\
    w \\
  \end{pmatrix}

and

.. math::

  \begin{pmatrix}
    u \\
    v \\
    t \\
    w \\
  \end{pmatrix}
  =
  \begin{pmatrix}
    ( 2  U - V ) / 3 \\
    ( 2  V - U ) / 3 \\
    - ( u + v ) \\
    W \\
  \end{pmatrix}.


Hexagonal Plane
===============

indices
-------

Let input plane :math:`\boldsymbol{K}` is :math:`(HKL)`
in three indices and :math:`(hkil)` in four indices.
There are the relation between two kinds of plane indices as

.. math::

  \begin{pmatrix}
    H \\
    K \\
    L \\
  \end{pmatrix}
  =
  \begin{pmatrix}
    h \\
    k \\
    l \\
  \end{pmatrix}

and

.. math::

  \begin{pmatrix}
    h \\
    k \\
    i \\
    l \\
  \end{pmatrix}
  =
  \begin{pmatrix}
    H \\
    K \\
    -( H + K ) \\
    L \\
  \end{pmatrix}.


direction normal to plane
-------------------------

When the vector normal to :math:`\boldsymbol{K}` is :math:`\boldsymbol{k}`,
then

.. math::

   [\boldsymbol{a}_1,
    \boldsymbol{a}_2,
    \boldsymbol{c}]
   \boldsymbol{k}
   =
   [\boldsymbol{a}^*_1,
    \boldsymbol{a}^*_2,
    \boldsymbol{c}^*]
  \begin{pmatrix}
    H \\
    K \\
    L \\
  \end{pmatrix}
  =
  [\boldsymbol{e}_x,
   \boldsymbol{e}_y,
   \boldsymbol{e}_z]
  \boldsymbol{H}^*
  \begin{pmatrix}
    H \\
    K \\
    L \\
  \end{pmatrix}
  =
  [\boldsymbol{a}_1,
   \boldsymbol{a}_2,
   \boldsymbol{c}]
  \boldsymbol{H}^{-1}
  \boldsymbol{H}^*
  \begin{pmatrix}
    H \\
    K \\
    L \\
  \end{pmatrix}

so

.. math::

   \boldsymbol{k}
   =
  \boldsymbol{H}^{-1}
  \boldsymbol{H}^*
  \begin{pmatrix}
    H \\
    K \\
    L \\
  \end{pmatrix}

where :math:`\boldsymbol{H}^*` is reciprocal lattice of hexagonal lattice
:math:`\boldsymbol{H}`.


distance from plane
-------------------

Let :math:`\boldsymbol{x}` is the position in the basis of hexagonal lattice.
Distance from the plane :math:`d` are

.. math::

   d
   =
   |\boldsymbol{e}_{\boldsymbol{k}}
   \cdot \boldsymbol{H} \boldsymbol{x}|

where `\boldsymbol{e}_{\boldsymbol{k}}` is unit vector along
:math:`\boldsymbol{k}` direction on the basis of
:math:`[\boldsymbol{e}_x, \boldsymbol{e}_y, \boldsymbol{e}_z]`

.. math::
   \boldsymbol{e}_{\boldsymbol{k}}
   =
   \frac{\boldsymbol{H}\boldsymbol{k}}{|\boldsymbol{H}\boldsymbol{k}|}
