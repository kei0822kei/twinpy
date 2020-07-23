======
phonon
======

In this page, some formulation for phonon calculation

.. contents::
   :depth: 2
   :local:


Crystal Rotaion for VASP
========================

Twinpy can creates hexagonal twin structures and twin boundaries structures.
After creating these structures, crystal standardization is take place
for VASP calculation.
After all necessary calculations are finished, it is interesting to
compare with the structures which are a little bit different from
each other corresponding to different shear strain ratio.
However, before comparing, it is necessary to transform structures
from the coordinates suitable for VASP calculations to the ones
which are easiest to analize their difference.

Let, the shear lattice basis as :math:`[\boldsymbol{a}_{s},
\boldsymbol{b}_{s}, \boldsymbol{c}_{s}]` is defined as

.. math::

   [\boldsymbol{a}_{s},
    \boldsymbol{b}_{s},
    \boldsymbol{c}_{s}]
   =
   [\boldsymbol{e}_x,
    \boldsymbol{e}_y,
    \boldsymbol{e}_z]
   \boldsymbol{M}_{s}.

and let transformation matrix :math:`\boldsymbol{P}^{-1}`
to standardized lattice, then

.. math::

   [\boldsymbol{a}_{std},
    \boldsymbol{b}_{std},
    \boldsymbol{c}_{std}]
   =
   [\boldsymbol{a}_{s},
    \boldsymbol{b}_{s},
    \boldsymbol{c}_{s}] \boldsymbol{P}^{-1}

equivalently

.. math::

   \boldsymbol{M}_{std} = \boldsymbol{M}_{s} \boldsymbol{P}^{-1}

Note in this stage, crystal body IS NOT ROTATED.

Next, let the conventional and primitive lattice of
this crystal as :math:`\boldsymbol{\bar{M}}_{std}`
and :math:`\boldsymbol{\bar{M}}_{prim}`:

.. math::

   [\boldsymbol{\bar{a}}_{std},
    \boldsymbol{\bar{b}}_{std},
    \boldsymbol{\bar{c}}_{std}]
   =
   [\boldsymbol{e}_x,
    \boldsymbol{e}_y,
    \boldsymbol{e}_z]
   \boldsymbol{\bar{M}}_{std}

and

.. math::

   [\boldsymbol{\bar{a}}_{prim},
    \boldsymbol{\bar{b}}_{prim},
    \boldsymbol{\bar{c}}_{prim}]
   =
   [\boldsymbol{e}_x,
    \boldsymbol{e}_y,
    \boldsymbol{e}_z]
   \boldsymbol{\bar{M}}_{prim}.

Relation between :math:`\boldsymbol{\bar{M}}_{std}` and
:math:`\boldsymbol{\bar{M}}_{prim}` depends on lattice centering
(you can find in spglib documentation 'Definitions and conventions')

.. math::

   [\boldsymbol{\bar{a}}_{prim},
    \boldsymbol{\bar{b}}_{prim},
    \boldsymbol{\bar{c}}_{prim}]
   =
   [\boldsymbol{\bar{a}}_{std},
    \boldsymbol{\bar{b}}_{std},
    \boldsymbol{\bar{c}}_{std}]
   \boldsymbol{P}_c

equivalently

.. math::

   \boldsymbol{\bar{M}}_{prim}
   =
   \boldsymbol{\bar{M}}_{std} \boldsymbol{P}_c

where :math:`\bar{}` indicates crystal body is rotated
from the original structure. Rotation matrix :math:`\boldsymbol{R}`
is defined as

.. math::

   [\boldsymbol{\bar{a}}_{std},
    \boldsymbol{\bar{b}}_{std},
    \boldsymbol{\bar{c}}_{std}]
   =
   [\boldsymbol{R}\boldsymbol{a}_{std},
    \boldsymbol{R}\boldsymbol{b}_{std},
    \boldsymbol{R}\boldsymbol{c}_{std}]

equivalently

.. math::

   \boldsymbol{\bar{M}}_{std} = \boldsymbol{R}\boldsymbol{M}_{std}.

For summary

.. math::

   \boldsymbol{\bar{M}}_{prim}
   =
   \boldsymbol{\bar{M}}_{std} \boldsymbol{P}_c
   =
   \boldsymbol{RM}_{std} \boldsymbol{P}_c
   =
   \boldsymbol{RM}_{s} \boldsymbol{P}^{-1} \boldsymbol{P}_c


Qpoints Transformation
======================

