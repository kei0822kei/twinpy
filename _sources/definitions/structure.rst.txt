=========
structure
=========

In this page, some formulations are provided used in structure.py file.

.. contents::
   :depth: 2
   :local:


Hexagonal Structure
===================

set parent
----------

Parent lattice can be defined by specifying twin mode.
By this, the following properties are determined.

#. twin indices
#. shear strain functoin :math:`\gamma(r)`
#. parent matrix

where parent matrix :math:`\boldsymbol{M}` is constructed as bellow

.. math::

   [\boldsymbol{m}^{p},
    \boldsymbol{\eta}^{p}_1,
    \boldsymbol{\eta}^{p}_2]
   =
   [\boldsymbol{a}_1,
    \boldsymbol{a}_2,
    \boldsymbol{c}]
   \boldsymbol{M}.

where :math:`\boldsymbol{m}^{p}`,
:math:`\boldsymbol{\eta}^{p}_1` and
:math:`\boldsymbol{\eta}^{p}_2` are
a little bit different from the ones in 'get_twin_indices' in twinmode.py
because they are rescaled to be the vectors whose element are all integer.
Therefore, every element of supercell matrix
:math:`\boldsymbol{M}` is integer. The fractional points :math:`\boldsymbol{X}`
is transformed to :math:`\boldsymbol{X}^{p}` as

.. math::

   \boldsymbol{X}^{p} = \boldsymbol{M}^{-1} \boldsymbol{X}.


shear structure
---------------

Shear structure here means the structure which is sheared from parent to
twin structure. Let the bases of shear and twin structure is defined as
:math:`[\boldsymbol{m}^{s}, \boldsymbol{\eta}^{s}_{1},
\boldsymbol{\eta}^{s}_{2}]`
and
:math:`[\boldsymbol{m}^{t}, \boldsymbol{\eta}^{t}_{1},
\boldsymbol{\eta}^{t}_{2}]`
then

.. math::

   [\boldsymbol{m}^{s},
    \boldsymbol{\eta}^{s}_{1},
    \boldsymbol{\eta}^{s}_{2}]
   =
   [\boldsymbol{m}^{p},
    \boldsymbol{\eta}^{p}_{1},
    \boldsymbol{\eta}^{p}_{2}]
   \boldsymbol{S}

where

.. math::

   \boldsymbol{S}
   =
   \begin{pmatrix}
    1 & 0 & 0 \\
    0 & 1 & \alpha s \\
    0 & 0 & 1 \\
   \end{pmatrix}

where :math:`\alpha` is shear ratio.

The shear value :math:`s` is determined as below.
From the figure, shear vector from parent to twin
:math:`s \boldsymbol{\eta}^{p}_{1}` is

.. math::

   s \boldsymbol{\eta}^{p}_{1}
   =
   \frac{\gamma d}{|\boldsymbol{\eta}^{p}_{1}|}
   \boldsymbol{\eta}^{p}_{1}

so

.. math::

   s = \frac{\gamma d}{|\boldsymbol{\eta}^{p}_{1}|}

where :math:`d` is the distance of the point :math:`\boldsymbol{\eta}^{p}_{2}`
from :math:`\boldsymbol{K}_2` plane.
The bases of twin structure :math:`[\boldsymbol{m}^{t},
\boldsymbol{\eta}^{t}_{1}, \boldsymbol{\eta}^{t}_{2}]`
corresponds to :math:`\alpha=1`.

The fractional positions are unfixed during this transformation
from parent structure to shear and twin structure.

Let the bases of the primitive structure of the shear and twin structure
as
:math:`[\boldsymbol{a}^{s}_{1}, \boldsymbol{a}^{s}_{2}, \boldsymbol{c}^{s}]`
and
:math:`[\boldsymbol{a}^{t}_{1}, \boldsymbol{a}^{t}_{2}, \boldsymbol{c}^{t}]`
can be written as

.. math::

   [\boldsymbol{a}^{s}_{1}, \boldsymbol{a}^{s}_{2}, \boldsymbol{c}^{s}]
   =
   [\boldsymbol{a}_1,
    \boldsymbol{a}_2,
    \boldsymbol{c}] \boldsymbol{M} \boldsymbol{S} \boldsymbol{M}^{-1}.




FUTURE EDITED
=============

First, supercell matrix :math:`\boldsymbol{M}` is constructed as bellow

.. math::

   [\boldsymbol{m}^{p},
    \boldsymbol{\eta}^{p}_1,
    \boldsymbol{\eta}^{p}_2]
   =
   [\boldsymbol{a}_1,
    \boldsymbol{a}_2,
    \boldsymbol{c}]
   \boldsymbol{M}.

where :math:`\boldsymbol{m}^{p}`,
:math:`\boldsymbol{\eta}^{p}_1` and
:math:`\boldsymbol{\eta}^{p}_2` are
a little bit different from the ones in 'get_twin_indices' in twinmode.py
because they are rescaled to be the vectors whose element are all integer.
Therefore, every element of supercell matrix
:math:`\boldsymbol{M}` is integer.
In this step, the positions of the lattice points
in the original hexagonal lattice :math:`\boldsymbol{X}^{p}_{lat}` and
the relative positions of the atoms from each lattice point
:math:`\boldsymbol{X}^{p}_{atm}` are defined in
:math:`[\boldsymbol{m}^{p}, \boldsymbol{\eta}^{p}_1, \boldsymbol{\eta}^{p}_2]` basis
where

.. math::

   [\boldsymbol{a}_1,
    \boldsymbol{a}_2,
    \boldsymbol{c}]
   \boldsymbol{X}_{atm}
   =
   [\boldsymbol{m}^{p},
    \boldsymbol{\eta}^{p}_1,
    \boldsymbol{\eta}^{p}_2]
   \boldsymbol{X}^{p}_{atm}
   =
   [\boldsymbol{a}_1,
    \boldsymbol{a}_2,
    \boldsymbol{c}]
   \boldsymbol{M} \boldsymbol{X}^{p}_{atm}

so

.. math::

   \boldsymbol{X}^{p}_{atm} = \boldsymbol{M}^{-1} \boldsymbol{X}_{atm}

In the next step, the cartesian basis vectors from
:math:`[\boldsymbol{e}_x, \boldsymbol{e}_y, \boldsymbol{e}_z]` to
:math:`[\boldsymbol{e}_{\boldsymbol{m}^{p}},
\boldsymbol{e}_{\boldsymbol{\eta}^{p}_1},
\boldsymbol{e}_{\boldsymbol{k}^{p}_1}]`
where
:math:`\boldsymbol{e}_{\boldsymbol{m}^{p}}
= \frac{\boldsymbol{m}^{p}}{|\boldsymbol{m}^{p}|}`
:math:`\boldsymbol{e}_{\boldsymbol{\eta}^{p}_1}
= \frac{\boldsymbol{\eta}^{p}_1}{|\boldsymbol{\eta}^{p}_1|}`
:math:`\boldsymbol{e}_{\boldsymbol{k}^{p}_1}
= \frac{\boldsymbol{k}_1}{|\boldsymbol{k}_1|}`.
In this step, rotation matrix :math:`\boldsymbol{R}` is defined

.. math::

   [\boldsymbol{e}_{\boldsymbol{m}^{p}},
    \boldsymbol{e}_{\boldsymbol{\eta}^{p}_1},
    \boldsymbol{e}_{\boldsymbol{k}^{p}_1}]
   =
   [\boldsymbol{e}_x, \boldsymbol{e}_y, \boldsymbol{e}_z]
   \boldsymbol{R}

:math:`\boldsymbol{R}` is orthogonal matrix.

.. math::

   \boldsymbol{R}^{-1} = \boldsymbol{R}^{T}

If the number vectors :math:`\boldsymbol{X}_{0}` are transformed to
:math:`\boldsymbol{X}^{p}_{0}` by this operation, the following relations hold.

.. math::

   [\boldsymbol{e}_x, \boldsymbol{e}_y, \boldsymbol{e}_z]
   \boldsymbol{X}_{0}
   =
   [\boldsymbol{e}_{\boldsymbol{m}^{p}},
    \boldsymbol{e}_{\boldsymbol{\eta}^{p}_1},
    \boldsymbol{e}_{\boldsymbol{k}^{p}_1}]
   \boldsymbol{X}^{p}_{0}
   =
   [\boldsymbol{e}_x, \boldsymbol{e}_y, \boldsymbol{e}_z]
   \boldsymbol{R} \boldsymbol{X}^{p}_{0}

so

.. math::

   \boldsymbol{X}^{p}_{0}
   =
   \boldsymbol{R}^{-1} \boldsymbol{X}_{0}
   =
   \boldsymbol{R}^{T} \boldsymbol{X}_{0}

Be careful if you use `def pymatgen.core.structure.Structure.apply_operation`,
**its input rotation matrix is** :math:`\boldsymbol{R}^{T}`,
**NOT** :math:`\boldsymbol{R}`.

In the last step, parent lattice :math:`\boldsymbol{L}_p` is defined as

.. math::

   [\boldsymbol{m}^{p},
    \boldsymbol{\eta}^{p}_1,
    \boldsymbol{\eta}^{p}_2]
   =
   [\boldsymbol{a}_1, \boldsymbol{a}_2, \boldsymbol{c}]
   \boldsymbol{M}
   =
   [\boldsymbol{e}_x, \boldsymbol{e}_y, \boldsymbol{e}_z]
   \boldsymbol{H} \boldsymbol{M}
   =
   [\boldsymbol{e}_{\boldsymbol{m}^{p}},
    \boldsymbol{e}_{\boldsymbol{\eta}^{p}_1},
    \boldsymbol{e}_{\boldsymbol{k}^{p}_1}]
   \boldsymbol{R}^{-1} \boldsymbol{H} \boldsymbol{M}
   \equiv
   [\boldsymbol{e}_{\boldsymbol{m}^{p}},
    \boldsymbol{e}_{\boldsymbol{\eta}^{p}_1},
    \boldsymbol{e}_{\boldsymbol{k}^{p}_1}]
   \boldsymbol{L}_p

where

.. math::

   \boldsymbol{L}^p = \boldsymbol{R}^{-1} \boldsymbol{H} \boldsymbol{M}.

The coordinates of each points in
:math:`[\boldsymbol{m}^{p}, \boldsymbol{\eta}^{p}_1, \boldsymbol{\eta}^{p}_2]`
basis such as :math:`\boldsymbol{X}^{p}_{lat}` and
:math:`\boldsymbol{X}^{p}_{atm}`
**DOES NOT** be affected by this transformation.

**HOWEVER**, :math:`\boldsymbol{X}^{p}_{atm}` may have to revise
because two rigid atoms must be the nearest atoms from the specified
:math:`\boldsymbol{K}_1` plane. In the case you choose 'c' wyckoff letter,
this revising (probably) always occur.


get_sheared_structure
---------------------

When the shear ratio is :math:`\boldsymbol{r}`,
the basis vectors of sheared structure are given as

.. math::

   [\boldsymbol{m}^{s},
    \boldsymbol{\eta}^{s}_1,
    \boldsymbol{\eta}^{s}_2]
   =
   [\boldsymbol{m}^{p},
    \boldsymbol{\eta}^{p}_1,
    \boldsymbol{\eta}^{p}_2 + \boldsymbol{d}]

where :math:`\boldsymbol{d}` is given as

.. math::

   \boldsymbol{d}
   =
   r |\gamma(\boldsymbol{r})|
   (\boldsymbol{\eta}^{p}_2 \cdot \boldsymbol{e}_{\boldsymbol{k}^{p}_1})
   \boldsymbol{e}_{\boldsymbol{\eta}^{p}_1}
   =
   r s \boldsymbol{e}_{\boldsymbol{\eta}^{p}_1}

where

.. math::

   s
   =
   |\gamma(\boldsymbol{r})|
   (\boldsymbol{\eta}^{p}_2 \cdot \boldsymbol{e}_{\boldsymbol{k}^{p}_1}).

Therefore, sheared lattice :math:`\boldsymbol{L}_s` is given as

.. math::

   [\boldsymbol{m}^{s},
    \boldsymbol{\eta}^{s}_1,
    \boldsymbol{\eta}^{s}_2]
   =
   [\boldsymbol{m}^{p},
    \boldsymbol{\eta}^{p}_1,
    \boldsymbol{\eta}^{p}_2 + r s \boldsymbol{e}_{\boldsymbol{\eta}^{p}_1}]
   =
   [\boldsymbol{e}_{\boldsymbol{m}^{p}},
    \boldsymbol{e}_{\boldsymbol{\eta}^{p}_1},
    \boldsymbol{e}_{\boldsymbol{k}^{p}_1}]
   \boldsymbol{L}^s

where

.. math::

   \boldsymbol{L}^s = \boldsymbol{L}^p + r \boldsymbol{S}

where

.. math::

   \boldsymbol{S}
   =
   \begin{pmatrix}
    0 & 0 & 0 \\
    0 & 0 & s \\
    0 & 0 & 0 \\
   \end{pmatrix}.


get_twin_structure
------------------

The operation from parent lattice to twin lattice
:math:`\boldsymbol{W}^{t}` is defineda as

.. math::

   [\boldsymbol{e}_{\boldsymbol{m}^{p}},
    \boldsymbol{e}_{\boldsymbol{\eta}^{p}_1},
    \boldsymbol{e}_{\boldsymbol{k}^{p}_1}]
   \boldsymbol{X}^{p}_{0}
   \longrightarrow
   [\boldsymbol{e}_{\boldsymbol{m}^{p}},
    \boldsymbol{e}_{\boldsymbol{\eta}^{p}_1},
    \boldsymbol{e}_{\boldsymbol{k}^{p}_1}]
   \boldsymbol{X}^{t}_{0}.

where

.. math::

  \boldsymbol{X}^{t}_{0} = \boldsymbol{W}^{t} \boldsymbol{X}^{p}_{0}.

In twin type :math:`\rm{I}`,
rotation matrix :math:`\boldsymbol{W}^{t}`
is given as

.. math::

   \boldsymbol{W}^{t}
   =
   \begin{pmatrix}
    1 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & -1 \\
   \end{pmatrix}.

In twin type :math:`\rm{I\hspace{-1pt}I}`,
rotation matrix :math:`\boldsymbol{W}^t`
is given as

.. math::

   \boldsymbol{W}^{t}
   =
   \begin{pmatrix}
    -1 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & -1 \\
   \end{pmatrix}.

In both type, the following equation hold.

.. math::

   \boldsymbol{W}^{t,-1} = \boldsymbol{W}^{t,T} = \boldsymbol{W}

The relation between parent lattice :math:`\boldsymbol{L}^{p}`
and twin lattice :math:`\boldsymbol{L}^{t}` is

.. math::

   [\boldsymbol{m}^{p},
    \boldsymbol{\eta}^{p}_1,
    \boldsymbol{\eta}^{p}_2]
   =
   [\boldsymbol{e}_{\boldsymbol{m}^{p}},
    \boldsymbol{e}_{\boldsymbol{\eta}^{p}_1},
    \boldsymbol{e}_{\boldsymbol{k}^{p}_1}]
   \boldsymbol{L}^{p}
   \longrightarrow
   [\boldsymbol{e}_{\boldsymbol{m}^{p}},
    \boldsymbol{e}_{\boldsymbol{\eta}^{p}_1},
    \boldsymbol{e}_{\boldsymbol{k}^{p}_1}]
   \boldsymbol{W}^{t}
   \boldsymbol{L}^{p}
   \equiv
   [\boldsymbol{m}^{t},
    \boldsymbol{\eta}^{t}_1,
    \boldsymbol{\eta}^{t}_2]

so

.. math::

   [\boldsymbol{m}^{t},
    \boldsymbol{\eta}^{t}_1,
    \boldsymbol{\eta}^{t}_2]
   =
   [\boldsymbol{e}_{\boldsymbol{m}^{p}},
    \boldsymbol{e}_{\boldsymbol{\eta}^{p}_1},
    \boldsymbol{e}_{\boldsymbol{k}^{p}_1}]
   \boldsymbol{L}^{t}

where

.. math::

   \boldsymbol{L}^{t}
   =
   \boldsymbol{W}^{t}
   \boldsymbol{L}^{p}.

By this transformation, number vectors in the basis of
parent vectors :math:`\boldsymbol{X}^{p}` (including
:math:`\boldsymbol{X}^{p}_{lat}` and :math:`\boldsymbol{X}^{p}_{atm}`)
are not affected.

.. math::

   [\boldsymbol{m}^{p},
    \boldsymbol{\eta}^{p}_1,
    \boldsymbol{\eta}^{p}_2]
   \boldsymbol{X}^{p}
   &=
   [\boldsymbol{e}_{\boldsymbol{m}^{p}},
    \boldsymbol{e}_{\boldsymbol{\eta}^{p}_1},
    \boldsymbol{e}_{\boldsymbol{k}^{p}_1}]
   \boldsymbol{L}^{p}
   \boldsymbol{X}^{p} \\
   &\longrightarrow
   [\boldsymbol{e}_{\boldsymbol{m}^{p}},
    \boldsymbol{e}_{\boldsymbol{\eta}^{p}_1},
    \boldsymbol{e}_{\boldsymbol{k}^{p}_1}]
   \boldsymbol{W}^{t}
   \boldsymbol{L}^{p}
   \boldsymbol{X}^{p} \\
   &=
   [\boldsymbol{m}^{t},
    \boldsymbol{\eta}^{t}_1,
    \boldsymbol{\eta}^{t}_2]
   \boldsymbol{L}^{p,-1}
   \boldsymbol{W}^{t,-1}
   \boldsymbol{W}^{t}
   \boldsymbol{L}^{p}
   \boldsymbol{X}^{p} \\
   &=
   [\boldsymbol{m}^{t},
    \boldsymbol{\eta}^{t}_1,
    \boldsymbol{\eta}^{t}_2]
   \boldsymbol{X}^{p}
   \equiv
   [\boldsymbol{m}^{t},
    \boldsymbol{\eta}^{t}_1,
    \boldsymbol{\eta}^{t}_2]
   \boldsymbol{X}^{t}

so number vectors in the basis of
twin vectors :math:`\boldsymbol{X}^{t}` (including
:math:`\boldsymbol{X}^{t}_{lat}` and :math:`\boldsymbol{X}^{t}_{atm}`)

.. math::

   \boldsymbol{X}^{t} = \boldsymbol{X}^{p}

HexagonalTwinBoundary
=====================

__init__
--------

To create 'HexagonalTwinBoundary' object, you have to specify
the norm of a and c axis and its specie as a hexagonal metal
information. Moreover, twinmode, twintype, dimension and
x- y- shift respectively. If you set dimension equal '[1,1,2]'
and x-shift equal '1/2', then parent and twin structures with
its supecell as [1,1,2] and fix all the parent lattice point to
-1/4 from its original points and fix all the twin lattice point to
1/4 from its original points. Then, dichromatic lattice
:math:`\boldsymbol{L}^{d}` is created.

.. math::

   [\boldsymbol{m}^{d},
    \boldsymbol{\eta}^{d}_1,
    \boldsymbol{k}^{d}_1]
   =
   [\boldsymbol{e}_{\boldsymbol{m}^{p}},
    \boldsymbol{e}_{\boldsymbol{\eta}^{p}_1},
    \boldsymbol{e}_{\boldsymbol{k}^{p}_1}]
   \boldsymbol{L}^{d}

In the case 'dim=[1,1,1]', dichromatic lattice
:math:`\boldsymbol{L}^{d}` becomes

.. math::

   \boldsymbol{L}^{d}
   =
   \boldsymbol{L}^{s}(r=0.5)
   \begin{pmatrix}
    1 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & 2 \\
   \end{pmatrix}.

After this, number vectors :math:`\boldsymbol{X}^{p,t}`
in the bases both parent and twin
are transformed into the dichromatic lattice frame.

.. math::

   \boldsymbol{L}^{p,t} \boldsymbol{X}^{p,t}
   =
   \boldsymbol{L}^{d} \boldsymbol{X}^{d}

so

.. math::
   \boldsymbol{X}^{d}
   =
   \boldsymbol{L}^{d -1} \boldsymbol{L}^{p,t} \boldsymbol{X}^{p,t}


get_sheared_structure
---------------------

The twin boundary structure can be sheared by this function.
Input 'gamma' represents shear strain (:math:`\gamma'`).
dichromatic lattice are sheared as

.. math::

   \boldsymbol{L}^{d, s} = \boldsymbol{L}^{d} + \boldsymbol{S}

where

.. math::

   \boldsymbol{S}
   =
   \begin{pmatrix}
    0 & 0 & 0 \\
    0 & 0 & s' \\
    0 & 0 & 0 \\
   \end{pmatrix}.

where

.. math::

   s'
   =
   \gamma' |\boldsymbol{k}^{d}_1|
