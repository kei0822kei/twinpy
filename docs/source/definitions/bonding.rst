=======
bonding
=======

In this page, some formulations are provided used in bonding.

.. contents::
   :depth: 2
   :local:


get_neighbor
------------

Let's think about getting neighbor atoms from specific atom.
For simplicity, consider in one dimension where lattice parameter
:math:`a`, fractional coordinate of specific atom :math:`x`
and cutoff distance :math:`d`. In this case, we have to get
atoms from :math:`ax-d` to :math:`ax+d`. We define :math:`i` as

.. math::

   i \equiv [\frac{d}{a}]

then

.. math::

   ax+d = a(x+\frac{d}{a}) < a(1+\frac{d}{a}) < a(i+2)

and

.. math::

   ax-d = a(x-\frac{d}{a}) > -a(\frac{d}{a}) > -a(i+1).

Therefore, we have to consider only from :math:`-(i+1)`
to :math:`(i+2)` periodic unitcell.
