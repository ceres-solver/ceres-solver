.. _chapter-about:

=====
About
=====

Ceres Solver grew out of the need for general least squares solving at Google.
Around 2010, Sameer Agarwal and Keir Mierle decided to replace a custom bundle
adjuster at Google (known as BlockBundler) with something more modern. After
two years of on-and-off development, Ceres Solver was released as open source
in May of 2012.

Acknowledgements
----------------

A number of people have helped with the development and open sourcing
of Ceres.

Fredrik Schaffalitzky when he was at Google started the development of
Ceres, and even though much has changed since then, many of the ideas
from his original design are still present in the current code.

Amongst Ceres' users at Google two deserve special mention: William
Rucklidge and James Roseborough. William was the first user of
Ceres. He bravely took on the task of porting production code to an
as-yet unproven optimization library, reporting bugs and helping fix
them along the way. James is perhaps the most sophisticated user of
Ceres at Google. He has reported and fixed bugs and helped evolve the
API for the better.

Since the initial release of Ceres, a number of people have
contributed to Ceres by porting it to new platforms, reporting bugs,
fixing bugs and adding new functionality. We acknowledge all of these
contributions in the :ref:`chapter-version-history`.

Origin of the name Ceres Solver
-------------------------------
While there is some debate as to who invented the method of Least Squares
[Stigler]_, there is no debate that it was `Carl Friedrich Gauss
<http://en.wikipedia.org/wiki/Carl_Friedrich_Gauss>`_ who brought it to the
attention of the world. Using just 22 observations of the newly discovered
asteroid `Ceres <http://en.wikipedia.org/wiki/Ceres_(dwarf_planet)>`_, Gauss
used the method of least squares to correctly predict when and where the
asteroid will emerge from behind the Sun [TenenbaumDirector]_. We named our
solver after Ceres to celebrate this seminal event in the history of astronomy,
statistics and optimization.

