Armstrong Sphinx Theme
======================
Sphinx theme for Armstrong documentation


Usage
-----
Symlink this repository into your documentation at ``docs/_themes/armstrong``
then add the following two settings to your Sphinx ``conf.py`` file::

    html_theme = "armstrong"
    html_theme_path = ["_themes", ]

You can also change colors and such by adjusting the ``html_theme_options``
dictionary.  For a list of all settings, see ``theme.conf``.


Defaults
--------
This repository has been customized for Armstrong documentation, but you can
use the original default color scheme on your project by copying the
``rtd-theme.conf`` over the existing ``theme.conf``.


Contributing
------------

* Create something awesome -- make the code better, add some functionality,
  whatever (this is the hardest part).
* `Fork it`_
* Create a topic branch to house your changes
* Get all of your commits in the new topic branch
* Submit a `pull request`_

.. _Fork it: http://help.github.com/forking/
.. _pull request: http://help.github.com/pull-requests/


State of Project
----------------
Armstrong is an open-source news platform that is freely available to any
organization.  It is the result of a collaboration between the `Texas Tribune`_
and `Bay Citizen`_, and a grant from the `John S. and James L. Knight
Foundation`_.  The first stable release is scheduled for September, 2011.

To follow development, be sure to join the `Google Group`_.

``armstrong_sphinx`` is part of the `Armstrong`_ project.  Unless you're
looking for a Sphinx theme, you're probably looking for the main project.

.. _Armstrong: http://www.armstrongcms.org/
.. _Bay Citizen: http://www.baycitizen.org/
.. _John S. and James L. Knight Foundation: http://www.knightfoundation.org/
.. _Texas Tribune: http://www.texastribune.org/
.. _Google Group: http://groups.google.com/group/armstrongcms


Credit
------
This theme is based on the the excellent `Read the Docs`_ theme.  The original
can be found in the `readthedocs.org`_ repository on GitHub.

.. _Read the Docs: http://readthedocs.org/
.. _readthedocs.org: https://github.com/rtfd/readthedocs.org


License
-------
Like the original RTD code, this code is licensed under a BSD.  See the
associated ``LICENSE`` file for more information.
