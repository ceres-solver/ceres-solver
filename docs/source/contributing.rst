.. _chapter-contributing:

============
Contributing
============

We welcome contributions to Ceres, whether they are new features, bug
fixes, or tests. The `Ceres mailing list
<https://groups.google.com/g/ceres-solver>`_ is the best place for
development discussions. If you plan to contribute a new feature or
significant change, please post your proposal on the mailing list
before starting development; early feedback often saves effort during
code review. If you are looking for a project, tell us about your
interests and background on the mailing list and we will be glad to
suggest open problems.

We follow Google's `C++ Style Guide
<https://google.github.io/styleguide/cppguide.html>`_ and use `git
<https://git-scm.com/>`_ for version control. Code review takes place
on `Gerrit <https://ceres-solver-review.googlesource.com/>`_, which
enables pre-commit review so that Ceres maintains a linear history of
clean, reviewed commits without merge commits.

The sections below describe how to set up your development environment
and submit a change list for review on Gerrit.

Setting up Your Environment
===========================

1. Download and configure ``git``:

   * macOS: ``brew install git``
   * Linux: ``sudo apt-get install git``
   * Windows: Download `Git for Windows
     <https://git-scm.com/download/win>`__.

2. Sign up for `Gerrit
   <https://ceres-solver-review.googlesource.com/>`_ and `sign the
   Google Contributor License Agreement (CLA)
   <https://opensource.google/documentation/reference/cla>`_, which
   grants Google a royalty-free, unlimited license to use your
   contributions while you retain copyright.

3. Clone the Ceres Solver ``git`` repository from Gerrit:

   .. code-block:: bash

      git clone https://ceres-solver.googlesource.com/ceres-solver


4. Build Ceres following the instructions in
   :ref:`chapter-installation`.

   On macOS and Linux, the ``CMake`` build automatically downloads and
   installs the Gerrit ``commit-msg`` hook, which appends a
   ``Change-Id: ...`` line to each commit message.

   If the hook is not installed automatically, or if you are on
   Windows, run the following commands in the root directory of your
   local ``git`` repository:

   .. code-block:: bash

      curl -o .git/hooks/commit-msg https://ceres-solver-review.googlesource.com/tools/hooks/commit-msg
      chmod +x .git/hooks/commit-msg

5. Configure a ``.gitcookies`` file so that you can push to Gerrit
   without entering a generated password each time:

   * Sign in to `https://ceres-solver-review.googlesource.com
     <https://ceres-solver-review.googlesource.com>`_.

   * Click ``Settings -> HTTP Credentials -> Obtain Password``.

   * If prompted by multi-login, select the account associated with
     your Gerrit login.

   * Click ``Allow access`` when prompted to grant access to your
     ``git`` repositories.

   * Follow the instructions on the page to create a ``.gitcookies``
     file in ``$HOME/.gitcookies`` (macOS and Linux) or
     ``%USERPROFILE%\.gitcookies`` (Windows). On Windows, use a recent
     release of `Git for Windows <https://git-scm.com/download/win>`_
     so that ``%USERPROFILE%\.gitcookies`` is detected automatically.

6. Install ``clang-format``:

   * macOS: ``brew install clang-format``
   * Linux: ``sudo apt-get install clang-format``
   * Windows: Install ``clang-format`` via Visual Studio, LLVM, or
     `npm <https://superuser.com/a/1505297/1141693>`_.

   Format all modified source files before committing by running
   ``clang-format -i FILENAME``, executing ``./scripts/format_all.sh``,
   or configuring your editor to run ``clang-format`` on save.

Submitting a Change
===================

1. Make your changes against ``master`` (or a local branch) and format
   them with ``clang-format``. Commit your changes as a single logical
   patch; the Gerrit hook will append a ``Change-Id:`` footer to the
   commit message.

   Format your commit message in the standard `50/72 style
   <https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html>`_.

2. Push your commit to Gerrit for review:

   .. code-block:: bash

      git push origin HEAD:refs/for/master

   When the push succeeds, ``git`` prints the URL of the Gerrit review.
   Open that URL and add at least one maintainer (Sameer Agarwal, Keir
   Mierle, Alex Stewart, William Rucklidge, or Sergiu Deitsch) as a
   reviewer.

3. Wait for review feedback.

4. Address any review comments and reply to each comment thread in
   Gerrit. Instead of creating a second commit, amend your existing
   commit so that the ``Change-Id:`` line is preserved:

   .. code-block:: bash

      git commit --amend -a

   Push the amended commit to ``HEAD:refs/for/master`` again. Gerrit
   uses the ``Change-Id:`` footer to attach the new patch set to the
   existing review while preserving earlier patch sets for comparison.
   Publish your replies on Gerrit to notify the reviewers.
