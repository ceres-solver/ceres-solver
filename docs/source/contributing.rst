.. _chapter-contributing:

=============
Contributions
=============


We welcome contributions to Ceres, whether they are new features, bug
fixes or tests. The Ceres mailing list [#f1]_ is the best place for
all development related discussions. Please consider joining it. If
you have ideas on how you would like to contribute to Ceres, it is a
good idea to let us know on the mailing list before you start
development. We may have suggestions that will save effort when trying
to merge your work into the main branch. If you are looking for ideas,
please let us know about your interest and skills and we will be happy
to make a suggestion or three.

We follow Google's C++ Style Guide [#f2]_ and use ``git`` for version
control. We use the Gerrit code review system to collaborate and
review changes to Ceres. Gerrit enables pre-commit reviews so that
Ceres can maintain a linear history with clean, reviewed commits, and
no merges. The overall development flow is as follows:

  * Sign up for the mailing list.
  * Download and configure Git.
  * Sign up for the Ceres Gerrit instance and sign a CLA.
  * Clone the repository; build.
  * Make a single commit or amend a commit if a re-review.
  * Push the commit to Gerrit for review.
  * Address review comments after they come back.
  * Amend the patch and re-upload. Repeat until merged by the reviewer.


Detailed Steps
==============

  0. Sign up for the `mailing
     list <https://groups.google.com/group/ceres-solver>`_ (optional
     but suggested).

  1.  Download and configure Git.

    - Mac.

       .. code-block:: bash

          brew install git

    - Linux.

       .. code-block:: bash

          sudo apt-get install git

    - Windows. Download
       `msysgit <https://code.google.com/p/msysgit/>`_, which includes
       a minimal `Cygwin <http://www.cygwin.com/>`_ install.

  2. Sign up for `Gerrit
     <https://ceres-solver-review.googlesource.com/>`. You will also
     need to sign the Contributor License Agreement (CLA) with Google,
     which gives Google a royalty-free unlimited license to use your
     contributions. You retain copyright.

  3. Clone the Ceres Solver git repository from Gerrit.

     .. code-block:: bash

        git clone https://ceres-solver.googlesource.com/ceres-solver


  4. Build Ceres, following the instructions in
     :ref:`chapter-building`.

     On Mac and Linux, the ``CMake`` build will download and enable
     the Gerrit pre-commit hook automatically. This pre-submit hook
     creates `Change-Id: ...` lines in your commits.

     If this does not work or you are on Windows, execute the
     following in the root directory of the local Git repository:

     .. code-block:: bash

        curl -o .git/hooks/commit-msg https://ceres-solver-review.googlesource.com/tools/hooks/commit-msg
        chmod +x .git/hooks/commit-msg

    5. Configure your Gerrit password with a ``.netrc`` (Mac and
    Linux) or ``_netrc`` (Windows) which allows pushing to Gerrit
    without having to enter a very long random password every time:

      - Go to `http://ceres-solver-review.googlesource.com
         <http://ceres-solver-review.googlesource.com>`_

      - Sign in

      - Click ``Settings`` link in the top right.

      - Click the ``HTTP Password`` link on the left.

      - Click the ``Obtain password`` link

      - (maybe) Select an account for multi-login. This should be the same as your Gerrit login.

      - Click ``Allow access`` when the page requests access to your git repositories.

      - Copy the contents of the netrc listed into the clipboard. On
         Mac and Linux, paste the contents into ``~/.netrc``. On
         Windows, by default users do not have a ``%HOME%`` setting. Run
         the following in a command terminal:

         .. code-block:: bash

            setx HOME %USERPROFILE%


         This will set the ``%HOME%`` environment variable persistently,
         and is used by git to find ``%HOME%\_netrc``. Then, create a
         new text file named ``_netr`c` and put it in
         e.g. ``C:\Users\username`` where ``username`` is your
         user name.

    6. Hack away on Ceres! Take a peek at our bug tracker, or look for
       some TODO's in the code. Before embarking on major work, please
       email the list with your idea and plans, so we can avoid
       accidental duplicated work.

    7. Make your changes against master or whatever branch you
       like. Commit your changes, preferably as one patch. When you
       commit, the Gerrit hook will add a `Change-Id:` line as the
       last line of the commit.

    8. Push your changes to the Ceres Gerrit instance:

       .. code-block:: bash

          git push origin HEAD:refs/for/master

       When the push succeeds, the console will display a URL showing
       the address of the review. Go to the URL and add reviewers;
       typically this is Sameer or Keir at this point.

    9. Wait for a review.

   10. Responding to review consists of three steps

       - Update the patch and push it to Gerrit.
       - Respond to each comment in Gerrit.
       - Publish the responses.

       In more detail:

       Once review comments come in, address them. Please reply to
       each comment in Gerrit, which makes the re-review process
       easier. After modifying the code in your git instance, *don't
       make a new commit*. Instead, update the last commit using a
       command like the following:

       .. code-block:: bash

          git commit --amend -a

      This will update the last commit, so that it has both the
      original patch and your updates as a single commit. You will
      have a chance to edit the commit message as well.

      Push the new commit to Gerrit as before:

      .. code-block:: bash

         git push origin HEAD:refs/for/master


      Gerrit will use the ``Change-Id:`` to match the previous commit
      with the new one. The review interface retains your original
      patch, but also shows the new patch.

      Publish your responses to the comments, and wait for a new round
      of reviews.


.. rubric:: Footnotes

.. [#f1] http://groups.google.com/group/ceres-solver
.. [#f2] http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml
