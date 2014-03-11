.. _chapter-tricks:

===================
Tips, Tricks & FAQs
===================

A collection of miscellanous tips, tricks and frequently asked
questions

Derivatives
===========

The single most important bit of advice for users of Ceres Solver is
to use analytic/automatic differentiation when you can. It is tempting
to take the easy way out and use numeric differentiation. This is a
bad idea. Numeric differentiation is slow, ill-behaved, hard to get
right and results in poor convergence behaviour.
