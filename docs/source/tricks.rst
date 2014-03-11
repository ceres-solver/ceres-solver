.. _chapter-tricks:

===================
Tips, Tricks & FAQs
===================

A collection of miscellanous tips, tricks and answers to frequently
asked questions.

1. Use analytical/automatic derivatives when possible.

   This is the single most important piece of advice we can give to
   you. It is tempting to take the easy way out and use numeric
   differentiation. This is a bad idea. Numeric differentiation is
   slow, ill-behaved, hard to get right, and results in poor
   convergence behaviour.

   Ceres allows the user to define templated functors which will
   be automatically differentiated. For most situations this is enough
   and we recommend using this facility. In some cases the derivatives
   are simple enough or the performance considerations are such that
   the overhead of automatic differentiation is too much. In such
   cases, analytic derivatives are recommended.

   The use of numerical derivatives should be a measure of last
   resort, where it is simply not possible to write a templated
   implementation of the cost function.

   In many cases where it is not possible to do analytic or automatic
   differentiation of the entire cost function. But it is generally
   the case that it is possible to decompose the cost function into
   parts that need to be numerically differentiated and parts that can
   be automatically or analytically differentiated.

   To this end, Ceres has extensive support for mixing analytic,
   automatic and numeric differentiation. See
   :class:`NumericDiffFunctor` and :class:`CostFunctionToFunctor`.


2. Diagnosing convergence issues.

   TBD

3. Diagnoising performance issues.

   TBD
