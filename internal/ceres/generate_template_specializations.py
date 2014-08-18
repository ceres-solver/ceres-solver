# Ceres Solver - A fast non-linear least squares minimizer
# Copyright 2015 Google Inc. All rights reserved.
# http://ceres-solver.org/
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of Google Inc. nor the names of its contributors may be
#   used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: sameeragarwal@google.com (Sameer Agarwal)
#
# Script for explicitly generating template specialization of the
# SchurEliminator class. It is a rather large class
# and the number of explicit instantiations is also large. Explicitly
# generating these instantiations in separate .cc files breaks the
# compilation into separate compilation unit rather than one large cc
# file which takes 2+GB of RAM to compile.
#
# This script creates two sets of files.
#
# 1. schur_eliminator_x_x_x.cc
# where, the x indicates the template parameters and
#
# 2. schur_eliminator.cc
#
# that contains a factory function for instantiating these classes
# based on runtime parameters.
#
# The list of tuples, specializations indicates the set of
# specializations that is generated.

# Set of template specializations to generate
SPECIALIZATIONS = [(2, 2, 2),
                   (2, 2, 3),
                   (2, 2, 4),
                   (2, 2, "Eigen::Dynamic"),
                   (2, 3, 3),
                   (2, 3, 4),
                   (2, 3, 6),
                   (2, 3, 9),
                   (2, 3, "Eigen::Dynamic"),
                   (2, 4, 3),
                   (2, 4, 4),
                   (2, 4, 8),
                   (2, 4, 9),
                   (2, 4, "Eigen::Dynamic"),
                   (2, "Eigen::Dynamic", "Eigen::Dynamic"),
                   (4, 4, 2),
                   (4, 4, 3),
                   (4, 4, 4),
                   (4, 4, "Eigen::Dynamic")]

import schur_eliminator_template
import partitioned_matrix_view_template

def SuffixForSize(size):
  if size == "Eigen::Dynamic":
    return "d"
  return str(size)

def SpecializationFilename(prefix, row_block_size, e_block_size, f_block_size):
  return "_".join([prefix] + map(SuffixForSize, (row_block_size,
                                                 e_block_size,
                                                 f_block_size)))

def GenerateFactoryConditional(row_block_size, e_block_size, f_block_size):
  conditionals = []
  if (row_block_size != "Eigen::Dynamic"):
    conditionals.append("(options.row_block_size == %s)" % row_block_size)
  if (e_block_size != "Eigen::Dynamic"):
    conditionals.append("(options.e_block_size == %s)" % e_block_size)
  if (f_block_size != "Eigen::Dynamic"):
    conditionals.append("(options.f_block_size == %s)" % f_block_size)
  if (len(conditionals) == 0):
    return "%s"

  if (len(conditionals) == 1):
    return " if " + conditionals[0] + "{\n  %s\n }\n"

  return " if (" + " &&\n  ".join(conditionals) + ") {\n  %s\n }\n"

def Specialize(name, data):
  """
  Generate specialization code and the conditionals to instantiate it.
  """
  f = open(name + ".cc", "w")
  f.write(data["HEADER"])
  f.write(data["FACTORY_FILE_HEADER"])

  for row_block_size, e_block_size, f_block_size in SPECIALIZATIONS:
    output = SpecializationFilename("generated/" + name,
                                    row_block_size,
                                    e_block_size,
                                    f_block_size) + ".cc"
    fptr = open(output, "w")
    fptr.write(data["HEADER"])

    template = data["SPECIALIZATION_FILE"]
    if (row_block_size == "Eigen::Dynamic" and
        e_block_size == "Eigen::Dynamic" and
        f_block_size == "Eigen::Dynamic"):
      template = data["DYNAMIC_FILE"]

    fptr.write(template % (row_block_size, e_block_size, f_block_size))
    fptr.close()

    FACTORY_CONDITIONAL = GenerateFactoryConditional(row_block_size, e_block_size, f_block_size)
    f.write(FACTORY_CONDITIONAL % data["FACTORY"] % (row_block_size, e_block_size, f_block_size));
    #f.write(FACTORY_CONDITIONAL % (row_block_size, row_block_size,
    #                               e_block_size, e_block_size,
    #                               f_block_size, f_block_size,
    #                               data["FACTORY"] % (row_block_size, e_block_size, f_block_size)));

  f.write(data["FACTORY_FOOTER"])
  f.close()



QUERY_HEADER = """// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2017 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: sameeragarwal@google.com (Sameer Agarwal)
//
// What template specializations are available.
//
// ========================================
// THIS FILE IS AUTOGENERATED. DO NOT EDIT.
// THIS FILE IS AUTOGENERATED. DO NOT EDIT.
// THIS FILE IS AUTOGENERATED. DO NOT EDIT.
// THIS FILE IS AUTOGENERATED. DO NOT EDIT.
//=========================================
//
// This file is generated using generate_template_specializations.py.
"""

QUERY_FILE_HEADER = """
#include "ceres/internal/eigen.h"
#include "ceres/schur_templates.h"

namespace ceres {
namespace internal {

void GetBestSchurTemplateSpecialization(int* row_block_size,
                                        int* e_block_size,
                                        int* f_block_size) {
  LinearSolver::Options options;
  options.row_block_size = *row_block_size;
  options.e_block_size = *e_block_size;
  options.f_block_size = *f_block_size;
  *row_block_size = Eigen::Dynamic;
  *e_block_size = Eigen::Dynamic;
  *f_block_size = Eigen::Dynamic;
#ifndef CERES_RESTRICT_SCHUR_SPECIALIZATION
"""

QUERY_FOOTER = """
#endif
  return;
}

}  // namespace internal
}  // namespace ceres
"""

QUERY_ACTION = """*row_block_size = %s;
  *e_block_size = %s;
  *f_block_size = %s;
  return;"""

def GenerateQueryFile():
  """
  Generate specialization code and the conditionals to instantiate it.
  """
  f = open("schur_templates.cc", "w")
  f.write(QUERY_HEADER)
  f.write(QUERY_FILE_HEADER)

  for row_block_size, e_block_size, f_block_size in SPECIALIZATIONS:
    FACTORY_CONDITIONAL = GenerateFactoryConditional(row_block_size, e_block_size, f_block_size)
    f.write(FACTORY_CONDITIONAL % QUERY_ACTION % (row_block_size, e_block_size, f_block_size));

  f.write(QUERY_FOOTER)
  f.close()


if __name__ == "__main__":
  Specialize("schur_eliminator", schur_eliminator_template.__dict__)
  Specialize("partitioned_matrix_view", partitioned_matrix_view_template.__dict__)
  GenerateQueryFile()
