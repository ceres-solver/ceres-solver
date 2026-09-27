<style>
col {
  width: auto !important;
}
table:has(th:nth-child(4):last-child) :is(td, th):first-child {
  width: 16em;
}
table:has(th:nth-child(4):last-child) :is(td, th) + :is(td, th) {
  width: 8em;
}
.failed {
  color: red;
}
</style>

# oneMKL Integration Benchmark

This report compares the sparse linear algebra libraries of Ceres Solver on
the bundle adjustment problems of the
[BAL datasets](https://grail.cs.washington.edu/projects/bal/). It evaluates the
oneMKL integration of the `mkl-pardiso` branch, which adds `MKL_SPARSE`, against
`SUITE_SPARSE` and `EIGEN_SPARSE`. All runs took place on 2026-10-07 between
10:17 and 17:18.

## Machine

| | |
|:--|:--|
| CPU | AMD Ryzen 9 5950X 16-Core Processor |
| Cores | 16 per socket, 1 socket, 32 hardware threads |
| Maximum CPU clock | 5.09 GHz |
| L3 cache | 64 MiB (2 instances) |
| Memory | 125.7 GiB usable |
| Operating system | Arch Linux |
| Linux kernel | 7.2.9-arch1-1 |
| CPU frequency scaling | `amd-pstate-epp` driver, `powersave` governor, `balance_performance` energy preference, boost enabled |

## Software

| | |
|:--|:--|
| Benchmark binary | `/home/sergiu/Projects/ceres-solver-dev/build/bin/bundle_adjuster` |
| Ceres Solver | 2.3.0 |
| Eigen | 3.5.0 |
| oneMKL | 2026.0.0 |
| SuiteSparse | 7.12.3 |
| METIS | 5.2.1 |
| CUDA | 13040 |
| Toolchain | GCC 16.2.1 20260810, mold 3.0.0 |
| Ceres sources when described | `2.2.0-183-g69e87d91` on branch `mkl-pardiso` |
| Ceres compiler flags | `-O3 -DNDEBUG -flto=auto -fuse-ld=mold` |
| `CMAKE_BUILD_TYPE` | `Release` |
| `BUILD_SHARED_LIBS` | `OFF` |
| `MKL_INTERFACE_FULL` | `intel_lp64` |
| `MKL_THREADING` | `intel_thread` |
| `MKL_LINK` | `dynamic` |

### Shared Libraries

The BLAS, LAPACK and OpenMP libraries that `bundle_adjuster` loads, according to `ldd`:

| | |
|:--|:--|
| `libcholmod.so.5` | `/home/sergiu/Projects/SuiteSparse/build/libcholmod.so.5` |
| `libmkl_intel_lp64.so.3` | `/opt/intel/oneapi/mkl/latest/lib/libmkl_intel_lp64.so.3` |
| `libmkl_intel_thread.so.3` | `/opt/intel/oneapi/mkl/latest/lib/libmkl_intel_thread.so.3` |
| `libmkl_core.so.3` | `/opt/intel/oneapi/mkl/latest/lib/libmkl_core.so.3` |
| `libiomp5.so` | `/opt/intel/oneapi/compiler/2026.0/lib/libiomp5.so` |

The corresponding libraries of CHOLMOD:

| | |
|:--|:--|
| `libmkl_intel_lp64.so.3` | `/opt/intel/oneapi/mkl/latest/lib/libmkl_intel_lp64.so.3` |
| `libmkl_intel_thread.so.3` | `/opt/intel/oneapi/mkl/latest/lib/libmkl_intel_thread.so.3` |
| `libmkl_core.so.3` | `/opt/intel/oneapi/mkl/latest/lib/libmkl_core.so.3` |
| `libiomp5.so` | `/opt/intel/oneapi/compiler/latest/lib/libiomp5.so` |

**For a fair comparison, SuiteSparse was compiled against the same oneMKL.**
The shared libraries above show that CHOLMOD and Ceres load the same oneMKL
interface, threading layer and OpenMP runtime. SuiteSparse and PARDISO
therefore use the same BLAS and LAPACK implementation and the same threading
runtime, so that the timings compare the sparse factorizations rather than the
dense kernels underneath them.

## Benchmark Configuration

Each problem was solved once with the `bundle_adjuster` example of Ceres,
changing only `--sparse_linear_algebra_library` between runs. The remaining
settings are the defaults of the example, as recorded in the solver summaries:

| | |
|:--|:--|
| Linear solver | `SPARSE_SCHUR` |
| Fill reducing ordering | `AMD` |
| Minimizer | Trust region with Levenberg-Marquardt |
| Iterations | at most 5 |
| Threads | 32 |
| Memory limit per run | 110 GB, set by `systemd-run -p MemoryMax=110G` |
| Logging | `--v=3`, whose overhead is part of every timing |

The reported duration is the `Total` time of the Ceres solver summary in
seconds. Every problem ran once per library, so the variance between repeated
runs was not measured.

## Reproduction

`bundle_adjuster` must be in the `PATH`. The evaluation script appends one line
per problem and library to `evaluation/results.csv`, skips problems already
listed there, and keeps the log of every run in `evaluation/<library>/`. When
an evaluation starts, it also records the machine and the software in
`evaluation/environment.md` using `environment.py`. The report script then
writes `report.html` and this report to `results.md`, taking the text that does
not depend on the results from `report_header.md`:

```sh
PATH=~/Projects/ceres-solver-dev/build/bin:$PATH \
  ./evaluate.sh grail.cs.washington.edu/projects/bal/data
python3 report.py
```

## Summary

Speedups are geometric means of the ratio of total solve times. Problems that failed with a library are excluded from the speedup over that library. The tables of the datasets show the shortest duration of every problem in bold.

| Dataset   |   Problems |   Fastest: oneMKL |   Fastest: SuiteSparse |   Fastest: Eigen |   oneMKL speedup over SuiteSparse |   oneMKL speedup over Eigen |
|:----------|-----------:|------------------:|-----------------------:|-----------------:|----------------------------------:|----------------------------:|
| dubrovnik |         16 |                15 |                      0 |                1 |                              1.12 |                        1.67 |
| final     |          8 |                 2 |                      5 |                0 |                              0.99 |                       14.09 |
| ladybug   |         31 |                29 |                      1 |                1 |                              1.22 |                       13.73 |
| trafalgar |         14 |                11 |                      0 |                3 |                              1.06 |                        1.18 |
| venice    |         29 |                28 |                      0 |                1 |                              1.79 |                       11.15 |
| **all**   |         98 |                85 |                      6 |                6 |                              1.30 |                        6.41 |

## Durations

### Dubrovnik

| Problem                   |                              Eigen [s] |                             oneMKL [s] |                        SuiteSparse [s] |
| :------------------------ | -------------------------------------: | -------------------------------------: | -------------------------------------: |
| problem-16-22106-pre      |                               **0.23** |                                   0.31 |                                   0.26 |
| problem-88-64298-pre      |                                   1.15 |                               **1.05** |                                   1.12 |
| problem-135-90642-pre     |                                   1.98 |                               **1.63** |                                   1.70 |
| problem-142-93602-pre     |                                   2.05 |                               **1.63** |                                   1.77 |
| problem-150-95821-pre     |                                   2.08 |                               **1.60** |                                   1.74 |
| problem-161-103832-pre    |                                   2.15 |                               **1.69** |                                   1.80 |
| problem-173-111908-pre    |                                   2.48 |                               **1.80** |                                   1.97 |
| problem-182-116770-pre    |                                   2.65 |                               **1.80** |                                   2.00 |
| problem-202-132796-pre    |                                   3.35 |                               **2.09** |                                   2.31 |
| problem-237-154414-pre    |                                   4.57 |                               **2.33** |                                   2.72 |
| problem-253-163691-pre    |                                   4.99 |                               **2.42** |                                   2.77 |
| problem-262-169354-pre    |                                   5.34 |                               **2.49** |                                   2.93 |
| problem-273-176305-pre    |                                   5.71 |                               **2.57** |                                   3.03 |
| problem-287-182023-pre    |                                   7.36 |                               **2.60** |                                   3.16 |
| problem-308-195089-pre    |                                   7.60 |                               **2.80** |                                   3.50 |
| problem-356-226730-pre    |                                  12.19 |                               **3.33** |                                   4.41 |

### Final

| Problem                   |                              Eigen [s] |                             oneMKL [s] |                        SuiteSparse [s] |
| :------------------------ | -------------------------------------: | -------------------------------------: | -------------------------------------: |
| problem-93-61203-pre      |                                   1.10 |                                   0.90 |                               **0.90** |
| problem-394-100368-pre    |                                  22.23 |                               **2.36** |                                   2.71 |
| problem-871-527480-pre    |                                 148.02 |                              **10.97** |                                  16.30 |
| problem-961-187103-pre    |                                 330.08 |                                  32.12 |                              **30.04** |
| problem-1936-649673-pre   |                                2375.91 |                                 150.81 |                             **125.82** |
| problem-3068-310854-pre   |                                3654.64 |                                  40.26 |                              **37.09** |
| problem-4585-1324582-pre  |                                8911.16 |                                 185.20 |                             **139.70** |
| problem-13682-4456117-pre | <strong class="failed">failed</strong> | <strong class="failed">failed</strong> | <strong class="failed">failed</strong> |

### Ladybug

| Problem                   |                              Eigen [s] |                             oneMKL [s] |                        SuiteSparse [s] |
| :------------------------ | -------------------------------------: | -------------------------------------: | -------------------------------------: |
| problem-49-7776-pre       |                               **0.11** |                                   0.12 |                                   0.12 |
| problem-73-11032-pre      |                                   0.19 |                                   0.17 |                               **0.16** |
| problem-138-19878-pre     |                                   0.58 |                               **0.26** |                                   0.27 |
| problem-318-41628-pre     |                                   3.91 |                               **0.64** |                                   0.66 |
| problem-372-47423-pre     |                                   6.19 |                               **0.72** |                                   0.77 |
| problem-412-52215-pre     |                                   8.21 |                               **0.77** |                                   0.81 |
| problem-460-56811-pre     |                                   9.47 |                               **0.95** |                                   1.03 |
| problem-539-65220-pre     |                                  12.28 |                               **0.99** |                                   1.08 |
| problem-598-69218-pre     |                                  15.10 |                               **1.05** |                                   1.23 |
| problem-646-73584-pre     |                                  18.71 |                               **1.17** |                                   1.41 |
| problem-707-78455-pre     |                                  20.31 |                               **1.30** |                                   1.52 |
| problem-783-84444-pre     |                                  26.30 |                               **1.49** |                                   1.80 |
| problem-810-88814-pre     |                                  25.17 |                               **1.44** |                                   1.78 |
| problem-856-93344-pre     |                                  25.09 |                               **1.66** |                                   1.96 |
| problem-885-97473-pre     |                                  25.86 |                               **1.72** |                                   1.98 |
| problem-931-102699-pre    |                                  25.39 |                               **1.76** |                                   2.16 |
| problem-969-105826-pre    |                                  27.37 |                               **1.81** |                                   2.24 |
| problem-1031-110968-pre   |                                  28.99 |                               **1.93** |                                   2.30 |
| problem-1064-113655-pre   |                                  31.61 |                               **1.78** |                                   2.25 |
| problem-1118-118384-pre   |                                  32.44 |                               **1.82** |                                   2.40 |
| problem-1152-122269-pre   |                                  42.50 |                               **2.04** |                                   2.73 |
| problem-1197-126327-pre   |                                  35.17 |                               **2.11** |                                   2.66 |
| problem-1235-129634-pre   |                                  39.56 |                               **2.04** |                                   2.57 |
| problem-1266-132593-pre   |                                  44.76 |                               **2.30** |                                   2.85 |
| problem-1340-137079-pre   |                                  61.46 |                               **2.58** |                                   3.29 |
| problem-1469-145199-pre   |                                  69.62 |                               **2.81** |                                   4.36 |
| problem-1514-147317-pre   |                                  79.78 |                               **2.76** |                                   4.14 |
| problem-1587-150845-pre   |                                 104.37 |                               **2.89** |                                   4.23 |
| problem-1642-153820-pre   |                                 111.79 |                               **3.04** |                                   4.49 |
| problem-1695-155710-pre   |                                 119.97 |                               **3.13** |                                   4.33 |
| problem-1723-156502-pre   |                                 120.25 |                               **3.09** |                                   4.52 |

### Trafalgar

| Problem                   |                              Eigen [s] |                             oneMKL [s] |                        SuiteSparse [s] |
| :------------------------ | -------------------------------------: | -------------------------------------: | -------------------------------------: |
| problem-21-11315-pre      |                               **0.10** |                                   0.15 |                                   0.13 |
| problem-39-18060-pre      |                               **0.17** |                                   0.22 |                                   0.20 |
| problem-50-20431-pre      |                               **0.20** |                                   0.22 |                                   0.22 |
| problem-126-40037-pre     |                                   0.43 |                               **0.39** |                                   0.41 |
| problem-138-44033-pre     |                                   0.49 |                               **0.45** |                                   0.48 |
| problem-161-48126-pre     |                                   0.62 |                               **0.47** |                                   0.53 |
| problem-170-49267-pre     |                                   0.61 |                               **0.51** |                                   0.55 |
| problem-174-50489-pre     |                                   0.62 |                               **0.52** |                                   0.56 |
| problem-193-53101-pre     |                                   0.69 |                               **0.53** |                                   0.60 |
| problem-201-54427-pre     |                                   0.78 |                               **0.56** |                                   0.63 |
| problem-206-54562-pre     |                                   0.82 |                               **0.57** |                                   0.64 |
| problem-215-55910-pre     |                                   0.91 |                               **0.58** |                                   0.66 |
| problem-225-57665-pre     |                                   0.90 |                               **0.60** |                                   0.67 |
| problem-257-65132-pre     |                                   1.07 |                               **0.62** |                                   0.74 |

### Venice

| Problem                   |                              Eigen [s] |                             oneMKL [s] |                        SuiteSparse [s] |
| :------------------------ | -------------------------------------: | -------------------------------------: | -------------------------------------: |
| problem-52-64053-pre      |                               **0.90** |                                   0.92 |                                   1.00 |
| problem-89-110973-pre     |                                   1.65 |                               **1.51** |                                   1.66 |
| problem-245-198739-pre    |                                   5.17 |                               **2.67** |                                   3.38 |
| problem-427-310384-pre    |                                  21.51 |                               **4.04** |                                   6.74 |
| problem-744-543562-pre    |                                  98.87 |                               **9.20** |                                  15.75 |
| problem-951-708276-pre    |                                 166.60 |                              **10.81** |                                  20.20 |
| problem-1102-780462-pre   |                                 171.62 |                              **12.03** |                                  22.35 |
| problem-1158-802917-pre   |                                 190.62 |                              **11.86** |                                  23.04 |
| problem-1184-816583-pre   |                                 191.79 |                              **11.42** |                                  22.54 |
| problem-1238-843534-pre   |                                 189.22 |                              **11.96** |                                  22.85 |
| problem-1288-866452-pre   |                                 183.75 |                              **12.74** |                                  23.66 |
| problem-1350-894716-pre   |                                 188.27 |                              **12.79** |                                  24.42 |
| problem-1408-912229-pre   |                                 191.03 |                              **13.32** |                                  25.46 |
| problem-1425-916895-pre   |                                 194.34 |                              **13.38** |                                  25.57 |
| problem-1473-930345-pre   |                                 203.50 |                              **13.48** |                                  25.81 |
| problem-1490-935273-pre   |                                 201.90 |                              **14.02** |                                  26.44 |
| problem-1521-939551-pre   |                                 239.58 |                              **13.89** |                                  26.25 |
| problem-1544-942409-pre   |                                 251.63 |                              **14.55** |                                  27.23 |
| problem-1638-976803-pre   |                                 222.98 |                              **14.06** |                                  27.04 |
| problem-1666-983911-pre   |                                 219.29 |                              **13.77** |                                  26.94 |
| problem-1672-986962-pre   |                                 221.40 |                              **14.42** |                                  27.35 |
| problem-1681-983415-pre   |                                 220.60 |                              **14.33** |                                  27.43 |
| problem-1682-983268-pre   |                                 221.02 |                              **14.35** |                                  27.35 |
| problem-1684-983269-pre   |                                 221.17 |                              **14.66** |                                  27.73 |
| problem-1695-984689-pre   |                                 219.80 |                              **14.06** |                                  27.18 |
| problem-1696-984816-pre   |                                 219.61 |                              **14.56** |                                  27.85 |
| problem-1706-985529-pre   |                                 201.28 |                              **14.85** |                                  28.10 |
| problem-1776-993909-pre   |                                 195.51 |                              **15.17** |                                  28.61 |
| problem-1778-993923-pre   |                                 194.40 |                              **15.28** |                                  28.57 |

## Failed Runs

The following runs recorded no duration, with the reason given in their logs in `evaluation/<library>/<dataset>/`.

- Eigen on `final/problem-13682-4456117-pre`: the log ends without a solver summary
- oneMKL on `final/problem-13682-4456117-pre`: `PARDISO symbolic analysis reports -1763537418 nonzeros in the Cholesky factor, expected a value in [0, 2147483647]. Configure Ceres with MKL_INTERFACE_FULL=intel_ilp64 to use 64-bit integers.`
- SuiteSparse on `final/problem-13682-4456117-pre`: `cholmod_analyze failed. error code: -3`

## Scripts

### `evaluate.sh`

```bash
#!/bin/bash

set -eou pipefail

if [[ $# -eq 0 ]]; then
    echo error: no dataset directory specified >&2
    exit 1
fi

if [[ ! $(command -v bundle_adjuster) ]]; then
    echo error: bundle_adjuster must be in the PATH >&2
    exit 1
fi

dry_run=false

out=./evaluation
out_csv="${out}/results.csv"

if ! $dry_run; then
    mkdir -pv "${out}"
#else
    #out_csv=/dev/stdout
fi

if [[ ! -s ${out_csv} && -d ${out} ]]; then
    echo dataset,problem,library,duration > "${out_csv}"
fi

# Describe the machine and the software once when the evaluation starts.
out_environment="${out}/environment.md"
if ! $dry_run && [[ ! -s ${out_environment} ]]; then
    python3 "$(dirname "$0")/environment.py" > "${out_environment}"
fi

for base_dir in "$@"; do
    for lib in mkl_sparse suite_sparse eigen_sparse; do
        readarray -t -d '' files < <(find "$base_dir" -name '*.txt.bz2' -type f -print0)

        for file in "${files[@]}"; do
            read -d '' rel_fn < <(realpath --relative-base="${base_dir}" --zero "${file}")
            read problem_dir < <(basename "$(dirname "$rel_fn")")
            read problem_fn < <(basename "$rel_fn")
            problem_name="${problem_fn%%.txt.bz2}"

            if [[ ! $problem_name =~ problem-([[:digit:]]+)-([[:digit:]]+)-pre ]]; then
                continue
            fi

            declare -i images=${BASH_REMATCH[1]}
            declare -i points=${BASH_REMATCH[2]}
            #echo $images $observations
            #echo "${problem_dir}"
            eval_dir="${out}/${lib}/${problem_dir}"
            log_fn="${eval_dir}/${problem_name}.log"

            if grep --quiet -e "^${problem_dir},${problem_name},${lib}" -- "${out_csv}"; then
                echo "🚫 skipping $file"
                continue
            fi

            echo "📑 ${rel_fn}"

            if ! $dry_run; then
                mkdir -pv "${eval_dir}"

                #ldd "$(command -v bundle_adjuster)" | grep -E 'mkl|iomp|blas'
                #exit 1

                # gdb --ex r --args
                if systemd-run --user --scope -p MemoryMax=110G -- bundle_adjuster --input=<(bzcat "$file") \
                    --sparse_linear_algebra_library ${lib} --v=3 --stderrthreshold=0 2>&1 | \
                    tee "${log_fn}"; then
                    read duration < <(rg -oN '^Total\s+(.+)$' --replace '$1' "${log_fn}")

                    # A failing solver still exits successfully and reports
                    # the time until the failure.
                    if rg --quiet '^Termination:\s+(USER_)?FAILURE' "${log_fn}"; then
                        duration=
                    fi

                    printf '%s,%s,%s,%s\n' "${problem_dir}" "${problem_name}" "${lib}" "${duration}" >> "${out_csv}"
                else
                    # Leave cell empty on failure
                    printf '%s,%s,%s,\n' "${problem_dir}" "${problem_name}" "${lib}" >> "${out_csv}"
                fi
            else
                # if (($points < 4456117)); then
                #     duration='✅'
                # else
                #     duration='❌'
                # fi
                duration='✅'
                printf '%s,%s,%s,%s\n' "${problem_dir}" "${problem_name}" "${lib}" "${duration}"
            fi
        done
    done
done
```

### `environment.py`

```python
#!/usr/bin/env python3
"""Describes the machine and the software of a benchmark run as Markdown.

The software is read from the bundle_adjuster binary used by the benchmark:
the library versions Ceres was compiled with, the compiler, the build
configuration of its CMake build directory and the BLAS, LAPACK and OpenMP
libraries it and CHOLMOD load.
"""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

# Shared libraries that determine the dense linear algebra and threading.
LINKED_LIBRARY = re.compile(
    r'^lib(mkl_|iomp|gomp|blas|cblas|lapack|openblas|flexiblas|cholmod)')
# CMake cache entries that describe the build of Ceres.
CACHE_ENTRIES = [
    'CMAKE_BUILD_TYPE',
    'BUILD_SHARED_LIBS',
    'MKL_INTERFACE_FULL',
    'MKL_THREADING',
    'MKL_LINK',
]
CPU_FREQUENCY = Path('/sys/devices/system/cpu/cpu0/cpufreq')
# Names of the libraries in the version string of Ceres.
LIBRARY_NAMES = {
    'eigen': 'Eigen',
    'mkl': 'oneMKL',
    'suitesparse': 'SuiteSparse',
    'metis': 'METIS',
    'cuda': 'CUDA',
}
MHZ_PER_GHZ = 1000
BYTES_PER_KIB = 1024
KIB_PER_GIB = 1024**2


def run(*command):
    # The C locale keeps decimal points in numbers.
    return subprocess.run(
        command, check=True, capture_output=True, text=True,
        env={'LC_ALL': 'C', 'PATH': os.environ.get('PATH', '')},
    ).stdout


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def lscpu():
    """Returns the fields of lscpu, including the nested ones."""
    fields = {}

    def collect(entries):
        for entry in entries:
            fields[entry['field'].rstrip(':')] = entry['data']
            collect(entry.get('children', []))

    collect(json.loads(run('lscpu', '--json'))['lscpu'])
    return fields


def memory_gib():
    meminfo = read('/proc/meminfo')
    kib = int(re.search(r'^MemTotal:\s+(\d+) kB', meminfo, re.M).group(1))
    return kib / KIB_PER_GIB


def os_name():
    os_release = read('/etc/os-release')
    return re.search(r'^PRETTY_NAME="?([^"\n]*)', os_release, re.M).group(1)


def cpu_frequency_scaling():
    settings = [
        f'`{read(CPU_FREQUENCY / name)}` {label}'
        for name, label in [
            ('scaling_driver', 'driver'),
            ('scaling_governor', 'governor'),
            ('energy_performance_preference', 'energy preference'),
        ]
        if read(CPU_FREQUENCY / name)
    ]
    boost = read('/sys/devices/system/cpu/cpufreq/boost')
    if boost is not None:
        settings.append('boost ' + ('enabled' if boost == '1' else 'disabled'))
    return ', '.join(settings)


def ceres_version(binary):
    """Returns the version string Ceres reports in its solver summary."""
    match = re.search(r'^\d+\.\d+\.\d+\S*-eigen-\S+$', run('strings', binary),
                      re.M)
    return match.group(0) if match else None


def library_versions(version):
    """Splits the Ceres version string into the versions of its libraries."""
    ceres, _, libraries = version.partition('-')
    versions = {'Ceres Solver': ceres}
    for name, library_version in re.findall(r'([a-z]+)-\(([^)]+)\)', libraries):
        versions[LIBRARY_NAMES.get(name, name)] = library_version
    return versions


def toolchain(binary):
    """Returns the compilers and linkers recorded in the binary."""
    comment = run('readelf', '--string-dump=.comment', binary)
    tools = set()
    for entry in re.findall(r'\]\s+(.+)$', comment, re.M):
        match = re.match(r'GCC: \([^)]*\) (\S+(?: \d+)?)', entry)
        if match:
            tools.add(f'GCC {match.group(1)}')
        elif match := re.match(r'(\S+) (\d[\d.]*)', entry):
            tools.add(f'{match.group(1)} {match.group(2)}')
    return ', '.join(sorted(tools))


def cmake_cache(binary):
    """Returns the CMake cache of the build directory containing binary."""
    for directory in Path(binary).resolve().parents:
        cache = directory / 'CMakeCache.txt'
        if cache.is_file():
            entries = {}
            for line in cache.read_text().splitlines():
                match = re.match(r'^([^#/][^:=]*):[A-Z]+=(.*)$', line)
                if match:
                    entries[match.group(1)] = match.group(2)
            return entries
    return {}


def source_revision(directory):
    def git(*arguments):
        return run('git', '-C', directory, *arguments).strip()

    try:
        revision = git('describe', '--always', '--dirty')
        branch = git('rev-parse', '--abbrev-ref', 'HEAD')
    except (OSError, subprocess.CalledProcessError):
        return None
    return f'`{revision}` on branch `{branch}`'


def linked_libraries(binary):
    """Returns the matching shared libraries binary loads with their paths."""
    libraries = {}
    for line in run('ldd', binary).splitlines():
        match = re.match(r'\s*(\S+) => (\S+)', line)
        if match and LINKED_LIBRARY.search(match.group(1)):
            libraries[match.group(1)] = match.group(2)
    return libraries


def table(rows):
    lines = ['| | |', '|:--|:--|']
    lines += [f'| {name} | {value} |' for name, value in rows if value]
    return '\n'.join(lines)


def describe(binary):
    cpu = lscpu()
    machine = [
        ('CPU', cpu.get('Model name')),
        ('Cores', f'{cpu["Core(s) per socket"]} per socket, '
                  f'{cpu["Socket(s)"]} socket, '
                  f'{cpu["CPU(s)"]} hardware threads'),
        ('Maximum CPU clock',
         f'{float(cpu["CPU max MHz"]) / MHZ_PER_GHZ:.2f} GHz'),
        ('L3 cache', cpu.get('L3 cache')),
        ('Memory', f'{memory_gib():.1f} GiB usable'),
        ('Operating system', os_name()),
        ('Linux kernel', platform.release()),
        ('CPU frequency scaling', cpu_frequency_scaling()),
    ]

    cache = cmake_cache(binary)
    build_type = cache.get('CMAKE_BUILD_TYPE', '')
    flags = ' '.join(
        cache.get(name, '')
        for name in ['CMAKE_CXX_FLAGS', f'CMAKE_CXX_FLAGS_{build_type.upper()}']
    ).strip()
    software = [('Benchmark binary', f'`{binary}`')]
    version = ceres_version(binary)
    if version:
        software += list(library_versions(version).items())
    software += [
        ('Toolchain', toolchain(binary)),
        ('Ceres sources when described',
         source_revision(cache.get('CMAKE_HOME_DIRECTORY', ''))),
        ('Ceres compiler flags', f'`{flags}`' if flags else None),
    ]
    software += [(f'`{name}`', f'`{cache[name]}`')
                 for name in CACHE_ENTRIES if name in cache]

    libraries = linked_libraries(binary)
    sections = [
        '## Machine\n', table(machine),
        '\n## Software\n', table(software),
        '\n### Shared Libraries\n',
        'The BLAS, LAPACK and OpenMP libraries that `bundle_adjuster` loads, '
        'according to `ldd`:\n',
        table([(f'`{name}`', f'`{path}`') for name, path in libraries.items()]),
    ]
    cholmod = next((path for name, path in libraries.items()
                    if name.startswith('libcholmod')), None)
    if cholmod:
        sections += [
            '\nThe corresponding libraries of CHOLMOD:\n',
            table([(f'`{name}`', f'`{path}`')
                   for name, path in linked_libraries(cholmod).items()]),
        ]
    return '\n'.join(sections) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--binary', default=shutil.which('bundle_adjuster'),
        help='bundle_adjuster binary of the benchmark, by default the one in '
        'the PATH')
    args = parser.parse_args()
    if args.binary is None:
        parser.error('bundle_adjuster must be in the PATH or given by --binary')
    print(describe(args.binary), end='')


if __name__ == '__main__':
    main()
```

### `report.py`

```python
import re

import numpy as np
import pandas as pd
from datetime import timedelta

LIBRARY_NAMES = {
    'eigen_sparse': 'Eigen',
    'mkl_sparse': 'oneMKL',
    'suite_sparse': 'SuiteSparse',
}
SUMMARY_LIBRARIES = ['mkl_sparse', 'suite_sparse', 'eigen_sparse']
# Introduces the report and describes the benchmark configuration.
HEADER = 'report_header.md'
# Scripts reproduced in full at the end of the report.
SCRIPTS = [
    ('evaluate.sh', 'bash'),
    ('environment.py', 'python'),
    ('report.py', 'python'),
]
# Converted to HTML, every table is sized to its content. The dataset tables,
# which have four columns, get fixed column widths so that they are rendered
# alike. pandoc assigns percentage widths to columns of wide tables, which
# would stretch them to the page width and are therefore overridden.
STYLE = '''<style>
col {
  width: auto !important;
}
table:has(th:nth-child(4):last-child) :is(td, th):first-child {
  width: 16em;
}
table:has(th:nth-child(4):last-child) :is(td, th) + :is(td, th) {
  width: 8em;
}
.failed {
  color: red;
}
</style>
'''
# Marks failed runs in bold, and in red where the style applies.
FAILED = '<strong class="failed">failed</strong>'
# Describes the machine and the software of the evaluation and replaces this
# placeholder of the header.
ENVIRONMENT = 'evaluation/environment.md'
ENVIRONMENT_PLACEHOLDER = '<!-- environment -->'


def problem_size(problem):
    """Returns the number of cameras and points of a BAL problem name."""
    _, cameras, points, _ = problem.split('-')
    return int(cameras), int(points)


def format_durations(durations):
    """Formats a row of durations in seconds with the minimum in bold."""
    fastest = None if durations.isna().all() else durations.idxmin()
    formatted = {}
    for library, duration in durations.items():
        if np.isnan(duration):
            formatted[library] = FAILED
        elif library == fastest:
            formatted[library] = f'**{duration:.2f}**'
        else:
            formatted[library] = f'{duration:.2f}'
    return pd.Series(formatted)


def pipe_table(table, widths):
    """Formats a table of strings as Markdown with the given column widths.

    The first column is left aligned and the others right aligned.
    """
    def row(cells):
        padded = [cell.ljust(widths[column]) if index == 0 else
                  cell.rjust(widths[column])
                  for index, (column, cell) in enumerate(cells)]
        return '| ' + ' | '.join(padded) + ' |'

    separator = ['-' * (widths[column] - 1) + ('-' if index == 0 else ':')
                 for index, column in enumerate(table.columns)]
    separator[0] = ':' + separator[0][1:]
    lines = [row(zip(table.columns, table.columns)),
             '| ' + ' | '.join(separator) + ' |']
    lines += [row(zip(table.columns, values))
              for values in table.itertuples(index=False)]
    return '\n'.join(lines)


def failure_reason(dataset, problem, library):
    """Returns the reason the log of a failed run gives for the failure."""
    try:
        with open(f'evaluation/{library}/{dataset}/{problem}.log') as file:
            log = file.read()
    except OSError:
        return 'no log'
    match = re.search(r'Linear solver fatal error: (.+)$', log, re.M)
    if match:
        return f'`{match.group(1).strip()}`'
    match = re.search(r'^Termination:\s+(.+)$', log, re.M)
    if match:
        return f'`{match.group(1).strip()}`'
    return 'the log ends without a solver summary'


def speedup(durations, baseline):
    """Returns the geometric mean speedup of oneMKL over baseline."""
    ratio = (durations[baseline] / durations['mkl_sparse']).dropna()
    return np.exp(np.log(ratio).mean())


def write_markdown(durations, filename):
    """Writes the report with a summary and the durations of every dataset."""
    # Problems that failed with every library have no fastest library.
    fastest = durations.dropna(how='all').idxmin(axis='columns')

    def summarize(group):
        row = {'Problems': len(group)}
        for library in SUMMARY_LIBRARIES:
            row[f'Fastest: {LIBRARY_NAMES[library]}'] = int(
                fastest.reindex(group.index).eq(library).sum())
        row['oneMKL speedup over SuiteSparse'] = speedup(group, 'suite_sparse')
        row['oneMKL speedup over Eigen'] = speedup(group, 'eigen_sparse')
        return row

    rows = {dataset: summarize(group)
            for dataset, group in durations.groupby('dataset')}
    rows['**all**'] = summarize(durations)
    summary = pd.DataFrame.from_dict(rows, orient='index')
    speedups = [column for column in summary.columns if 'speedup' in column]
    summary[speedups] = summary[speedups].map('{:.2f}'.format)
    summary.index.name = 'Dataset'

    with open(HEADER) as file:
        header = file.read()
    with open(ENVIRONMENT) as file:
        header = header.replace(ENVIRONMENT_PLACEHOLDER, file.read().rstrip())

    sections = [
        STYLE,
        header,
        '## Summary\n',
        'Speedups are geometric means of the ratio of total solve times. '
        'Problems that failed with a library are excluded from the speedup '
        'over that library. The tables of the datasets show the shortest '
        'duration of every problem in bold.\n',
        summary.to_markdown(
            colalign=['left'] + ['right'] * len(summary.columns),
            disable_numparse=True),
    ]
    tables = {}
    for dataset, group in durations.groupby('dataset'):
        group = group.droplevel('dataset')
        group = group.loc[sorted(group.index, key=problem_size)]
        table = group.apply(format_durations, axis='columns')
        table.columns = [f'{LIBRARY_NAMES[library]} [s]'
                         for library in table.columns]
        tables[dataset] = table.rename_axis('Problem').reset_index()

    # Converters such as pandoc derive the relative column widths of a wide
    # table from its separator line, so all dataset tables share the widths
    # of their widest cells to be rendered alike.
    columns = next(iter(tables.values())).columns
    widths = {
        column: max(len(column), *(int(table[column].str.len().max())
                                   for table in tables.values()))
        for column in columns
    }
    sections.append('\n## Durations')
    for dataset, table in tables.items():
        sections += [f'\n### {dataset.capitalize()}\n',
                     pipe_table(table, widths)]

    failed = durations.stack(future_stack=True)
    failed = failed[failed.isna()].index
    if len(failed):
        sections.append('\n## Failed Runs\n')
        sections.append('The following runs recorded no duration, with the '
                        'reason given in their logs in '
                        '`evaluation/<library>/<dataset>/`.\n')
        sections += [f'- {LIBRARY_NAMES[library]} on `{dataset}/{problem}`: '
                     f'{failure_reason(dataset, problem, library)}'
                     for dataset, problem, library in failed]

    sections.append('\n## Scripts')
    for script, language in SCRIPTS:
        with open(script) as file:
            sections += [f'\n### `{script}`\n',
                         f'```{language}\n{file.read().rstrip()}\n```']

    with open(filename, 'w') as file:
        file.write('\n'.join(sections) + '\n')


def main():
    df = pd.read_csv(
        'evaluation/results.csv', index_col=['dataset', 'problem', 'library']
    )

    df_by_duration = df.unstack(level='library')

    df_duration = df_by_duration.loc(axis=1)[['duration']]
    df_speedup = df_duration.rename(columns=dict(duration='speedup to eigen_sparse'))
    df_speedup_base = df_speedup.loc(axis=1)[
        [('speedup to eigen_sparse', 'eigen_sparse')]
    ]
    df_speedup_other = df_speedup.loc(axis=1)[
        [
            ('speedup to eigen_sparse', 'mkl_sparse'),
            ('speedup to eigen_sparse', 'suite_sparse'),
        ]
    ]
    df_speedup = df_speedup_base.values / df_speedup_other
    # print(df_speedup_base)
    # print(df_speedup)

    df_all = pd.concat([df_by_duration, df_speedup], axis='columns')
    # print(df_all)

    # print(df_by_duration)
    styler_duration = df_all.style.highlight_min(
        axis=1,
        subset=[
            ('duration', 'eigen_sparse'),
            ('duration', 'mkl_sparse'),
            ('duration', 'suite_sparse'),
        ],
        props='font-weight: bold',
    )
    styler_speedup = styler_duration.highlight_max(
        axis=1,
        subset=[
            ('speedup to eigen_sparse', 'mkl_sparse'),
            ('speedup to eigen_sparse', 'suite_sparse'),
        ],
        props='font-weight: bold',
    )

    df_duration.style.highlight_min(axis=1, props='font-weight: bold').to_html('report.html')
    write_markdown(df_duration['duration'], 'results.md')


    #print(df_by_duration.sum(axis='index'))

    #print(timedelta(seconds=df_by_duration.to_numpy().sum(where=df_by_duration.notna().to_numpy())))

    # print(styler_speedup.to_html())


if __name__ == '__main__':
    main()
```
