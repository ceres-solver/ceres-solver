#/usr/bin/env bash

# directory of this script and the repository root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
REPO_DIR="$SCRIPT_DIR/.."

# exit on error
set -e

#set -x

# run file generation
cd "$REPO_DIR"/internal/ceres
python generate_template_specializations.py
python generate_bundle_adjustment_tests.py

# check if any files are now modified and error if yes
(if [ -n "`git diff --name-only --diff-filter=M --ignore-submodules`" ]; then echo $'\n    > Some generated files are different from the committed version. Rerun code generation.\n'; git diff --diff-filter=M; false; fi)

echo $'\n    > Generated files match the checked-in files.\n'

# run formatting
cd "$REPO_DIR"
./scripts/format_all.sh

# check if any files are now modified and error if yes
(if [ -n "`git diff --name-only --diff-filter=M --ignore-submodules`" ]; then echo $'\n    > Some files are not properly formatted. You can use "./scripts/format_all.sh".\n'; git diff --diff-filter=M; false; fi)

echo $'\n    > Formatting ok'
