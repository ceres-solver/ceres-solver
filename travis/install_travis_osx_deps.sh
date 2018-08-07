#!/bin/bash

# Stop processing on any error.
set -e

function install_if_not_installed() {
  declare -r formula="$1"
  if [[ $(brew list ${formula} &>/dev/null; echo $?) -ne 0 ]]; then
    brew install ${formula}
  else
    echo "$0 - ${formula} is already installed."
  fi
}

install_if_not_installed cmake
install_if_not_installed glog
install_if_not_installed gflags
install_if_not_installed eigen
install_if_not_installed suite-sparse
