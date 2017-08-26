#!/bin/bash

# Bash script to assert that the current version of the NDK is at least the
# specified version. Prints 'true' to standard out if it's the right version,
# 'false' if it's not.
#
# Typically used like this, in your jni/Android.mk:
#
#   ifneq ($(shell $(LOCAL_PATH)/assert_ndk_version.sh "r5c" "ndk-dir"), true)
#     $(error NDK version r5c or greater required)
#   endif
#
# See https://gist.github.com/2878774 for asserting SDK version.
#
# Retrieved from: https://gist.github.com/jorgenpt/1961404 on 2014-06-03.
#
# Copyright (c) 2012, Lookout, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Author: jorgenpt@gmail.com (Jorgen Tjerno)
#         alexs.mac@gmail.com (Alex Stewart)

# Extracts 'r5c' into '5 c', also handles newer versions of the form
# 'r9d (64-bit)' and versions >= 10.
function get_major_minor_rNx_style() {
  # r9d (64-bit) -> '9d', also handle versions >= 10.
  local version=$(echo "$1" | sed 's/r\([0-9]\{1,2\}[a-z]\{0,1\}\).*/\1/')
  local major=$(echo "$version" | sed 's/\([0-9]\{1,2\}\).*/\1/')
  local minor=$(echo "$version" | sed 's/^[0-9]*//')
  echo "$major $minor"
}

# Extracts the major and minor versions from the <NDK_ROOT>/source.properties
# file and converts to the standard <NUMBER> <LETTER> format, e.g. 15 c.
#
# Usage: get_major_minor_from_source_properties <SOURCE_PROPERTIES_CONTENTS>
function get_major_minor_from_source_properties() {
  # <NDK_ROOT>/source.properties contains (e.g. for r15c):
  #
  # Pkg.Desc = Android NDK
  # Pkg.Revision = 15.2.4203891
  #
  # match to 15 c
  version=$(echo $1 | sed 's/.*Pkg.Revision[[:space:]]*=[[:space:]]*//')
  declare -r major=$(echo $version | sed 's/\([0-9]\{1,2\}\).*/\1/')
  declare -r minor=$(echo $version | sed 's/\([0-9]\{1,2\}\)\.\([0-9]\{1,2\}\).*/\2/')
  declare -r patch=$(echo $version | sed 's/\([0-9]\{1,2\}\)\.\([0-9]\{1,2\}\)\.\([0-9]*\)/\3/')
  # Convert numeric minor version to letter version, e.g: 0 -> a, 1 -> b, 2 -> c etc.
  minor_letter_ascii_code=$(($minor + 97)) # 97 = 'a' in ASCII.
  minor_letter=($(printf "\\$(printf %o "$minor_letter_ascii_code")"))
  echo "$major $minor_letter"
}

if [[ -z "$2" ]]; then
  echo "Usage: $0 <required version> <NDK_ROOT>" >&2
  echo " For example: $0 r5c android-ndk-r9d" >&2
  exit 1
fi

# Assert that the expected version is at least 4.
declare -a expected_version
expected_version=( $(get_major_minor_rNx_style "$1") )
if [[ ${expected_version[0]} -le 4 ]]; then
  echo "Cannot test for versions less than r5: r4 doesn't have a version file." >&2
  echo false
  exit 1
fi

# NDK versions <= r4 did not have RELEASE.TXT, nor do versions >= r11, where it was
# replaced by source.properties.  As we just asserted that we are looking for >= r5
# if RELEASE.TXT is not present, source.properties should be.
declare -r release_file="$2/RELEASE.TXT"
declare -r source_properties_file="$2/source.properties"
declare -a actual_version
if [ ! -s "$release_file" ]; then
  if [ ! -s "$source_properties_file" ]; then
    echo "ERROR: Failed to find either RELEASE.TXT or source.properties in NDK_ROOT=$2" >&2
    echo false
    exit 1
  fi
  # NDK version >= r11.
  if [ ! -s "$source_properties_file" ]; then
     echo "ERROR: Failed to find source.properties file in NDK_ROOT=$1" >&2
     echo false
     exit 1
  fi
  source_properties=$(<"$source_properties_file")
  actual_version=($(get_major_minor_from_source_properties "$source_properties"))
  if [ -z "$source_properties" ] || [ -z "${actual_version[0]}" ]; then
    echo "ERROR: Invalid source.properties: $(cat $source_properties_file)" >&2
    echo false
    exit 1
  fi
else
  # NDK version >= r5 && < r11.
  version=$(grep '^r' $release_file)
  actual_version=( $(get_major_minor_rNx_style "$version") )
  if [ -z "$version" ] || [ -z "${actual_version[0]}" ]; then
    echo "ERROR: Invalid RELEASE.TXT: $(cat $release_file)" >&2
    echo false
    exit 1
  fi
fi

if [[ ${actual_version[0]} -lt ${expected_version[0]} ]]; then
  echo "false"
elif [[ ${actual_version[0]} -eq ${expected_version[0]} ]]; then
  # This uses < and not -lt because they're string identifiers (a, b, c, etc)
  if [[ "${actual_version[1]}" < "${expected_version[1]}" ]]; then
    echo "false"
  else
    echo "true"
  fi
else
  echo "true"
fi
