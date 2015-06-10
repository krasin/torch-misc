#!/bin/bash
# Update all third_party packages.

set -ue

git pull -s subtree git@github.com:karpathy/char-rnn.git master
