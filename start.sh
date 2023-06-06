#!/bin/bash
DIR=$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd);
cd $DIR
export COMMANDLINE_ARGS="--skip-python-version-check --skip-version-check --skip-torch-cuda-test --skip-prepare-environment --allow-code --opt-split-attention --disable-safe-unpickle --xformers --theme dark --no-gradio-queue --api --api-log --add-stop-route --port 6006"
python_cmd="<path to env python bin>"
LAUNCH_SCRIPT="$DIR/launch.py"
"${python_cmd}" "${LAUNCH_SCRIPT}" "$@"

