#!/usr/bin/env bash
# torch and faiss-cpu each ship their own libomp.dylib. Loading both aborts
# with "OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib
# already initialized" the first time a FAISS search and a torch model run in
# one process. The documented fix is to have a single OpenMP runtime, so this
# points faiss at torch's copy. Both are LLVM libomp at ABI 5.0.0.
#
# macOS only, and only needed for the wheels; a conda faiss build does not hit
# this. Re-run after recreating the venv.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
site="$(cd "$root" && ./venv/bin/python -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
faiss_lib="$site/faiss/.dylibs/libomp.dylib"
torch_lib="$site/torch/lib/libomp.dylib"

[ -e "$torch_lib" ] || { echo "no torch libomp at $torch_lib; nothing to do"; exit 0; }
[ -e "$faiss_lib" ] || { echo "no faiss libomp at $faiss_lib; nothing to do"; exit 0; }
[ -L "$faiss_lib" ] && { echo "already linked"; exit 0; }

cp "$faiss_lib" "$faiss_lib.orig"
ln -sf "$torch_lib" "$faiss_lib"
echo "linked faiss libomp -> torch libomp (original kept at libomp.dylib.orig)"
