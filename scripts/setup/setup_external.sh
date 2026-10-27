#!/usr/bin/env bash
# Fetches and builds the external programs used by the experiments:
#   - CoCo+ : the authors' public code (https://github.com/majid75/CoCo, shipped there as CoCo.zip), algorithm 4,
#             with the seed-offset patch of patches/coco_larc.patch (13 lines, outside the algorithm);
#   - linkern: Concorde's chained Lin-Kernighan, built from the Concorde sources bundled with CoCo, used by LARC.
# Output: external/CoCo-src/coco_seeded and bin/linkern. Needs git, unzip, patch, make and a C/C++ compiler.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXT="$ROOT/external"
mkdir -p "$EXT" "$ROOT/bin"
cd "$EXT"
[ -d CoCo ] || git clone https://github.com/majid75/CoCo.git
if [ ! -d CoCo-src ]; then
  unzip -q CoCo/CoCo.zip -d unzipped && mv unzipped/CoCo CoCo-src && rm -rf unzipped
  cd CoCo-src
  rm -rf build dist lib/concorde.a CoCo coco          # rebuild everything from source
  patch -p1 --forward < "$ROOT/patches/coco_larc.patch"
  cd ..
fi
cd CoCo-src
echo "building Concorde (library and linkern)..."
rm -rf concorde_build && mkdir concorde_build && cd concorde_build
# Concorde (2003) predates C99: compile it in the gnu89 dialect. Its configure script cannot guess Apple Silicon hosts,
# so the host type is given explicitly on macOS (the compiler still builds natively).
HOSTOPT=""
if [ "$(uname -s)" = "Darwin" ]; then HOSTOPT="--host=x86_64-apple-darwin"; fi
CFLAGS="-O2 -std=gnu89 -Wno-implicit-function-declaration -Wno-implicit-int -Wno-int-conversion -Wno-incompatible-pointer-types" \
  ../concorde/configure $HOSTOPT > configure.log 2>&1 || { echo "Concorde configure failed: external/CoCo-src/concorde_build/configure.log"; exit 1; }
make > make.log 2>&1 || { echo "Concorde build failed: external/CoCo-src/concorde_build/make.log"; exit 1; }
mkdir -p ../lib && cp concorde.a ../lib/
cp LINKERN/linkern "$ROOT/bin/linkern"
cd ..
echo "building CoCo+..."
make -f Makefile CONF=Release > build.log 2>&1 || { echo "CoCo build failed: external/CoCo-src/build.log"; exit 1; }
cp dist/Release/GNU-Linux/coco coco_seeded
echo "built: bin/linkern and external/CoCo-src/coco_seeded"
