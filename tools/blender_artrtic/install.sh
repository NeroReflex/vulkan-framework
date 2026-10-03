#!/usr/bin/env bash
# Install cook binaries next to the Blender add-on (no PATH required).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
BIN="$ROOT/bin"
mkdir -p "$BIN"

echo "Building artrtic-cook..."
(cd "$REPO_ROOT" && cargo build -p artrtic-cook --release)
cp -f "$REPO_ROOT/target/release/artrtic-cook" "$BIN/artrtic-cook"
chmod +x "$BIN/artrtic-cook"

link_compressonator() {
  local dest="$BIN/compressonatorcli"
  if [[ -x "$dest" ]]; then
    return 0
  fi
  if command -v compressonatorcli >/dev/null 2>&1; then
    local found
    found="$(command -v compressonatorcli)"
    ln -sf "$found" "$dest"
    echo "Linked compressonatorcli from PATH: $found"
    return 0
  fi
  for dir in "$HOME/.bin/compressonatorcli-"*; do
    if [[ -x "$dir/compressonatorcli" ]]; then
      ln -sf "$dir/compressonatorcli" "$dest"
      echo "Linked compressonatorcli from $dir"
      return 0
    fi
  done
  echo "Note: compressonatorcli not found — download AMD Compressonator, then re-run install.sh"
  echo "      or copy/symlink compressonatorcli into: $BIN/"
  return 1
}

link_compressonator || true

if command -v toktx >/dev/null 2>&1; then
  ln -sf "$(command -v toktx)" "$BIN/toktx"
  echo "Linked toktx (manifest cook only; OBJ export uses BC7)."
fi

echo "Done. Add-on bin directory:"
ls -la "$BIN"

VERSION="${1:-}"
if [[ -n "$VERSION" ]]; then
  DEST="${HOME}/.config/blender/${VERSION}/scripts/addons/artrtic"
  mkdir -p "$(dirname "$DEST")"
  ln -sfn "$ROOT" "$DEST"
  echo "Symlinked add-on for Blender ${VERSION}: $DEST"
fi
