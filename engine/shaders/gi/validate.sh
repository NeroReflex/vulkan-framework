#!/bin/sh
# Compiles every GI shader the way inline_spirv! does (#version 460 wrapper,
# Vulkan 1.2) and runs spirv-val on the result. Run from anywhere.
set -u
root="$(cd "$(dirname "$0")/../../.." && pwd)"
out="$(mktemp -d)"
status=0
for path in "$root"/engine/shaders/gi/*.comp "$root"/engine/shaders/gi/*.rgen \
    "$root"/engine/shaders/gi/*.rchit "$root"/engine/shaders/gi/*.rmiss \
    "$root"/engine/shaders/ao/*.comp "$root"/engine/shaders/ao/*.rgen \
    "$root"/engine/shaders/ao/*.rchit "$root"/engine/shaders/ao/*.rmiss \
    "$root"/engine/shaders/final_rendering/final_rendering.frag; do
    rel="${path#"$root"/}"
    ext="${path##*.}"
    name="$(basename "$path")"
    printf '#version 460\n#extension GL_GOOGLE_include_directive : enable\n#include "%s"\n' "$rel" \
        > "$out/$name.wrap.$ext"
    if ! glslangValidator -I"$root" --target-env vulkan1.2 -S "$ext" -o "$out/$name.spv" \
        "$out/$name.wrap.$ext" > "$out/$name.log" 2>&1; then
        echo "COMPILE FAILED: $rel"
        grep -E "ERROR|error" "$out/$name.log" | head -5
        status=1
        continue
    fi
    if ! spirv-val --target-env vulkan1.2 "$out/$name.spv" > "$out/$name.val" 2>&1; then
        echo "SPIR-V INVALID: $rel"
        head -5 "$out/$name.val"
        status=1
    fi
done
rm -rf "$out"
[ $status -eq 0 ] && echo "all GI shaders compile and validate"
exit $status
