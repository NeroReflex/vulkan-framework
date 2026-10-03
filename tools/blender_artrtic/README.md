# ArtRTic Blender add-on

Exports a **runtime-ready** folder for the engine:

```
my_level/
  scene.json
  assets/
    Chair.tar    # one self-contained object per mesh (reusable)
    Table.tar
```

## One-time setup (no PATH chore)

1. **Install the add-on** (Blender 4.3 example):

   ```bash
   ./install.sh 4.3
   ```

   This builds `artrtic-cook`, copies it to `addon/bin/`, symlinks **Compressonator** from your machine if found, and links the add-on into Blender.

2. In Blender: **Edit → Preferences → Add-ons → ArtRTic** → enable.

3. If you skipped step 1 or moved machines: **ArtRTic panel → Install cook tools into addon** (or run `install.sh` again).

The add-on uses **`addon/bin/artrtic-cook`** automatically. `compressonatorcli` is resolved from the same `bin/` folder (or `~/.bin/compressonatorcli-*` at install time). You do **not** need global PATH entries.

Optional override: **Cook override** in the panel (only if you want a custom binary).

## Export

1. **3D Viewport → N → ArtRTic**
2. Export directory, mesh scope → **Export ArtRTic scene**

## Run the engine

```bash
cd /path/to/my_level
/path/to/vulkan-framework/target/release/artrtic
```

`scene.json` paths are resolved relative to that file.

## Layout of `addon/bin/`

| File | Source |
|------|--------|
| `artrtic-cook` | Built from this repo (`cargo build -p artrtic-cook --release`) |
| `compressonatorcli` | Symlink at install (AMD Compressonator — still required for BC7, not redistributable in git) |
| `toktx` | Optional symlink (manifest cook only) |

Compressonator is too large to ship inside the repo; `install.sh` only **links** your local install into `bin/`.

## Manual add-on link

```bash
mkdir -p ~/.config/blender/4.3/scripts/addons
ln -sf /path/to/vulkan-framework/tools/blender_artrtic ~/.config/blender/4.3/scripts/addons/artrtic
./install.sh   # without version: only fills bin/, does not symlink Blender
```
