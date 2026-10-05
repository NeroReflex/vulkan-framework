# Third-party sources

## Dear ImGui

```bash
git submodule update --init third_party/imgui
```

To bump the version: `cd third_party/imgui && git fetch && git checkout <tag>`, then commit the submodule pointer in the parent repo.

`artrtic_imgui` compiles `imgui/*.cpp` and `backends/imgui_impl_sdl2.cpp` from this tree (no OpenGL backend).
