fn main() {
    let imgui = std::path::Path::new("..").join("third_party").join("imgui");
    println!("cargo:rerun-if-changed={}", imgui.join("imgui.cpp").display());
    println!("cargo:rerun-if-env-changed=IMGUI_DIR");
}
