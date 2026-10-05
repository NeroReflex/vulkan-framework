//! Builds an object tar. Compressonator and toktx are required processes.
//! This crate does not link them, and the engine never calls them.

use std::fs::{self, File};
use std::path::{Path, PathBuf};

use serde::Deserialize;

mod fbx;
mod obj;
mod repack;
mod skin_format;
mod tar_out;
mod tools;

const COMPRESSONATOR: &str = "compressonatorcli";
const TOKTX: &str = "toktx";

#[derive(Debug, Deserialize)]
struct Manifest {
    #[serde(default)]
    textures: Vec<TextureJob>,
    #[serde(default)]
    meshes: Vec<MeshJob>,
}

#[derive(Debug, Deserialize)]
struct TextureJob {
    name: String,
    source: PathBuf,
    /// `bc7` or `astc`.
    format: String,
}

#[derive(Debug, Deserialize)]
struct MeshJob {
    name: String,
    indexes: PathBuf,
    material: String,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let command = args.next().unwrap_or_else(|| {
        print_usage();
        std::process::exit(2);
    });

    match command.as_str() {
        "repack" => {
            let input = args.next().unwrap_or_else(|| {
                eprintln!("usage: artrtic-cook repack <in.tar> <out.tar>");
                std::process::exit(2);
            });
            let output = args.next().unwrap_or_else(|| {
                eprintln!("usage: artrtic-cook repack <in.tar> <out.tar>");
                std::process::exit(2);
            });
            repack::repack(Path::new(&input), Path::new(&output)).unwrap_or_else(|err| {
                eprintln!("repack failed: {err}");
                std::process::exit(1);
            });
        }
        "obj" => {
            let scene_dir = args.next().unwrap_or_else(|| {
                eprintln!("usage: artrtic-cook obj <scene_dir> <out.tar>");
                std::process::exit(2);
            });
            let output = args.next().unwrap_or_else(|| {
                eprintln!("usage: artrtic-cook obj <scene_dir> <out.tar>");
                std::process::exit(2);
            });
            if let Err(missing) = require_bc7_tool() {
                eprintln!("obj packing requires {COMPRESSONATOR}; missing {missing}");
                std::process::exit(1);
            }
            obj::pack_obj(
                Path::new(&scene_dir),
                Path::new(&output),
                |source, dest| {
                    let status = run_compressonator(source, dest).map_err(|err| err.to_string())?;
                    if status.success() {
                        Ok(())
                    } else {
                        Err(format!("compressonatorcli failed for {}", source.display()))
                    }
                },
            )
            .unwrap_or_else(|err| {
                eprintln!("obj pack failed: {err}");
                std::process::exit(1);
            });
        }
        "fbx" => {
            let input = args.next().unwrap_or_else(|| {
                eprintln!("usage: artrtic-cook fbx <in.fbx> <out.tar>");
                std::process::exit(2);
            });
            let output = args.next().unwrap_or_else(|| {
                eprintln!("usage: artrtic-cook fbx <in.fbx> <out.tar>");
                std::process::exit(2);
            });
            fbx::pack_fbx(Path::new(&input), Path::new(&output)).unwrap_or_else(|err| {
                eprintln!("fbx pack failed: {err}");
                std::process::exit(1);
            });
        }
        "validate" => {
            let input = args.next().unwrap_or_else(|| {
                eprintln!("usage: artrtic-cook validate <object.tar>");
                std::process::exit(2);
            });
            repack::validate_textures(Path::new(&input)).unwrap_or_else(|err| {
                eprintln!("validate failed: {err}");
                std::process::exit(1);
            });
        }
        _ => {
            let manifest_path = command;
            let out_path = args.next().unwrap_or_else(|| {
                print_usage();
                std::process::exit(2);
            });

            if let Err(missing) = require_tools() {
                eprintln!("creating a tar requires {COMPRESSONATOR} and {TOKTX}; missing {missing}");
                std::process::exit(1);
            }

            let text = fs::read_to_string(&manifest_path).expect("manifest");
            let manifest: Manifest = serde_json::from_str(&text).expect("manifest json");
            cook_manifest(&manifest, Path::new(&out_path)).expect("cook");
        }
    }
}

fn print_usage() {
    eprintln!("usage:");
    eprintln!("  artrtic-cook <manifest.json> <out.tar>");
    eprintln!("  artrtic-cook repack <in.tar> <out.tar>");
    eprintln!("  artrtic-cook validate <object.tar>");
    eprintln!("  artrtic-cook obj <scene_dir> <out.tar>");
    eprintln!("  artrtic-cook fbx <in.fbx> <out.tar>");
}

pub fn cook_manifest(manifest: &Manifest, out_path: &Path) -> Result<(), String> {
    require_tools()?;
    let scratch = std::env::temp_dir().join(format!("artrtic-cook-{}", std::process::id()));
    fs::create_dir_all(&scratch).map_err(|err| err.to_string())?;

    let file = File::create(out_path).map_err(|err| err.to_string())?;
    let mut builder = tar::Builder::new(file);

    for texture in &manifest.textures {
        let cooked = cook_texture(texture, &scratch);
        tar_out::append_bytes(
            &mut builder,
            &format!("textures/{}/data", texture.name),
            &cooked,
        );
    }

    for mesh in &manifest.meshes {
        let indexes = fs::read(&mesh.indexes).map_err(|err| err.to_string())?;
        tar_out::append_bytes(
            &mut builder,
            &format!("models/{}/indexes", mesh.name),
            &indexes,
        );
        tar_out::append_bytes(
            &mut builder,
            &format!("models/{}/material", mesh.name),
            mesh.material.as_bytes(),
        );
    }

    builder.finish().map_err(|err| err.to_string())?;
    let _ = fs::remove_dir_all(&scratch);
    Ok(())
}

pub fn require_tools() -> Result<(), String> {
    tools::require_tools(&[COMPRESSONATOR, TOKTX])
}

pub fn require_bc7_tool() -> Result<(), String> {
    tools::require_tools(&[COMPRESSONATOR])
}

fn run_compressonator(source: &Path, dest: &Path) -> std::io::Result<std::process::ExitStatus> {
    tools::run_tool(
        COMPRESSONATOR,
        &[
            "-fd",
            "BC7",
            "-mipsize",
            "1",
            source.to_str().unwrap_or(""),
            dest.to_str().unwrap_or(""),
        ],
    )
}

fn cook_texture(texture: &TextureJob, scratch: &Path) -> Vec<u8> {
    match texture.format.as_str() {
        "bc7" => {
            let dest = scratch.join(format!("{}.dds", texture.name));
            let status = run_compressonator(&texture.source, &dest)
                .expect("compressonatorcli");
            if !status.success() {
                panic!("compressonatorcli failed for {}", texture.name);
            }
            fs::read(dest).unwrap()
        }
        "astc" => {
            let dest = scratch.join(format!("{}.ktx2", texture.name));
            let status = tools::run_tool(
                TOKTX,
                &[
                    "--encode",
                    "astc",
                    "--astc_blk_d",
                    "4x4",
                    "--genmipmap",
                    dest.to_str().unwrap_or(""),
                    texture.source.to_str().unwrap_or(""),
                ],
            )
            .expect("toktx");
            if !status.success() {
                panic!("toktx failed for {}", texture.name);
            }
            fs::read(dest).unwrap()
        }
        other => panic!("unsupported cook format {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cooker_names_the_external_tools_and_does_not_link_them() {
        assert_eq!(COMPRESSONATOR, "compressonatorcli");
        assert_eq!(TOKTX, "toktx");
    }

    /// Runs when `compressonatorcli` and `toktx` are on PATH (see `~/.bashrc`).
    #[test]
    fn cooks_bc7_and_astc_into_a_tar_when_tools_are_installed() {
        if require_tools().is_err() {
            eprintln!("skipping cook integration: compressonatorcli or toktx not on PATH");
            return;
        }

        let scratch = std::env::temp_dir().join(format!("artrtic-cook-it-{}", std::process::id()));
        fs::create_dir_all(&scratch).unwrap();
        let png = scratch.join("tex.png");
        fs::write(&png, MINIMAL_PNG_16).unwrap();

        let manifest = Manifest {
            textures: vec![
                TextureJob {
                    name: "bc7tex".into(),
                    source: png.clone(),
                    format: "bc7".into(),
                },
                TextureJob {
                    name: "astctex".into(),
                    source: png.clone(),
                    format: "astc".into(),
                },
            ],
            meshes: vec![],
        };

        let out = scratch.join("out.tar");
        cook_manifest(&manifest, &out).expect("cook_manifest");

        let file = File::open(&out).unwrap();
        let mut archive = tar::Archive::new(file);
        let mut bc7 = None;
        let mut astc = None;
        for entry in archive.entries().unwrap() {
            let mut entry = entry.unwrap();
            let path = entry.path().unwrap().into_owned();
            let mut data = Vec::new();
            std::io::copy(&mut entry, &mut data).unwrap();
            if path.ends_with("textures/bc7tex/data") {
                bc7 = Some(data);
            } else if path.ends_with("textures/astctex/data") {
                astc = Some(data);
            }
        }
        let bc7 = bc7.expect("bc7 payload in tar");
        let astc = astc.expect("astc payload in tar");
        assert_eq!(&bc7[0..4], b"DDS ");
        assert_eq!(&astc[0..12], KTX2_IDENT);

        let _ = fs::remove_dir_all(scratch);
    }

    const KTX2_IDENT: [u8; 12] = [
        0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
    ];

    const MINIMAL_PNG_16: &[u8] = &[
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 16, 0, 0, 0,
        16, 8, 2, 0, 0, 0, 144, 145, 104, 54, 0, 0, 0, 22, 73, 68, 65, 84, 120, 218, 99, 56,
        97, 100, 68, 18, 98, 24, 213, 48, 170, 97, 248, 106, 0, 0, 20, 4, 44, 16, 227, 99,
        117, 140, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
    ];
}
