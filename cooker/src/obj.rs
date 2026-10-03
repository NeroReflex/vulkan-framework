//! Pack a Wavefront OBJ scene (obj + mtl + textures/) into an object tar.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use crate::tar_out::{append_bytes, append_symlink};

const VERTEX_BYTES: usize = 32;

#[derive(Clone, Default)]
struct MaterialMaps {
    diffuse: Option<String>,
    displacement: Option<String>,
    normal: Option<String>,
}

struct MeshPart {
    name: String,
    material: String,
    indices: Vec<u32>,
}

pub fn pack_obj(scene_dir: &Path, out_path: &Path, cook_bc7: fn(&Path, &Path) -> Result<(), String>) -> Result<(), String> {
    let obj_path = find_obj(scene_dir)?;
    let mtl_path = obj_path.with_extension("mtl");
    if !mtl_path.is_file() {
        return Err(format!("missing mtl next to {}", obj_path.display()));
    }

    let materials = parse_mtl(&mtl_path)?;
    let (vertex_data, meshes) = parse_obj(&obj_path)?;

    let texture_sources = collect_texture_paths(&materials, scene_dir)?;
    let scratch = std::env::temp_dir().join(format!("artrtic-obj-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).map_err(|err| err.to_string())?;

    let file = File::create(out_path).map_err(|err| err.to_string())?;
    let mut builder = tar::Builder::new(file);

    append_bytes(&mut builder, "vertex_buffer", &vertex_data);

    let mut cooked_textures: HashMap<String, Vec<u8>> = HashMap::new();
    for (archive_name, source_path) in texture_sources {
        eprintln!("cooking texture {archive_name} from {}", source_path.display());
        let dest = scratch.join(format!("{archive_name}.dds"));
        cook_bc7(&source_path, &dest)?;
        let bytes = std::fs::read(&dest).map_err(|err| err.to_string())?;
        cooked_textures.insert(archive_name, bytes);
    }

    for (name, bytes) in &cooked_textures {
        append_bytes(&mut builder, &format!("textures/{name}/data"), bytes);
    }

    for (material_name, maps) in &materials {
        if let Some(diffuse) = &maps.diffuse {
            let tex_name = texture_archive_name(diffuse);
            append_symlink(
                &mut builder,
                &format!("materials/{material_name}/diffuse_texture"),
                &format!("../../textures/{tex_name}"),
            );
        }
        if let Some(disp) = &maps.displacement {
            let tex_name = texture_archive_name(disp);
            append_symlink(
                &mut builder,
                &format!("materials/{material_name}/displacement_texture"),
                &format!("../../textures/{tex_name}"),
            );
        }
        if let Some(normal) = &maps.normal {
            let tex_name = texture_archive_name(normal);
            append_symlink(
                &mut builder,
                &format!("materials/{material_name}/normal_texture"),
                &format!("../../textures/{tex_name}"),
            );
        }
    }

    for mesh in &meshes {
        let index_bytes = mesh
            .indices
            .iter()
            .flat_map(|index| index.to_le_bytes())
            .collect::<Vec<_>>();
        append_bytes(&mut builder, &format!("models/{}/indexes", mesh.name), &index_bytes);
        append_symlink(
            &mut builder,
            &format!("models/{}/material", mesh.name),
            &format!("../../materials/{}", mesh.material),
        );
    }

    builder.finish().map_err(|err| err.to_string())?;
    let _ = std::fs::remove_dir_all(scratch);
    eprintln!(
        "packed {} vertices, {} meshes, {} textures -> {}",
        vertex_data.len() / VERTEX_BYTES,
        meshes.len(),
        cooked_textures.len(),
        out_path.display()
    );
    Ok(())
}

fn find_obj(dir: &Path) -> Result<PathBuf, String> {
    let mut objs: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|err| err.to_string())?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension().is_some_and(|ext| ext == "obj") {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    objs.sort();
    if objs.len() == 1 {
        return Ok(objs[0].clone());
    }
    if objs.is_empty() {
        return Err(format!("no .obj in {}", dir.display()));
    }
    Err(format!(
        "expected one .obj in {}, found {}",
        dir.display(),
        objs.len()
    ))
}

fn texture_archive_name(mtl_path: &str) -> String {
    let normalized = mtl_path.trim().replace('\\', "/");
    let rel = normalized
        .strip_prefix("textures/")
        .unwrap_or(normalized.as_str());
    format!("textures_{}", rel.replace('/', "_"))
}

fn collect_texture_paths(
    materials: &HashMap<String, MaterialMaps>,
    scene_dir: &Path,
) -> Result<Vec<(String, PathBuf)>, String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for maps in materials.values() {
        for path in [
            maps.diffuse.as_ref(),
            maps.displacement.as_ref(),
            maps.normal.as_ref(),
        ]
            .into_iter()
            .flatten()
        {
            let archive = texture_archive_name(path);
            if !seen.insert(archive.clone()) {
                continue;
            }
            let source = scene_dir.join(path.replace('\\', "/"));
            if !source.is_file() {
                return Err(format!("texture file missing: {}", source.display()));
            }
            out.push((archive, source));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

fn parse_mtl(path: &Path) -> Result<HashMap<String, MaterialMaps>, String> {
    let file = File::open(path).map_err(|err| err.to_string())?;
    let reader = BufReader::new(file);
    let mut materials = HashMap::new();
    let mut current = String::new();
    for line in reader.lines() {
        let line = line.map_err(|err| err.to_string())?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(name) = line.strip_prefix("newmtl ") {
            current = name.trim().to_string();
            materials.insert(current.clone(), MaterialMaps::default());
            continue;
        }
        let Some(entry) = materials.get_mut(&current) else {
            continue;
        };
        if let Some(path) = line.strip_prefix("map_Kd ") {
            entry.diffuse = Some(path.trim().to_string());
        } else if let Some(path) = line.strip_prefix("map_Disp ") {
            entry.displacement = Some(path.trim().to_string());
        } else if let Some(path) = line
            .strip_prefix("map_Bump ")
            .or_else(|| line.strip_prefix("map_bump "))
            .or_else(|| line.strip_prefix("map_Normal "))
            .or_else(|| line.strip_prefix("map_normal "))
        {
            entry.normal = Some(path.trim().to_string());
        }
    }
    Ok(materials)
}

fn parse_obj(path: &Path) -> Result<(Vec<u8>, Vec<MeshPart>), String> {
    let file = File::open(path).map_err(|err| err.to_string())?;
    let reader = BufReader::new(file);

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut texcoords: Vec<[f32; 2]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();

    let mut vertex_map: HashMap<VertexKey, u32> = HashMap::new();
    let mut vertex_data: Vec<u8> = Vec::new();

    let mut meshes: Vec<MeshPart> = Vec::new();
    let mut current_mesh: Option<MeshPart> = None;
    let mut default_material = String::from("default");

    for line in reader.lines() {
        let line = line.map_err(|err| err.to_string())?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace().collect::<Vec<_>>();
        let tag = parts[0];
        parts.remove(0);

        match tag {
            "v" if parts.len() >= 3 => {
                positions.push([
                    parts[0].parse().map_err(|_| "bad vertex")?,
                    parts[1].parse().map_err(|_| "bad vertex")?,
                    parts[2].parse().map_err(|_| "bad vertex")?,
                ]);
            }
            "vt" if parts.len() >= 2 => {
                texcoords.push([
                    parts[0].parse().map_err(|_| "bad vt")?,
                    parts[1].parse().map_err(|_| "bad vt")?,
                ]);
            }
            "vn" if parts.len() >= 3 => {
                normals.push([
                    parts[0].parse().map_err(|_| "bad vn")?,
                    parts[1].parse().map_err(|_| "bad vn")?,
                    parts[2].parse().map_err(|_| "bad vn")?,
                ]);
            }
            "usemtl" if !parts.is_empty() => {
                default_material = parts[0].to_string();
                if let Some(mesh) = &mut current_mesh {
                    mesh.material = default_material.clone();
                }
            }
            "g" if !parts.is_empty() => {
                if let Some(mesh) = current_mesh.take() {
                    if !mesh.indices.is_empty() {
                        meshes.push(mesh);
                    }
                }
                current_mesh = Some(MeshPart {
                    name: parts.join("_"),
                    material: default_material.clone(),
                    indices: Vec::new(),
                });
            }
            "f" if !parts.is_empty() => {
                let mesh = match &mut current_mesh {
                    Some(mesh) => mesh,
                    None => {
                        current_mesh = Some(MeshPart {
                            name: String::from("default_mesh"),
                            material: default_material.clone(),
                            indices: Vec::new(),
                        });
                        current_mesh.as_mut().unwrap()
                    }
                };
                let corners: Vec<VertexKey> = parts
                    .iter()
                    .map(|corner| parse_face_corner(corner, positions.len(), texcoords.len(), normals.len()))
                    .collect::<Result<Vec<_>, String>>()?;
                for tri in triangulate(&corners) {
                    for key in tri {
                        let index = lookup_vertex(
                            key,
                            &positions,
                            &texcoords,
                            &normals,
                            &mut vertex_map,
                            &mut vertex_data,
                        )?;
                        mesh.indices.push(index);
                    }
                }
            }
            _ => {}
        }
    }

    if let Some(mesh) = current_mesh.take() {
        if !mesh.indices.is_empty() {
            meshes.push(mesh);
        }
    }

    if vertex_data.is_empty() {
        return Err("obj produced no vertices".into());
    }
    if meshes.is_empty() {
        return Err("obj produced no meshes".into());
    }

    Ok((vertex_data, meshes))
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct VertexKey {
    position: usize,
    texcoord: Option<usize>,
    normal: Option<usize>,
}

fn parse_face_corner(
    corner: &str,
    n_pos: usize,
    n_uv: usize,
    n_norm: usize,
) -> Result<VertexKey, String> {
    let pieces: Vec<&str> = corner.split('/').collect();
    let vi = resolve_index(pieces.first().copied().unwrap_or(""), n_pos)?;
    let ti = if pieces.len() > 1 && !pieces[1].is_empty() {
        Some(resolve_index(pieces[1], n_uv)?)
    } else {
        None
    };
    let ni = if pieces.len() > 2 && !pieces[2].is_empty() {
        Some(resolve_index(pieces[2], n_norm)?)
    } else {
        None
    };
    Ok(VertexKey {
        position: vi,
        texcoord: ti,
        normal: ni,
    })
}

fn resolve_index(token: &str, count: usize) -> Result<usize, String> {
    if count == 0 {
        return Err("face references attributes before they are defined".into());
    }
    let index: isize = token.parse().map_err(|_| format!("bad index '{token}'"))?;
    let resolved = if index < 0 {
        count as isize + index
    } else {
        index - 1
    };
    if resolved < 0 || resolved as usize >= count {
        return Err(format!("index {} out of range for count {}", token, count));
    }
    Ok(resolved as usize)
}

fn triangulate(corners: &[VertexKey]) -> Vec<[VertexKey; 3]> {
    if corners.len() < 3 {
        return Vec::new();
    }
    let mut tris = Vec::new();
    for i in 1..corners.len() - 1 {
        tris.push([corners[0], corners[i], corners[i + 1]]);
    }
    tris
}

fn lookup_vertex(
    key: VertexKey,
    positions: &[[f32; 3]],
    texcoords: &[[f32; 2]],
    normals: &[[f32; 3]],
    vertex_map: &mut HashMap<VertexKey, u32>,
    vertex_data: &mut Vec<u8>,
) -> Result<u32, String> {
    if let Some(index) = vertex_map.get(&key) {
        return Ok(*index);
    }
    let pos = positions[key.position];
    let normal = match key.normal {
        Some(index) => normals[index],
        None if normals.is_empty() => [0.0, 1.0, 0.0],
        None => normals[0],
    };
    let uv = match key.texcoord {
        Some(index) => texcoords[index],
        None => [0.0, 0.0],
    };

    let index = (vertex_data.len() / VERTEX_BYTES) as u32;
    for value in pos {
        vertex_data.extend_from_slice(&value.to_le_bytes());
    }
    for value in normal {
        vertex_data.extend_from_slice(&value.to_le_bytes());
    }
    for value in uv {
        vertex_data.extend_from_slice(&value.to_le_bytes());
    }
    assert_eq!(vertex_data.len() % VERTEX_BYTES, 0);
    vertex_map.insert(key, index);
    Ok(index)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texture_archive_name_matches_crytek_layout() {
        assert_eq!(
            texture_archive_name("textures/lion.tga"),
            "textures_lion.tga"
        );
        assert_eq!(
            texture_archive_name("textures/sponza_column_b_diff.tga"),
            "textures_sponza_column_b_diff.tga"
        );
    }
}
