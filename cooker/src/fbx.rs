//! Cook Assimp FBX into a skinned object tar (skeleton, clips, meshes).

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use russimp::animation::Animation;
use russimp::mesh::Mesh;
use russimp::node::Node;
use russimp::scene::{PostProcess, Scene};
use russimp::{Matrix4x4, Vector3D};
use serde::Serialize;

use crate::skin_format::{
    AnimationGpuChannel, ArmatureGpuElement, SkeletonGpuElement, MAX_ANIMATION_CHANNELS,
    MAX_ANIMATION_KEYS_PER_CHANNEL,
};
use crate::tar_out::append_bytes;

const STATIC_VERTEX_BYTES: usize = 32;
const SKINNED_VERTEX_BYTES: usize = 64;

#[derive(Serialize)]
struct AnimationMeta {
    name: String,
    duration_ticks: f64,
    ticks_per_second: f64,
    channel_count: u32,
}

#[derive(Serialize)]
struct ObjectMeta {
    bone_count: u32,
    armature_node_count: u32,
    animations: Vec<AnimationMeta>,
}

#[derive(Serialize)]
struct ObjectManifest {
    animations: Vec<String>,
    meshes: Vec<String>,
}

pub fn pack_fbx(input: &Path, output: &Path) -> Result<(), String> {
    let scene = Scene::from_file(
        input.to_str().ok_or("non-utf8 path")?,
        vec![
            PostProcess::Triangulate,
            PostProcess::JoinIdenticalVertices,
            PostProcess::SortByPrimitiveType,
            PostProcess::LimitBoneWeights,
            PostProcess::FlipUVs,
            PostProcess::GenerateSmoothNormals,
        ],
    )
    .map_err(|err| err.to_string())?;

    let root = scene
        .root
        .as_ref()
        .ok_or("fbx has no root node")?
        .clone();

    let mut armature_nodes = Vec::new();
    let mut name_to_armature = HashMap::new();
    flatten_armature(&root, 0, &mut armature_nodes, &mut name_to_armature, &mut 0u32);

    let mut skeleton = Vec::new();
    let mut bone_name_to_index = HashMap::new();
    for mesh in &scene.meshes {
        for bone in &mesh.bones {
            if bone_name_to_index.contains_key(&bone.name) {
                continue;
            }
            let Some(armature_index) = name_to_armature.get(&bone.name).copied() else {
                eprintln!(
                    "warning: bone '{}' has no armature node; skipping",
                    bone.name
                );
                continue;
            };
            let index = skeleton.len() as u32;
            bone_name_to_index.insert(bone.name.clone(), index);
            skeleton.push(SkeletonGpuElement {
                offset_matrix: assimp_to_glsl_mat4(&bone.offset_matrix),
                armature_node_index: armature_index,
                _padding: [0; 3],
            });
        }
    }

    if skeleton.is_empty() {
        return Err("no skinning bones found in fbx".into());
    }

    let file = File::create(output).map_err(|err| err.to_string())?;
    let mut builder = tar::Builder::new(file);

    append_struct_slice(&mut builder, "skeleton/original", &skeleton);
    append_struct_slice(&mut builder, "skeleton/armature", &armature_nodes);

    let mut animation_names = Vec::new();
    let mut animation_meta = Vec::new();
    for animation in &scene.animations {
        let channels = build_channels(animation, &name_to_armature)?;
        let archive_name = sanitize_clip_name(&animation.name);
        append_struct_slice(
            &mut builder,
            &format!("animations/{archive_name}/channels"),
            &channels,
        );
        animation_names.push(archive_name.clone());
        animation_meta.push(AnimationMeta {
            name: archive_name,
            duration_ticks: animation.duration,
            ticks_per_second: if animation.ticks_per_second > 0.0 {
                animation.ticks_per_second
            } else {
                1.0
            },
            channel_count: channels.len() as u32,
        });
    }

    let meta = ObjectMeta {
        bone_count: skeleton.len() as u32,
        armature_node_count: armature_nodes.len() as u32,
        animations: animation_meta,
    };
    append_bytes(
        &mut builder,
        "meta",
        serde_json::to_string_pretty(&meta)
            .map_err(|err| err.to_string())?
            .as_bytes(),
    );

    let mut mesh_names = Vec::new();
    for (mesh_index, mesh) in scene.meshes.iter().enumerate() {
        if mesh.vertices.is_empty() {
            continue;
        }
        let mesh_name = if mesh.name.is_empty() {
            format!("mesh_{mesh_index}")
        } else {
            sanitize_clip_name(&mesh.name)
        };
        mesh_names.push(mesh_name.clone());

        let (static_vertices, skinned_vertices, indices) =
            pack_mesh_vertices(mesh, &bone_name_to_index)?;

        append_bytes(&mut builder, "vertex_buffer", &static_vertices);
        append_bytes(&mut builder, "skinned_vertex_buffer", &skinned_vertices);
        append_bytes(
            &mut builder,
            &format!("models/{mesh_name}/indexes"),
            &indices,
        );
        append_bytes(
            &mut builder,
            &format!("models/{mesh_name}/material"),
            b"default",
        );
        append_bytes(&mut builder, &format!("models/{mesh_name}/skin"), &[]);
    }

    let manifest = ObjectManifest {
        animations: animation_names,
        meshes: mesh_names,
    };
    append_bytes(
        &mut builder,
        "manifest",
        serde_json::to_string_pretty(&manifest)
            .map_err(|err| err.to_string())?
            .as_bytes(),
    );

    builder.finish().map_err(|err| err.to_string())?;
    Ok(())
}

fn flatten_armature(
    node: &Rc<Node>,
    parent_index: u32,
    out: &mut Vec<ArmatureGpuElement>,
    names: &mut HashMap<String, u32>,
    next_index: &mut u32,
) {
    let index = *next_index;
    *next_index += 1;
    names.insert(node.name.clone(), index);
    out.push(ArmatureGpuElement {
        transform: assimp_to_glsl_mat4(&node.transformation),
        parent_index,
        _padding: [0; 3],
    });
    for child in node.children.borrow().iter() {
        flatten_armature(child, index, out, names, next_index);
    }
}

fn build_channels(
    animation: &Animation,
    name_to_armature: &HashMap<String, u32>,
) -> Result<Vec<AnimationGpuChannel>, String> {
    let mut channels = Vec::new();
    for node_anim in &animation.channels {
        let Some(armature_index) = name_to_armature.get(&node_anim.name).copied() else {
            eprintln!(
                "warning: animation channel '{}' not in armature; skipping",
                node_anim.name
            );
            continue;
        };
        if node_anim.position_keys.len() > MAX_ANIMATION_KEYS_PER_CHANNEL
            || node_anim.rotation_keys.len() > MAX_ANIMATION_KEYS_PER_CHANNEL
            || node_anim.scaling_keys.len() > MAX_ANIMATION_KEYS_PER_CHANNEL
        {
            return Err(format!(
                "animation '{}' exceeds {} keys on a channel",
                animation.name, MAX_ANIMATION_KEYS_PER_CHANNEL
            ));
        }
        if channels.len() >= MAX_ANIMATION_CHANNELS {
            return Err(format!(
                "animation '{}' exceeds {} channels",
                animation.name, MAX_ANIMATION_CHANNELS
            ));
        }

        let mut gpu = AnimationGpuChannel::default();
        gpu.armature_element_index = armature_index;
        gpu.position_key_count = node_anim.position_keys.len() as u32;
        for (i, key) in node_anim.position_keys.iter().enumerate() {
            gpu.position_key_times[i] = key.time as f32;
            gpu.position_key_value_x[i] = key.value.x;
            gpu.position_key_value_y[i] = key.value.y;
            gpu.position_key_value_z[i] = key.value.z;
        }
        gpu.rotation_key_count = node_anim.rotation_keys.len() as u32;
        for (i, key) in node_anim.rotation_keys.iter().enumerate() {
            gpu.rotation_key_times[i] = key.time as f32;
            gpu.rotation_key_value_x[i] = key.value.x;
            gpu.rotation_key_value_y[i] = key.value.y;
            gpu.rotation_key_value_z[i] = key.value.z;
            gpu.rotation_key_value_w[i] = key.value.w;
        }
        gpu.scaling_key_count = node_anim.scaling_keys.len() as u32;
        for (i, key) in node_anim.scaling_keys.iter().enumerate() {
            gpu.scaling_key_times[i] = key.time as f32;
            gpu.scaling_key_value_x[i] = key.value.x;
            gpu.scaling_key_value_y[i] = key.value.y;
            gpu.scaling_key_value_z[i] = key.value.z;
        }
        channels.push(gpu);
    }
    Ok(channels)
}

fn pack_mesh_vertices(
    mesh: &Mesh,
    bone_name_to_index: &HashMap<String, u32>,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>), String> {
    let vertex_count = mesh.vertices.len();
    let mut vertex_weights: Vec<Vec<(u32, f32)>> = vec![Vec::new(); vertex_count];
    for bone in &mesh.bones {
        let Some(bone_index) = bone_name_to_index.get(&bone.name).copied() else {
            continue;
        };
        for weight in &bone.weights {
            let slot = weight.vertex_id as usize;
            if slot >= vertex_weights.len() {
                continue;
            }
            vertex_weights[slot].push((bone_index, weight.weight));
        }
    }

    let mut static_vertices = Vec::with_capacity(vertex_count * STATIC_VERTEX_BYTES);
    let mut skinned_vertices = Vec::with_capacity(vertex_count * SKINNED_VERTEX_BYTES);
    for vi in 0..vertex_count {
        let pos = &mesh.vertices[vi];
        let normal = mesh.normals.get(vi).copied().unwrap_or(Vector3D {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        });
        let uv = mesh
            .texture_coords
            .first()
            .and_then(|layer| layer.as_ref())
            .and_then(|coords| coords.get(vi))
            .copied()
            .unwrap_or(Vector3D {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            });

        static_vertices.extend_from_slice(&pack_static_vertex(pos, &normal, &uv));

        let mut weights = vertex_weights[vi].clone();
        weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        weights.truncate(4);
        let weight_sum: f32 = weights.iter().map(|(_, w)| *w).sum();
        if weight_sum > 0.0 {
            for (_, w) in weights.iter_mut() {
                *w /= weight_sum;
            }
        }
        skinned_vertices.extend_from_slice(&pack_skinned_vertex(
            pos,
            &normal,
            &uv,
            &weights,
        ));
    }

    let mut indices = Vec::new();
    for face in &mesh.faces {
        if face.0.len() != 3 {
            continue;
        }
        for index in &face.0 {
            indices.extend_from_slice(&index.to_le_bytes());
        }
    }

    Ok((static_vertices, skinned_vertices, indices))
}

fn pack_static_vertex(pos: &Vector3D, normal: &Vector3D, uv: &Vector3D) -> [u8; 32] {
    let mut out = [0u8; 32];
    write_f32(&mut out[0..], pos.x);
    write_f32(&mut out[4..], pos.y);
    write_f32(&mut out[8..], pos.z);
    write_f32(&mut out[12..], normal.x);
    write_f32(&mut out[16..], normal.y);
    write_f32(&mut out[20..], normal.z);
    write_f32(&mut out[24..], uv.x);
    write_f32(&mut out[28..], uv.y);
    out
}

fn pack_skinned_vertex(
    pos: &Vector3D,
    normal: &Vector3D,
    uv: &Vector3D,
    weights: &[(u32, f32)],
) -> [u8; 64] {
    let mut joints = [0u32; 4];
    let mut w = [0.0f32; 4];
    for (i, (joint, weight)) in weights.iter().enumerate().take(4) {
        joints[i] = *joint;
        w[i] = *weight;
    }
    // Layout matches `deform.comp` SkinVertex packing.
    let mut out = [0u8; 64];
    write_f32(&mut out[0..], pos.x);
    write_f32(&mut out[4..], pos.y);
    write_f32(&mut out[8..], pos.z);
    write_f32(&mut out[12..], w[2]);
    write_f32(&mut out[16..], normal.x);
    write_f32(&mut out[20..], normal.y);
    write_f32(&mut out[24..], normal.z);
    write_f32(&mut out[28..], w[3]);
    write_f32(&mut out[32..], uv.x);
    write_f32(&mut out[36..], uv.y);
    write_f32(&mut out[40..], w[0]);
    write_f32(&mut out[44..], w[1]);
    for (slot, joint) in joints.iter().enumerate() {
        write_u32(&mut out[48 + slot * 4..], *joint);
    }
    out
}

fn write_f32(dst: &mut [u8], value: f32) {
    dst[..4].copy_from_slice(&value.to_le_bytes());
}

fn write_u32(dst: &mut [u8], value: u32) {
    dst[..4].copy_from_slice(&value.to_le_bytes());
}

fn assimp_to_glsl_mat4(m: &Matrix4x4) -> [f32; 16] {
    [
        m.a1, m.b1, m.c1, m.d1, m.a2, m.b2, m.c2, m.d2, m.a3, m.b3, m.c3, m.d3, m.a4, m.b4,
        m.c4, m.d4,
    ]
}

fn append_struct_slice<T: Copy>(builder: &mut tar::Builder<File>, path: &str, data: &[T]) {
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * std::mem::size_of::<T>())
    };
    append_bytes(builder, path, bytes);
}

fn sanitize_clip_name(name: &str) -> String {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return "clip".into();
    }
    trimmed
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect()
}

pub fn default_minotaur_path() -> PathBuf {
    PathBuf::from("/home/denis/projects/UNIVR_CGPROJ/animazioni/Minotaur@Jump.FBX")
}

pub fn minotaur_fbx_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("ART_RTIC_MINOTAUR_FBX") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }
    let default = default_minotaur_path();
    if default.is_file() {
        Some(default)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    #[ignore]
    fn minotaur_fbx_cooks() {
        let Some(input) = minotaur_fbx_path() else {
            eprintln!("skipping minotaur_fbx_cooks: set ART_RTIC_MINOTAUR_FBX or install default path");
            return;
        };
        let out = std::env::temp_dir().join(format!("minotaur-{}.tar", std::process::id()));
        pack_fbx(&input, &out).expect("pack_fbx");

        let file = File::open(&out).unwrap();
        let mut archive = tar::Archive::new(file);
        let mut saw_skeleton = false;
        let mut saw_channels = false;
        for entry in archive.entries().unwrap() {
            let mut entry = entry.unwrap();
            let path = entry.path().unwrap();
            let path = path.to_string_lossy();
            if path.contains("skeleton/original") {
                saw_skeleton = true;
                let size = entry.header().size().unwrap() as usize;
                assert!(size >= 80);
                assert_eq!(size % 80, 0);
            }
            if path.contains("animations/") && path.ends_with("/channels") {
                saw_channels = true;
                let mut payload = Vec::new();
                entry.read_to_end(&mut payload).unwrap();
                assert!(payload.len() >= std::mem::size_of::<AnimationGpuChannel>());
            }
        }
        assert!(saw_skeleton, "missing skeleton/original");
        assert!(saw_channels, "missing animation channels");
        let _ = std::fs::remove_file(out);
    }
}
