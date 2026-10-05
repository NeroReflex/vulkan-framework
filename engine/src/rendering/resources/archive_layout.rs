//! Path names inside an object tar. The loader still walks the archive once
//! and copies each payload into a mapped GPU buffer. This classification is
//! what that walk branches on.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArchiveEntry {
    VertexBuffer,
    SkinnedVertexBuffer,
    TextureData(String),
    MaterialDiffuse(String),
    MaterialNormal(String),
    MaterialReflection(String),
    MaterialDisplacement(String),
    ModelIndexes(String),
    ModelMaterial(String),
    ModelSkin(String),
    SkeletonOriginal,
    SkeletonArmature,
    AnimationChannels(String),
    Meta,
    Manifest,
}

pub fn classify(path: &str) -> Option<ArchiveEntry> {
    let trimmed = path.strip_prefix("./").unwrap_or(path);
    let mut parts = trimmed.split('/').filter(|part| !part.is_empty());
    let kind = parts.next()?;
    match kind {
        "vertex_buffer" => Some(ArchiveEntry::VertexBuffer),
        "skinned_vertex_buffer" => Some(ArchiveEntry::SkinnedVertexBuffer),
        "skeleton" => match parts.next()? {
            "original" => Some(ArchiveEntry::SkeletonOriginal),
            "armature" => Some(ArchiveEntry::SkeletonArmature),
            _ => None,
        },
        "animations" => {
            let name = parts.next()?.to_string();
            match parts.next()? {
                "channels" => Some(ArchiveEntry::AnimationChannels(name)),
                _ => None,
            }
        }
        "meta" => Some(ArchiveEntry::Meta),
        "manifest" => Some(ArchiveEntry::Manifest),
        "textures" => {
            let name = parts.next()?.to_string();
            match parts.next()? {
                "data" => Some(ArchiveEntry::TextureData(name)),
                "width.txt" | "height.txt" | "miplevel.txt" => None,
                _ => None,
            }
        }
        "materials" => {
            let name = parts.next()?.to_string();
            match parts.next()? {
                "diffuse_texture" => Some(ArchiveEntry::MaterialDiffuse(name)),
                "normal_texture" => Some(ArchiveEntry::MaterialNormal(name)),
                "reflection_texture" => Some(ArchiveEntry::MaterialReflection(name)),
                "displacement_texture" => Some(ArchiveEntry::MaterialDisplacement(name)),
                _ => None,
            }
        }
        "models" => {
            let name = parts.next()?.to_string();
            match parts.next()? {
                "indexes" => Some(ArchiveEntry::ModelIndexes(name)),
                "material" => Some(ArchiveEntry::ModelMaterial(name)),
                "skin" => Some(ArchiveEntry::ModelSkin(name)),
                _ => None,
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use tar::{Builder, Header};

    #[test]
    fn current_tar_names_classify() {
        assert_eq!(
            classify("./vertex_buffer"),
            Some(ArchiveEntry::VertexBuffer)
        );
        assert_eq!(
            classify("textures/brick/data"),
            Some(ArchiveEntry::TextureData("brick".into()))
        );
        assert_eq!(classify("./textures/brick/width.txt"), None);
        assert_eq!(
            classify("./models/floor/indexes"),
            Some(ArchiveEntry::ModelIndexes("floor".into()))
        );
        assert_eq!(
            classify("./models/floor/skin"),
            Some(ArchiveEntry::ModelSkin("floor".into()))
        );
        assert_eq!(
            classify("skeleton/original"),
            Some(ArchiveEntry::SkeletonOriginal)
        );
        assert_eq!(
            classify("./skeleton/armature"),
            Some(ArchiveEntry::SkeletonArmature)
        );
        assert_eq!(
            classify("animations/Jump/channels"),
            Some(ArchiveEntry::AnimationChannels("Jump".into()))
        );
        assert_eq!(classify("meta"), Some(ArchiveEntry::Meta));
        assert_eq!(classify("./manifest"), Some(ArchiveEntry::Manifest));
    }

    #[test]
    fn tar_round_trip_keeps_the_two_mip_payload_size() {
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut builder = Builder::new(&mut cursor);
            let payload = vec![0u8; 80];
            let mut header = Header::new_gnu();
            header.set_path("textures/brick/data").unwrap();
            header.set_size(payload.len() as u64);
            header.set_cksum();
            builder.append(&header, payload.as_slice()).unwrap();
            builder.finish().unwrap();
        }
        cursor.set_position(0);
        let mut archive = tar::Archive::new(cursor);
        let entry = archive.entries().unwrap().next().unwrap().unwrap();
        let path = entry.path().unwrap();
        assert_eq!(
            classify(&path.to_string_lossy()),
            Some(ArchiveEntry::TextureData("brick".into()))
        );
        assert_eq!(entry.header().size().unwrap(), 80);
    }
}
