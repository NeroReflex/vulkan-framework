pub mod archive_layout;
pub mod collection;
pub mod directional_lights;
pub mod materials;
pub mod mesh;
pub mod object;
pub mod skin_asset;
pub mod texture;

use thiserror::Error;

use crate::rendering::resources::object::MaterialGPU;

#[derive(Error, Debug)]
pub enum ResourceError {
    #[error("All material slots are occupied, there is no room for a new one")]
    NoMaterialSlotAvailable,

    #[error("All texture slots are occupied, there is no room for a new one")]
    NoTextureSlotAvailable,

    #[error("All mesh slots are occupied, there is no room for a new one")]
    NoMeshSlotAvailable,

    #[error("All directional lights slots are occupied, there is no room for a new one")]
    NoDirectionalLightingSlotAvailable,

    #[error("No such mesh found: {0}")]
    NoMesh(usize),

    #[error("Incomplete texture: {0}")]
    IncompleteTexture(String),

    #[error("Invalid object format")]
    InvalidObjectFormat,

    #[error("Resource is too large to fit in reserved GPU memory")]
    ResourceTooLarge,

    #[error("Cannot remove the empty texture")]
    AttemptedRemovalOfEmptyTexture,

    #[error("Missing vertex buffer data")]
    MissingVertexBuffer,

    #[error("Resource index is out of range: {0}")]
    ResourceIndexOutOfRange(usize),

    #[error("Upload timeline counter exhausted")]
    UploadCounterExhausted,

    #[error("Resource content revision exhausted")]
    RevisionExhausted,

    #[error("No such directional light found: {0}")]
    NoDirectionalLight(u32),

    #[error("Host TLAS builds are not supported by the device-only resource loader")]
    UnsupportedHostTLASBuild,

    #[error("The scene acceleration structure is not ready")]
    TLASNotReady,
}

pub type ResourceResult<T> = Result<T, ResourceError>;

pub(super) fn next_revision(revision: u64) -> crate::rendering::RenderingResult<u64> {
    revision.checked_add(1).ok_or_else(|| {
        crate::rendering::RenderingError::ResourceError(ResourceError::RevisionExhausted)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn revisions_are_monotonic_and_never_wrap() {
        assert_eq!(next_revision(0).unwrap(), 1);
        assert_eq!(next_revision(u64::MAX - 1).unwrap(), u64::MAX);
        assert!(matches!(
            next_revision(u64::MAX),
            Err(crate::rendering::RenderingError::ResourceError(
                ResourceError::RevisionExhausted
            ))
        ));
    }
}

const SIZEOF_MATERIAL_DEFINITION: usize = std::mem::size_of::<MaterialGPU>();
