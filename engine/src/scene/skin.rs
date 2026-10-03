//! CPU reference for the GPU skinning palette.
//!
//! Each bone walks its parent chain the same way `shaders/skin/animate.comp`
//! does: `current = local * current`, then `palette = global * inverse_bind`.
//! A parent index equal to the bone's own index is the root.

#[derive(Debug, Clone, Copy)]
pub struct BoneLocal {
    pub local: [f32; 16],
    pub inverse_bind: [f32; 16],
    pub parent: u32,
}

fn mul(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for col in 0..4 {
        for row in 0..4 {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += a[k * 4 + row] * b[col * 4 + k];
            }
            out[col * 4 + row] = sum;
        }
    }
    out
}

pub fn palette_from_locals(bones: &[BoneLocal]) -> Vec<[f32; 16]> {
    bones
        .iter()
        .enumerate()
        .map(|(bone_index, _)| {
            let mut current_index = bone_index;
            let mut global = [
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ];
            for _ in 0..64 {
                let bone = &bones[current_index];
                global = mul(&bone.local, &global);
                if bone.parent as usize == current_index {
                    break;
                }
                current_index = bone.parent as usize;
            }
            mul(&global, &bones[bone_index].inverse_bind)
        })
        .collect()
}

/// Second vertex stream for a skinned mesh. Static meshes stay 32 bytes.
#[derive(Debug, Clone, Copy)]
pub struct SkinnedVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub joints: [u32; 4],
    pub weights: [f32; 4],
}

pub fn dominant_joint(vertex: &SkinnedVertex) -> u32 {
    let mut best = 0usize;
    for i in 1..4 {
        if vertex.weights[i] > vertex.weights[best] {
            best = i;
        }
    }
    vertex.joints[best]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn translation(x: f32, y: f32, z: f32) -> [f32; 16] {
        [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, x, y, z, 1.0,
        ]
    }

    fn identity() -> [f32; 16] {
        [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]
    }

    #[test]
    fn two_bone_palette_walks_parent_then_child() {
        let bones = [
            BoneLocal {
                local: translation(1.0, 0.0, 0.0),
                inverse_bind: identity(),
                parent: 0,
            },
            BoneLocal {
                local: translation(0.0, 1.0, 0.0),
                inverse_bind: identity(),
                parent: 0,
            },
        ];
        let palette = palette_from_locals(&bones);
        assert!((palette[0][12] - 1.0).abs() < 1e-5);
        assert!(palette[0][13].abs() < 1e-5);
        assert!((palette[1][12] - 1.0).abs() < 1e-5);
        assert!((palette[1][13] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn dominant_weight_selects_the_joint() {
        let vertex = SkinnedVertex {
            position: [0.0; 3],
            normal: [0.0, 1.0, 0.0],
            uv: [0.0; 2],
            joints: [3, 7, 1, 0],
            weights: [0.1, 0.7, 0.2, 0.0],
        };
        assert_eq!(dominant_joint(&vertex), 7);
        assert_eq!(vertex.position, [0.0; 3]);
        assert_eq!(vertex.normal, [0.0, 1.0, 0.0]);
        assert_eq!(vertex.uv, [0.0; 2]);
    }
}
