//! Rigid scene graph. Node transforms are column-major 4x4 matrices, matching
//! the GLSL `mat4` the mesh shader builds from a Vulkan row-major 3x4.

mod skin;
mod file;

pub use file::load_scene_json;
pub use skin::{palette_from_locals, BoneLocal};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4 {
    /// Column-major, 16 floats.
    pub columns: [f32; 16],
}

impl Mat4 {
    pub const fn identity() -> Self {
        Self {
            columns: [
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        }
    }

    pub fn from_vulkan_rows(rows: &[f32; 12]) -> Self {
        Self {
            columns: [
                rows[0], rows[4], rows[8], 0.0, rows[1], rows[5], rows[9], 0.0, rows[2], rows[6],
                rows[10], 0.0, rows[3], rows[7], rows[11], 1.0,
            ],
        }
    }

    pub fn translation(x: f32, y: f32, z: f32) -> Self {
        let mut matrix = Self::identity();
        matrix.columns[12] = x;
        matrix.columns[13] = y;
        matrix.columns[14] = z;
        matrix
    }

    pub fn mul(self, other: Self) -> Self {
        let mut out = [0.0f32; 16];
        for col in 0..4 {
            for row in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += self.columns[k * 4 + row] * other.columns[col * 4 + k];
                }
                out[col * 4 + row] = sum;
            }
        }
        Self { columns: out }
    }

    pub fn transform_point(self, point: [f32; 3]) -> [f32; 3] {
        let m = &self.columns;
        [
            m[0] * point[0] + m[4] * point[1] + m[8] * point[2] + m[12],
            m[1] * point[0] + m[5] * point[1] + m[9] * point[2] + m[13],
            m[2] * point[0] + m[6] * point[1] + m[10] * point[2] + m[14],
        ]
    }

    /// Vulkan `TransformMatrixKHR` rows: translation lives in the last column
    /// of each row, which is what `row_major_3x4` in the mesh shader consumes.
    pub fn to_vulkan_rows(self) -> [f32; 12] {
        let m = &self.columns;
        [
            m[0], m[4], m[8], m[12], m[1], m[5], m[9], m[13], m[2], m[6], m[10], m[14],
        ]
    }
}

#[derive(Debug, Clone)]
pub struct Node {
    pub name: String,
    pub parent: Option<usize>,
    pub local: Mat4,
    /// Object archive this node instances. Empty when the node is only a transform.
    pub object: Option<String>,
    pub object_slot: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct SceneGraph {
    nodes: Vec<Node>,
    moved: bool,
}

impl SceneGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            moved: false,
        }
    }

    pub fn add_node(&mut self, node: Node) -> usize {
        self.moved = true;
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub fn nodes_mut(&mut self) -> &mut [Node] {
        &mut self.nodes
    }

    pub fn world(&self, index: usize) -> Mat4 {
        let node = &self.nodes[index];
        match node.parent {
            Some(parent) => self.world(parent).mul(node.local),
            None => node.local,
        }
    }

    pub fn set_local(&mut self, index: usize, local: Mat4) {
        self.nodes[index].local = local;
        self.moved = true;
    }

    pub fn take_moved(&mut self) -> bool {
        let moved = self.moved;
        self.moved = false;
        moved
    }

    pub fn mark_moved(&mut self) {
        self.moved = true;
    }
}

impl Default for SceneGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// A node whose world matrix changed must rebuild the spatial hash. Slot ids
/// stay put; only the world centers move.
pub fn transforms_dirty(previous: &Mat4, next: &Mat4) -> bool {
    previous.columns != next.columns
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn child_world_is_parent_times_local() {
        let mut scene = SceneGraph::new();
        let parent = scene.add_node(Node {
            name: "parent".into(),
            parent: None,
            local: Mat4::translation(1.0, 0.0, 0.0),
            object: None,
            object_slot: None,
        });
        let child = scene.add_node(Node {
            name: "child".into(),
            parent: Some(parent),
            local: Mat4::translation(0.0, 2.0, 0.0),
            object: None,
            object_slot: None,
        });
        let world = scene.world(child);
        let point = world.transform_point([0.0, 0.0, 0.0]);
        assert!((point[0] - 1.0).abs() < 1e-5);
        assert!((point[1] - 2.0).abs() < 1e-5);
        assert!(point[2].abs() < 1e-5);
    }

    #[test]
    fn local_center_times_matrix_is_world_and_a_move_dirties_the_index() {
        let rest = Mat4::identity();
        let moved = Mat4::translation(4.0, 0.0, 0.0);
        let local = [1.0, 2.0, 3.0];
        let world = moved.transform_point(local);
        assert!((world[0] - 5.0).abs() < 1e-5);
        assert!((world[1] - 2.0).abs() < 1e-5);
        assert!((world[2] - 3.0).abs() < 1e-5);
        assert!(transforms_dirty(&rest, &moved));
        assert!(!transforms_dirty(&moved, &moved));
    }
}
