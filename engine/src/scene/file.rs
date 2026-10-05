use std::{fs, path::Path};

use serde::{Deserialize, Serialize};

use super::{Mat4, Node, SceneGraph};

#[derive(Debug, Deserialize)]
struct SceneFile {
    nodes: Vec<SceneNodeFile>,
}

#[derive(Debug, Deserialize)]
struct SceneNodeFile {
    name: String,
    #[serde(default)]
    parent: Option<String>,
    #[serde(default)]
    translation: [f32; 3],
    #[serde(default)]
    object: Option<String>,
}

pub fn load_scene_json(text: &str) -> Result<SceneGraph, String> {
    let file: SceneFile = serde_json::from_str(text).map_err(|err| err.to_string())?;
    let mut scene = SceneGraph::new();
    let mut names = Vec::new();
    for node in &file.nodes {
        names.push(node.name.clone());
        let local = Mat4::translation(node.translation[0], node.translation[1], node.translation[2]);
        scene.add_node(Node {
            name: node.name.clone(),
            parent: None,
            local,
            object: node.object.clone(),
            object_slot: None,
        });
    }
    for (index, node) in file.nodes.iter().enumerate() {
        if let Some(parent_name) = &node.parent {
            let parent = names
                .iter()
                .position(|name| name == parent_name)
                .ok_or_else(|| format!("scene node '{}' has no parent '{parent_name}'", node.name))?;
            scene.nodes_mut()[index].parent = Some(parent);
        }
    }
    Ok(scene)
}

#[derive(Serialize)]
struct SceneFileOut {
    nodes: Vec<SceneNodeFileOut>,
}

#[derive(Serialize)]
struct SceneNodeFileOut {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    parent: Option<String>,
    translation: [f32; 3],
    #[serde(skip_serializing_if = "Option::is_none")]
    object: Option<String>,
}

pub fn save_scene_json(scene: &SceneGraph, path: &Path) -> Result<(), String> {
    let nodes = scene.nodes();
    let mut out = Vec::new();
    for (index, node) in nodes.iter().enumerate() {
        let parent = node.parent.map(|parent_index| nodes[parent_index].name.clone());
        let translation = node.local.transform_point([0.0, 0.0, 0.0]);
        out.push(SceneNodeFileOut {
            name: node.name.clone(),
            parent,
            translation,
            object: node.object.clone(),
        });
    }
    let text = serde_json::to_string_pretty(&SceneFileOut { nodes: out })
        .map_err(|err| err.to_string())?;
    fs::write(path, text).map_err(|err| err.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_parent_translation_matches_the_world_matrix() {
        let scene = load_scene_json(
            r#"{
                "nodes": [
                    {"name": "root", "translation": [1, 0, 0]},
                    {"name": "child", "parent": "root", "translation": [0, 2, 0], "object": "mesh.tar"}
                ]
            }"#,
        )
        .unwrap();
        let world = scene.world(1).transform_point([0.0, 0.0, 0.0]);
        assert!((world[0] - 1.0).abs() < 1e-5);
        assert!((world[1] - 2.0).abs() < 1e-5);
        assert_eq!(scene.nodes()[1].object.as_deref(), Some("mesh.tar"));
    }
}
