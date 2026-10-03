pub mod core;
pub mod preview;
pub mod rendering;
pub mod scene;

#[cfg(test)]
pub mod tests;

use rust_embed::*;

#[derive(Embed)]
#[folder = "embed/"]
struct EmbeddedAssets;
