pub mod core;
pub mod rendering;

#[cfg(test)]
pub mod tests;

use rust_embed::*;

#[derive(Embed)]
#[folder = "embed/"]
struct EmbeddedAssets;
