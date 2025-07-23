pub mod core;
pub mod rendering;

use rust_embed::*;

#[derive(Embed)]
#[folder = "embed/"]
struct EmbeddedAssets;
