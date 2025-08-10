use vulkan_framework::image::{Image2DDimensions, ImageDimensions};

/// Represents dimensions of the actual rendering target,
/// NOT the dimensions of the image that has to be presented.
pub struct RenderingDimensions {
    width: u32,
    height: u32,
}

impl From<&RenderingDimensions> for Image2DDimensions {
    fn from(val: &RenderingDimensions) -> Self {
        Image2DDimensions::new(val.width.to_owned(), val.height.to_owned())
    }
}

impl From<&RenderingDimensions> for ImageDimensions {
    fn from(val: &RenderingDimensions) -> Self {
        ImageDimensions::Image2D { extent: val.into() }
    }
}

impl RenderingDimensions {
    #[inline]
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    #[inline]
    pub fn width(&self) -> u32 {
        self.width.to_owned()
    }

    #[inline]
    pub fn height(&self) -> u32 {
        self.height.to_owned()
    }
}
