use vulkan_framework::image::Image2DDimensions;

/// Represents dimensions of the actual rendering target,
/// NOT the dimensions of the image that has to be presented.
pub struct RenderingDimensions {
    width: u32,
    height: u32,
}

impl Into<Image2DDimensions> for &RenderingDimensions {
    fn into(self) -> Image2DDimensions {
        Image2DDimensions::new(self.width.to_owned(), self.height.to_owned())
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
