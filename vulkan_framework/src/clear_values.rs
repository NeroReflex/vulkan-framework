#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ColorClearValues {
    Vec4(f32, f32, f32, f32),
    IVec4(i32, i32, i32, i32),
    UVec4(u32, u32, u32, u32),
}

impl Into<crate::ash::vk::ClearColorValue> for &ColorClearValues {
    fn into(self) -> crate::ash::vk::ClearColorValue {
        let mut result = crate::ash::vk::ClearColorValue::default();
        match self {
            ColorClearValues::Vec4(r, g, b, a) => {
                result.float32 = [r.to_owned(), g.to_owned(), b.to_owned(), a.to_owned()];
            }
            ColorClearValues::IVec4(r, g, b, a) => {
                result.int32 = [r.to_owned(), g.to_owned(), b.to_owned(), a.to_owned()];
            }
            ColorClearValues::UVec4(r, g, b, a) => {
                result.uint32 = [r.to_owned(), g.to_owned(), b.to_owned(), a.to_owned()];
            }
        }

        result
    }
}

impl Into<crate::ash::vk::ClearColorValue> for ColorClearValues {
    fn into(self) -> crate::ash::vk::ClearColorValue {
        (&self).into()
    }
}

impl Into<crate::ash::vk::ClearValue> for ColorClearValues {
    fn into(self) -> crate::ash::vk::ClearValue {
        crate::ash::vk::ClearValue { color: self.into() }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct DepthClearValues {
    depth: f32,
}

impl DepthClearValues {
    pub fn new(depth: f32) -> Self {
        Self { depth }
    }
}

impl Into<crate::ash::vk::ClearDepthStencilValue> for &DepthClearValues {
    fn into(self) -> crate::ash::vk::ClearDepthStencilValue {
        crate::ash::vk::ClearDepthStencilValue::default().depth(self.depth)
    }
}

impl Into<crate::ash::vk::ClearDepthStencilValue> for DepthClearValues {
    fn into(self) -> crate::ash::vk::ClearDepthStencilValue {
        (&self).into()
    }
}

impl Into<crate::ash::vk::ClearValue> for DepthClearValues {
    fn into(self) -> crate::ash::vk::ClearValue {
        crate::ash::vk::ClearValue {
            depth_stencil: self.into(),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct StencilClearValues {
    stencil: u32,
}

impl StencilClearValues {
    pub fn new(stencil: u32) -> Self {
        Self { stencil }
    }
}

impl Into<crate::ash::vk::ClearDepthStencilValue> for &StencilClearValues {
    fn into(self) -> crate::ash::vk::ClearDepthStencilValue {
        crate::ash::vk::ClearDepthStencilValue::default().stencil(self.stencil)
    }
}

impl Into<crate::ash::vk::ClearDepthStencilValue> for StencilClearValues {
    fn into(self) -> crate::ash::vk::ClearDepthStencilValue {
        (&self).into()
    }
}

impl Into<crate::ash::vk::ClearValue> for StencilClearValues {
    fn into(self) -> crate::ash::vk::ClearValue {
        crate::ash::vk::ClearValue {
            depth_stencil: self.into(),
        }
    }
}
