#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ColorClearValues {
    Vec4(f32, f32, f32, f32),
    IVec4(i32, i32, i32, i32),
    UVec4(u32, u32, u32, u32),
}

impl From<&ColorClearValues> for crate::ash::vk::ClearColorValue {
    fn from(val: &ColorClearValues) -> Self {
        let mut result = crate::ash::vk::ClearColorValue::default();
        match val {
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

impl From<ColorClearValues> for crate::ash::vk::ClearColorValue {
    fn from(val: ColorClearValues) -> Self {
        (&val).into()
    }
}

impl From<ColorClearValues> for crate::ash::vk::ClearValue {
    fn from(val: ColorClearValues) -> Self {
        crate::ash::vk::ClearValue { color: val.into() }
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

impl From<&DepthClearValues> for crate::ash::vk::ClearDepthStencilValue {
    fn from(val: &DepthClearValues) -> Self {
        crate::ash::vk::ClearDepthStencilValue::default().depth(val.depth)
    }
}

impl From<DepthClearValues> for crate::ash::vk::ClearDepthStencilValue {
    fn from(val: DepthClearValues) -> Self {
        (&val).into()
    }
}

impl From<DepthClearValues> for crate::ash::vk::ClearValue {
    fn from(val: DepthClearValues) -> Self {
        crate::ash::vk::ClearValue {
            depth_stencil: val.into(),
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

impl From<&StencilClearValues> for crate::ash::vk::ClearDepthStencilValue {
    fn from(val: &StencilClearValues) -> Self {
        crate::ash::vk::ClearDepthStencilValue::default().stencil(val.stencil)
    }
}

impl From<StencilClearValues> for crate::ash::vk::ClearDepthStencilValue {
    fn from(val: StencilClearValues) -> Self {
        (&val).into()
    }
}

impl From<StencilClearValues> for crate::ash::vk::ClearValue {
    fn from(val: StencilClearValues) -> Self {
        crate::ash::vk::ClearValue {
            depth_stencil: val.into(),
        }
    }
}
