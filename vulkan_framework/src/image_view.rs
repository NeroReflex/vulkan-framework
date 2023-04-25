use ash::vk::{Handle, ImageAspectFlags};

use crate::{
    device::DeviceOwned,
    image::*,
    instance::InstanceOwned,
    prelude::{VulkanError, VulkanResult},
};

use std::sync::Arc;

#[derive(Copy, Clone)]
pub enum ImageViewType {
    Image1D,
    Image2D,
    Image3D,
    Image1DArray,
    Image2DArray,
    CubeMap,
    CubeMapArray,
}

impl ImageViewType {
    pub(crate) fn ash_viewtype(&self) -> ash::vk::ImageViewType {
        match self {
            Self::Image1D => ash::vk::ImageViewType::TYPE_1D,
            Self::Image2D => ash::vk::ImageViewType::TYPE_2D,
            Self::Image3D => ash::vk::ImageViewType::TYPE_3D,
            Self::Image1DArray => ash::vk::ImageViewType::TYPE_1D_ARRAY,
            Self::Image2DArray => ash::vk::ImageViewType::TYPE_2D_ARRAY,
            Self::CubeMap => ash::vk::ImageViewType::CUBE,
            Self::CubeMapArray => ash::vk::ImageViewType::CUBE_ARRAY,
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
pub enum ImageViewColorMapping {
    rgba_rgba,
    bgra_rgba,
}

pub struct ImageView {
    image: Arc<dyn ImageTrait>,
    image_view: ash::vk::ImageView,
    view_type: ImageViewType,
    color_mapping: ImageViewColorMapping,
    subrange_base_mip_level: u32,
    subrange_level_count: u32,
    subrange_base_array_layer: u32,
    subrange_layer_count: u32,
}

#[derive(Copy, Clone)]
pub struct RecognisedImageAspect {
    color: bool,
    depth: bool,
    stencil: bool,
    metadata: bool,
}

impl RecognisedImageAspect {
    pub fn color(&self) -> bool {
        self.color
    }

    pub fn depth(&self) -> bool {
        self.depth
    }

    pub fn stencil(&self) -> bool {
        self.stencil
    }

    pub fn metadata(&self) -> bool {
        self.metadata
    }

    pub fn from_format(img_format: &ImageFormat) -> Option<Self> {
        match img_format {
            ImageFormat::s8_uint => Some(Self::new(false, false, true, false)),
            ImageFormat::d16_unorm => Some(Self::new(false, true, false, false)),
            ImageFormat::d32_sfloat => Some(Self::new(false, true, false, false)),
            ImageFormat::x8_d24_unorm_pack32 => Some(Self::new(true, true, false, false)),
            ImageFormat::d32_sfloat_s8_uint => Some(Self::new(false, true, true, false)),
            ImageFormat::d16_unorm_s8_uint => Some(Self::new(false, true, true, false)),
            ImageFormat::d24_unorm_s8_uint => Some(Self::new(false, true, true, false)),
            _ => None,
        }
    }

    pub fn new(color: bool, depth: bool, stencil: bool, metadata: bool) -> Self {
        Self {
            color,
            depth,
            stencil,
            metadata,
        }
    }
}

#[derive(Copy, Clone)]
pub enum ImageViewAspect {
    Recognised(RecognisedImageAspect),
    Other(u32),
}

impl ImageViewAspect {
    pub fn ash_aspect(&self) -> ash::vk::ImageAspectFlags {
        match self {
            Self::Recognised(fmt) => {
                (match fmt.color() {
                    true => ImageAspectFlags::COLOR,
                    false => ImageAspectFlags::from_raw(0u32),
                }) | (match fmt.depth() {
                    true => ImageAspectFlags::DEPTH,
                    false => ImageAspectFlags::from_raw(0u32),
                }) | (match fmt.stencil() {
                    true => ImageAspectFlags::STENCIL,
                    false => ImageAspectFlags::from_raw(0u32),
                }) | (match fmt.metadata() {
                    true => ImageAspectFlags::METADATA,
                    false => ImageAspectFlags::from_raw(0u32),
                })
            }
            Self::Other(raw) => ImageAspectFlags::from_raw(*raw),
        }
    }

    pub fn from_format(img_format: &ImageFormat) -> Self {
        match RecognisedImageAspect::from_format(img_format) {
            Some(fmt) => Self::Recognised(fmt),
            None => Self::Recognised(RecognisedImageAspect::new(true, false, false, false)),
        }
    }

    pub fn from_raw(raw: u32) -> Self {
        Self::Other(raw)
    }
}

impl ImageOwned for ImageView {
    fn get_parent_image(&self) -> Arc<dyn ImageTrait> {
        self.image.clone()
    }
}

impl Drop for ImageView {
    fn drop(&mut self) {
        let device = self.get_parent_image().get_parent_device();

        unsafe {
            device.ash_handle().destroy_image_view(
                self.image_view,
                device.get_parent_instance().get_alloc_callbacks(),
            )
        }
    }
}

impl ImageView {
    pub fn view_type(&self) -> ImageViewType {
        self.view_type
    }

    pub fn color_mapping(&self) -> ImageViewColorMapping {
        self.color_mapping
    }

    pub fn subrange_base_mip_level(&self) -> u32 {
        self.subrange_base_mip_level
    }

    pub fn subrange_level_count(&self) -> u32 {
        self.subrange_level_count
    }

    pub fn subrange_base_array_layer(&self) -> u32 {
        self.subrange_base_array_layer
    }

    pub fn subrange_layer_count(&self) -> u32 {
        self.subrange_layer_count
    }

    pub fn native_handle(&self) -> u64 {
        ash::vk::Handle::as_raw(self.image_view)
    }

    /**
     * Create an image view for the given image.
     *
     * General rule: options that are not provided are automatically detected from the image.
     *
     * If the aspect is not specified is inferred from the given image format, or from the image format:
     * this is to allow to create an image view with a format different from the image's own format.
     */
    pub fn new(
        image: Arc<dyn ImageTrait>,
        view_type: ImageViewType,
        maybe_format: Option<ImageFormat>,
        maybe_aspect: Option<ImageViewAspect>,
        maybe_specified_color_mapping: Option<ImageViewColorMapping>,
        maybe_specified_subrange_base_mip_level: Option<u32>,
        maybe_specified_subrange_level_count: Option<u32>,
        maybe_specified_subrange_base_array_layer: Option<u32>,
        maybe_specified_subrange_layer_count: Option<u32>,
        debug_name: Option<&str>,
    ) -> VulkanResult<Arc<Self>> {
        // by default do not swizzle colors
        let color_mapping =
            maybe_specified_color_mapping.unwrap_or(ImageViewColorMapping::rgba_rgba);
        let subrange_base_mip_level = maybe_specified_subrange_base_mip_level.unwrap_or(0);
        let subrange_level_count =
            maybe_specified_subrange_level_count.unwrap_or(image.mip_levels_count());
        let subrange_base_array_layer = maybe_specified_subrange_base_array_layer.unwrap_or(0);
        let subrange_layer_count = maybe_specified_subrange_layer_count.unwrap_or(1);

        let format = maybe_format.unwrap_or(image.format());

        let device = image.get_parent_device();

        let srr = ash::vk::ImageSubresourceRange {
            aspect_mask: match maybe_aspect {
                Some(user_given) => user_given.ash_aspect(),
                None => ImageViewAspect::from_format(&format).ash_aspect(),
            },
            base_mip_level: subrange_base_mip_level,
            level_count: subrange_level_count,
            base_array_layer: subrange_base_array_layer,
            layer_count: subrange_layer_count,
        };

        let create_info = ash::vk::ImageViewCreateInfo::builder()
            .image(ash::vk::Image::from_raw(image.native_handle()))
            .format(image.format().ash_format())
            .subresource_range(srr)
            .view_type(view_type.ash_viewtype())
            .build();

        match unsafe {
            device.ash_handle().create_image_view(
                &create_info,
                device.get_parent_instance().get_alloc_callbacks(),
            )
        } {
            Ok(image_view) => {
                let mut obj_name_bytes = vec![];
                match device.get_parent_instance().get_debug_ext_extension() {
                    Some(ext) => {
                        match debug_name {
                            Some(name) => {
                                for name_ch in name.as_bytes().iter() {
                                    obj_name_bytes.push(*name_ch);
                                }
                                obj_name_bytes.push(0x00);

                                unsafe {
                                    let object_name = std::ffi::CStr::from_bytes_with_nul_unchecked(
                                        obj_name_bytes.as_slice(),
                                    );
                                    // set device name for debugging
                                    let dbg_info = ash::vk::DebugUtilsObjectNameInfoEXT::builder()
                                        .object_type(ash::vk::ObjectType::IMAGE_VIEW)
                                        .object_handle(ash::vk::Handle::as_raw(image_view))
                                        .object_name(object_name)
                                        .build();

                                    match ext.set_debug_utils_object_name(
                                        device.ash_handle().handle(),
                                        &dbg_info,
                                    ) {
                                        Ok(_) => {
                                            #[cfg(debug_assertions)]
                                            {
                                                println!("Queue Debug object name changed");
                                            }
                                        }
                                        Err(err) => {
                                            #[cfg(debug_assertions)]
                                            {
                                                panic!("Error setting the Debug name for the newly created Queue, will use handle. Error: {}", err)
                                            }
                                        }
                                    }
                                }
                            }
                            None => {}
                        };
                    }
                    None => {}
                }

                Ok(Arc::new(Self {
                    image,
                    image_view,
                    view_type,
                    color_mapping,
                    subrange_base_mip_level,
                    subrange_level_count,
                    subrange_base_array_layer,
                    subrange_layer_count,
                }))
            }
            Err(err) => {
                #[cfg(debug_assertions)]
                {
                    panic!("Error creating the specified image view: {}", err)
                }

                Err(VulkanError::Unspecified)
            }
        }
    }
}
