pub const DDS_PIXELFORMAT_FLAG_FOURCC: u32 = 0x00000004;
pub const DDS_PIXELFORMAT_FLAG_DDS_RGB: u32 = 0x00000040;
pub const DDS_PIXELFORMAT_FLAG_DDS_RGBA: u32 = 0x00000041;
pub const DDS_PIXELFORMAT_FLAG_DDS_LUMINANCE: u32 = 0x00020000;
pub const DDS_PIXELFORMAT_FLAG_DDS_LUMINANCEA: u32 = 0x00020001;
pub const DDS_PIXELFORMAT_FLAG_DDS_ALPHA: u32 = 0x00000002;
pub const DDS_PIXELFORMAT_FLAG_DDS_PAL8: u32 = 0x00000020;

pub const DDS_FOURCC_DX10: u32 = 0x30315844;
pub const DDS_FOURCC_DXT1: u32 = 0x31347844;
pub const DDS_FOURCC_DXT2: u32 = 0x32347844;
pub const DDS_FOURCC_DXT3: u32 = 0x33347844;
pub const DDS_FOURCC_DXT4: u32 = 0x34347844;
pub const DDS_FOURCC_DXT5: u32 = 0x35347844;
pub const DDS_FOURCC_ATI1: u32 = 0x31394941;
pub const DDS_FOURCC_ATI2: u32 = 0x32394941;
pub const DDS_FOURCC_BC4U: u32 = 0x55344342;
pub const DDS_FOURCC_BC4S: u32 = 0x53344342;
pub const DDS_FOURCC_BC5U: u32 = 0x55354342;
pub const DDS_FOURCC_BC5S: u32 = 0x53354342;
pub const DDS_FOURCC_RGBG: u32 = 0x47424752;

#[repr(C)]
pub struct DDSPixelFormat {
    size: u32,
    flags: u32,
    four_cc: u32,
    rgb_bit_count: u32,
    rb_bit_mask: u32,
    gb_bit_mask: u32,
    bb_bit_mask: u32,
    ab_bit_mask: u32,
}

#[repr(u32)]
pub enum DXGIFormat {
    Unknown = 0,
    BC7Typeless = 97,
    BC7Unorm = 98,
    BC7UnormSrgb = 99,
}

#[repr(u32)]
pub enum D3D10ResourceDimension {
    Unknown = 0,
    Buffer = 1,
    Texture1D = 2,
    Texture2D = 3,
    Texture3D = 4,
}

#[repr(C)]
pub struct DDSHeaderDXT10 {
    format: DXGIFormat,
    dimensions: D3D10ResourceDimension,
    flags: u32,
    array_size: u32,
    flags2: u32,
}

#[repr(C)]
pub struct DDSHeader {
    size: u32,
    flags: u32,
    height: u32,
    width: u32,
    pitch_or_linear_size: u32,
    depth: u32,
    mip_map_count: u32,
    reserved1: [u32; 11],
    ddspf: DDSPixelFormat,
    caps: u32,
    caps2: u32,
    caps3: u32,
    caps4: u32,
    reserved2: u32,
}

impl DDSHeader {
    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn mip_map_count(&self) -> u32 {
        self.mip_map_count
    }

    pub fn is_followed_by_dxt10_header(&self) -> bool {
        ((self.ddspf.flags & DDS_PIXELFORMAT_FLAG_FOURCC) != 0u32)
            && (self.ddspf.four_cc == DDS_FOURCC_DX10)
    }
}

pub struct DirectDrawSurface {
    header: DDSHeader,
    dxt10_header: Option<DDSHeaderDXT10>,
}

impl DirectDrawSurface {
    pub fn new(header: DDSHeader, dxt10_header: Option<DDSHeaderDXT10>) -> Self {
        Self {
            header,
            dxt10_header,
        }
    }
}

impl DirectDrawSurface {
    pub fn width(&self) -> u32 {
        self.header.width()
    }

    pub fn height(&self) -> u32 {
        self.header.height()
    }

    pub fn mip_map_count(&self) -> u32 {
        self.header.mip_map_count()
    }

    pub fn vulkan_format(&self) -> vulkan_framework::ash::vk::Format {
        if (self.header.ddspf.flags & DDS_PIXELFORMAT_FLAG_DDS_RGB) != 0 {
            match self.header.ddspf.rgb_bit_count {
                32 => todo!(),
                24 => todo!(),
                16 => todo!(),
                _ => vulkan_framework::ash::vk::Format::UNDEFINED,
            }
        } else if (self.header.ddspf.flags & DDS_PIXELFORMAT_FLAG_DDS_LUMINANCE) != 0 {
            // TODO: missing bits here
            match self.header.ddspf.rgb_bit_count {
                8 => vulkan_framework::ash::vk::Format::R8_UNORM,
                _ => vulkan_framework::ash::vk::Format::UNDEFINED,
            }
        } else if (self.header.ddspf.flags & DDS_PIXELFORMAT_FLAG_DDS_ALPHA) != 0 {
            match self.header.ddspf.rgb_bit_count {
                8 => vulkan_framework::ash::vk::Format::A8_UNORM_KHR,
                _ => vulkan_framework::ash::vk::Format::UNDEFINED,
            }
        } else if (self.header.ddspf.flags & DDS_PIXELFORMAT_FLAG_FOURCC) != 0 {
            match self.header.ddspf.four_cc {
                DDS_FOURCC_DXT1 => vulkan_framework::ash::vk::Format::BC1_RGBA_UNORM_BLOCK,
                DDS_FOURCC_DXT2 => vulkan_framework::ash::vk::Format::BC1_RGBA_UNORM_BLOCK,
                DDS_FOURCC_DXT3 => vulkan_framework::ash::vk::Format::BC1_RGBA_UNORM_BLOCK,
                DDS_FOURCC_DXT4 => vulkan_framework::ash::vk::Format::BC1_RGBA_UNORM_BLOCK,
                DDS_FOURCC_DXT5 => vulkan_framework::ash::vk::Format::BC3_UNORM_BLOCK,
                DDS_FOURCC_ATI1 => vulkan_framework::ash::vk::Format::BC4_UNORM_BLOCK,
                DDS_FOURCC_ATI2 => vulkan_framework::ash::vk::Format::BC5_UNORM_BLOCK,
                DDS_FOURCC_BC4U => vulkan_framework::ash::vk::Format::BC4_UNORM_BLOCK,
                DDS_FOURCC_BC4S => vulkan_framework::ash::vk::Format::BC4_SNORM_BLOCK,
                DDS_FOURCC_BC5U => vulkan_framework::ash::vk::Format::BC5_UNORM_BLOCK,
                DDS_FOURCC_BC5S => vulkan_framework::ash::vk::Format::BC5_SNORM_BLOCK,
                DDS_FOURCC_RGBG => todo!(),
                DDS_FOURCC_DX10 => match &self.dxt10_header {
                    Some(dx10_header) => match dx10_header.format {
                        DXGIFormat::Unknown => vulkan_framework::ash::vk::Format::UNDEFINED,
                        DXGIFormat::BC7Typeless => {
                            vulkan_framework::ash::vk::Format::BC7_UNORM_BLOCK
                        }
                        DXGIFormat::BC7Unorm => vulkan_framework::ash::vk::Format::BC7_SRGB_BLOCK,
                        DXGIFormat::BC7UnormSrgb => {
                            vulkan_framework::ash::vk::Format::BC7_SRGB_BLOCK
                        }
                    },
                    None => panic!("Missing DDS DX10 header"),
                },
                36 => vulkan_framework::ash::vk::Format::R16G16B16A16_UNORM,
                110 => vulkan_framework::ash::vk::Format::R16G16B16A16_SNORM,
                111 => vulkan_framework::ash::vk::Format::R16_SFLOAT,
                112 => vulkan_framework::ash::vk::Format::R16G16_SFLOAT,
                113 => vulkan_framework::ash::vk::Format::R16G16B16_SFLOAT,
                114 => vulkan_framework::ash::vk::Format::R32_SFLOAT,
                115 => vulkan_framework::ash::vk::Format::R32G32_SFLOAT,
                116 => vulkan_framework::ash::vk::Format::R32G32B32A32_SFLOAT,
                _ => vulkan_framework::ash::vk::Format::UNDEFINED,
            }
        } else {
            vulkan_framework::ash::vk::Format::UNDEFINED
        }
    }
}
