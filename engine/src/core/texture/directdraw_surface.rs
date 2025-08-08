use core::ffi;

pub const DDS_PIXELFORMAT_FLAG_FOURCC: u32 = 0x00000004;
pub const DDS_PIXELFORMAT_FLAG_DDS_RGB: u32 = 0x00000040;
pub const DDS_PIXELFORMAT_FLAG_DDS_RGBA: u32 = 0x00000041;
pub const DDS_PIXELFORMAT_FLAG_DDS_LUMINANCE: u32 = 0x00020000;
pub const DDS_PIXELFORMAT_FLAG_DDS_LUMINANCEA: u32 = 0x00020001;
pub const DDS_PIXELFORMAT_FLAG_DDS_ALPHA: u32 = 0x00000002;
pub const DDS_PIXELFORMAT_FLAG_DDS_PAL8: u32 = 0x00000020;

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
        ((self.ddspf.flags & DDS_PIXELFORMAT_FLAG_FOURCC) != 0u32) && (self.ddspf.four_cc == 0x30315844u32)
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

    fn make_fourcc(ch0: std::ffi::c_char, ch1: std::ffi::c_char, ch2: std::ffi::c_char, ch3: std::ffi::c_char) -> u32 {
        (ch0 as u32) | ((ch1 as u32) << 8u32) | ((ch2 as u32) << 16) | ((ch3 as u32) << 24)
    }

    pub fn vulkan_format(&self) -> vulkan_framework::ash::vk::Format {
        if (self.header.ddspf.flags & DDS_PIXELFORMAT_FLAG_DDS_RGB) != 0 {
            match self.header.ddspf.rgb_bit_count {
                32 => todo!(),
                24 => todo!(),
                16 => todo!(),
                _ => panic!("Invalid DDS data"),
            }
        } else if (self.header.ddspf.flags & DDS_PIXELFORMAT_FLAG_DDS_LUMINANCE) != 0 {
            // TODO: missing bits here
            match self.header.ddspf.rgb_bit_count {
                8 => vulkan_framework::ash::vk::Format::R8_UNORM,
                _ => panic!("Invalid DDS data"),
            }
        } else if (self.header.ddspf.flags & DDS_PIXELFORMAT_FLAG_DDS_ALPHA) != 0 {
            match self.header.ddspf.rgb_bit_count {
                8 => vulkan_framework::ash::vk::Format::A8_UNORM_KHR,
                _ => panic!("Invalid DDS data"),
            }
        } else if (self.header.ddspf.flags & DDS_PIXELFORMAT_FLAG_FOURCC) != 0 {
            if self.header.ddspf.four_cc == Self::make_fourcc('D' as i8, 'X' as i8, 'T' as i8, '1' as i8) {
                vulkan_framework::ash::vk::Format::BC1_RGBA_UNORM_BLOCK
            } else if self.header.ddspf.four_cc == Self::make_fourcc('D' as i8, 'X' as i8, '1' as i8, '0' as i8) {
                todo!()
            } else {
                panic!("Unable to understand format")
            }
        } else {
            panic!("Unrecognised format")
        }
    }
}
