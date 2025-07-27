use std::{collections::HashMap, io::Read, path::PathBuf, sync::Arc};

use tar::Archive;

use crate::{
    EmbeddedAssets,
    rendering::{MAX_TEXTURES, RenderingResult, resources::ResourceError},
};

use super::{mesh::MeshManager, texture::TextureManager};

use vulkan_framework::{
    buffer::{AllocatedBuffer, Buffer, BufferUsage, BufferUseAs, ConcreteBufferDescriptor},
    device::{Device, DeviceOwned},
    image::{CommonImageFormat, Image2DDimensions, ImageFormat},
    memory_allocator::DefaultAllocator,
    memory_heap::{ConcreteMemoryHeapDescriptor, MemoryHeap, MemoryHostVisibility, MemoryType},
    memory_pool::{MemoryMap, MemoryPool, MemoryPoolBacked, MemoryPoolFeature, MemoryPoolFeatures},
    queue_family::QueueFamily,
};

pub struct Manager {
    device: Arc<Device>,
    queue_family: Arc<QueueFamily>,

    memory_pool: Arc<MemoryPool>,

    mesh_manager: MeshManager,
    texture_manager: TextureManager,
}

impl Manager {
    fn memory_pool_size(frames_in_flight: u32) -> u64 {
        (1024u64 * 1024u64 * 256u64) + (frames_in_flight as u64 * 8192u64)
    }

    pub fn new(queue_family: Arc<QueueFamily>, frames_in_flight: u32) -> RenderingResult<Self> {
        let device = queue_family.get_parent_device();

        let total_size = Self::memory_pool_size(frames_in_flight);

        let buffer = Buffer::new(
            device.clone(),
            ConcreteBufferDescriptor::new(
                BufferUsage::from([BufferUseAs::TransferSrc].as_slice()),
                1024u64 * 1024u64 * 24u64, // 24MiB
            ),
            None,
            Some("resource_management.stub_image_buffer"),
        )?;

        let memory_heap = MemoryHeap::new(
            device.clone(),
            ConcreteMemoryHeapDescriptor::new(
                MemoryType::DeviceLocal(Some(MemoryHostVisibility::MemoryHostVisibile {
                    cached: false,
                })),
                total_size,
            ),
            &[&buffer],
        )?;

        let memory_pool = MemoryPool::new(
            memory_heap,
            Arc::new(DefaultAllocator::new(total_size)),
            MemoryPoolFeatures::from([MemoryPoolFeature::DeviceAddressable {}].as_slice()),
        )?;

        let buffer = AllocatedBuffer::new(memory_pool.clone(), buffer)?;

        memory_pool.write_raw_data(
            buffer.allocation_offset(),
            EmbeddedAssets::get("stub.dds").unwrap().data.as_ref(),
        )?;

        let mesh_manager = MeshManager::new(device.clone(), frames_in_flight)?;
        let texture_manager = TextureManager::new(
            queue_family.clone(),
            buffer.clone(),
            MAX_TEXTURES,
            frames_in_flight,
        )?;

        Ok(Self {
            device,
            queue_family,

            memory_pool,

            mesh_manager,
            texture_manager,
        })
    }

    pub fn load_object(&mut self, file: PathBuf) -> RenderingResult<()> {
        #[derive(Default, Clone)]
        struct TextureDecl {
            width: Option<u32>,
            height: Option<u32>,
            miplevel: Option<u32>,
            data: Option<Arc<AllocatedBuffer>>,
        }

        #[derive(Default, Clone)]
        struct MaterialDecl {
            diffuse_texture: Option<String>,
            reflection_texture: Option<String>,
            displacement_texture: Option<String>,
            normal_texture: Option<String>,
        }

        #[derive(Default, Clone)]
        struct ModelDecl {
            material_name: Option<String>,
            indexes: Option<Vec<u32>>,
        }

        let mut textures: HashMap<String, TextureDecl> = HashMap::new();

        if !file.exists() {
            panic!("File doesn't exists!");
        }

        if !file.is_file() {
            panic!("Not a file");
        }

        let file = std::fs::File::open(file).unwrap();
        let mut archive = Archive::new(file);

        for file in archive.entries()? {
            let mut file = file?;

            let path = file.header().path()?;

            let path_str = path.to_string_lossy();
            let path_trimmed = match path_str.starts_with("./") {
                true => String::from(&path_str[2..]),
                false => String::from(path_str),
            };
            let splitted_path = path_trimmed.split('/').map(|val| val).collect::<Vec<_>>();
            if splitted_path.len() < 1 {
                println!("Invalid file name {:?}", file.header().path()?);
            }

            let obj_type = *(splitted_path.get(0).unwrap());
            match splitted_path.get(1) {
                Some(obj_name) => match obj_type {
                    "textures" => {
                        let texture_name = String::from(*obj_name);

                        let Some(property) = splitted_path.get(2) else {
                            // this is the directory definition
                            continue;
                        };

                        let mut texture_decl = match textures.get(&texture_name) {
                            Some(decl) => decl.clone(),
                            None => TextureDecl::default(),
                        };

                        match *property {
                            "data" => {
                                let texture_size = file.header().size()?;
                                // Allocate the buffer that will be used to upload the vertex data to the vulkan device
                                let buffer = Buffer::new(
                                    self.device.clone(),
                                    ConcreteBufferDescriptor::new(
                                        BufferUsage::from([BufferUseAs::TransferSrc].as_slice()),
                                        texture_size,
                                    ),
                                    None,
                                    Some("resource_management.vertex_buffer"),
                                )?;

                                println!("allocating texture size: {texture_size}");
                                let buffer =
                                    AllocatedBuffer::new(self.memory_pool.clone(), buffer).unwrap();

                                // Fill the buffer with actual data from the file
                                {
                                    let mut mem_map = MemoryMap::new(self.memory_pool.clone())?;
                                    let slice = mem_map.as_mut_slice::<u32>(
                                        buffer.clone() as Arc<dyn MemoryPoolBacked>
                                    )?;
                                    let len = slice.len() * std::mem::size_of::<u32>();
                                    let ptr = slice.as_mut_ptr() as *mut u8;
                                    let slice_u8 =
                                        unsafe { std::slice::from_raw_parts_mut(ptr, len) };
                                    file.read_exact(slice_u8).unwrap();
                                }

                                texture_decl.data.replace(buffer);
                            }
                            "height.txt" => {
                                let mut read_result = String::new();
                                let read_data = file.read_to_string(&mut read_result)?;
                                if read_data > 16 {
                                    panic!("invalid file!");
                                }

                                texture_decl
                                    .height
                                    .replace(read_result.parse::<u32>().unwrap());
                            }
                            "width.txt" => {
                                let mut read_result = String::new();
                                let read_data = file.read_to_string(&mut read_result)?;
                                if read_data > 16 {
                                    panic!("invalid file!");
                                }

                                texture_decl
                                    .width
                                    .replace(read_result.parse::<u32>().unwrap());
                            }
                            "miplevel.txt" => {
                                let mut read_result = String::new();
                                let read_data = file.read_to_string(&mut read_result)?;
                                if read_data > 16 {
                                    panic!("invalid file!");
                                }

                                texture_decl
                                    .miplevel
                                    .replace(read_result.parse::<u32>().unwrap());
                            }
                            "" => continue,
                            _ => println!(
                                "WARNING: unrecognised property for texture {texture_name}: {property}"
                            ),
                        };

                        textures.insert(texture_name, texture_decl);

                        print!("@");
                    }
                    "materials" => {
                        print!(".");
                    }
                    "models" => {
                        print!("-");
                    }
                    _ => {
                        print!("?");
                    }
                },
                None => match obj_type {
                    "vertex_buffer" => {
                        print!("!");

                        // Allocate the buffer that will be used to upload the vertex data to the vulkan device
                        let buffer = Buffer::new(
                            self.device.clone(),
                            ConcreteBufferDescriptor::new(
                                BufferUsage::from([BufferUseAs::TransferSrc].as_slice()),
                                file.header().size()?,
                            ),
                            None,
                            Some("resource_management.vertex_buffer"),
                        )?;

                        let buffer = AllocatedBuffer::new(self.memory_pool.clone(), buffer)?;

                        // Fill the buffer with actual data from the file
                        {
                            let mut mem_map = MemoryMap::new(self.memory_pool.clone())?;
                            let slice =
                                mem_map.as_mut_slice::<u32>(buffer as Arc<dyn MemoryPoolBacked>)?;
                            let len = slice.len() * std::mem::size_of::<u32>();
                            let ptr = slice.as_mut_ptr() as *mut u8;
                            let slice_u8 = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
                            file.read_exact(slice_u8).unwrap();
                        }
                    }
                    _ => {}
                },
            };
        }

        println!("");

        for (k, v) in textures.iter() {
            match (&v.width, &v.height, &v.miplevel, &v.data) {
                (Some(width), Some(height), Some(miplevel), Some(data)) => {
                    let texture_id = self.texture_manager.load(
                        ImageFormat::from(CommonImageFormat::bc7_srgb_block),
                        Image2DDimensions::new(*width, *height),
                        *miplevel,
                        data.clone(),
                    )?;

                    println!("Loaded texture {k} at id {texture_id}");
                }
                _ => {
                    return Err(crate::rendering::RenderingError::ResourceError(
                        ResourceError::IncompleteTexture(k.clone()),
                    ));
                }
            };
        }

        println!();

        Ok(())
    }
}
