use std::{collections::HashMap, io::Read, path::PathBuf, sync::Arc};

use tar::Archive;

use crate::{
    EmbeddedAssets,
    rendering::{
        MAX_FRAMES_IN_FLIGHT_NO_MALLOC, MAX_MATERIALS, MAX_TEXTURES, RenderingError,
        RenderingResult, resources::ResourceError,
    },
};

use super::{mesh::MeshManager, texture::TextureManager};

use vulkan_framework::{
    buffer::{AllocatedBuffer, Buffer, BufferUsage, BufferUseAs, ConcreteBufferDescriptor},
    device::DeviceOwned,
    image::{CommonImageFormat, Image2DDimensions, ImageFormat},
    memory_allocator::{DefaultAllocator, MemoryAllocator},
    memory_heap::{
        ConcreteMemoryHeapDescriptor, MemoryHeap, MemoryHeapOwned, MemoryHostVisibility,
        MemoryRequirements, MemoryType,
    },
    memory_pool::{MemoryMap, MemoryPool, MemoryPoolBacked, MemoryPoolFeatures},
    memory_requiring::MemoryRequiring,
    queue_family::QueueFamily,
};

#[derive(Debug, Default, Copy, Clone)]
struct LoadedTexture {
    texture: u32,
}

#[derive(Debug, Default, Copy, Clone)]

struct LoadedMaterial {
    material_index: u8,
    diffuse_texture: LoadedTexture,
    normal_texture: LoadedTexture,
}

#[derive(Debug, Default, Copy, Clone)]

struct LoadedVertexBuffer {
    vertex_buffer: u32,
}

#[derive(Debug, Default, Copy, Clone)]

struct LoadedMesh {
    index_buffer: u32,
    material: LoadedMaterial,
}

struct MeshDefinition {
    vertex_buffer: LoadedVertexBuffer,
    meshes: Vec<LoadedMesh>,
}

#[repr(C, packed)]
#[derive(Debug, Copy, Clone)]
pub struct MaterialGPU {
    pub diffuse_texture_index: u32,    // 4 bytes
    pub normal_texture_index: u32,     // 4 bytes
    pub reflection_texture_index: u32, // 4 bytes
    pub di_texture_index: u32,         // 4 bytes
                                       // No padding needed since all fields are u32
}

const SIZEOF_MATERIAL_DEFINITION: usize = std::mem::size_of::<MaterialGPU>();

type StateType = u64;
type MaterialsInFlightType =
    smallvec::SmallVec<[(StateType, Arc<AllocatedBuffer>); MAX_FRAMES_IN_FLIGHT_NO_MALLOC]>;

pub struct Manager {
    queue_family: Arc<QueueFamily>,

    memory_pool: Arc<MemoryPool>,

    mesh_manager: MeshManager,
    texture_manager: TextureManager,

    // I can load up to 128 materials
    materials: u128,
    materials_buffer: MaterialsInFlightType,
    current_materials_buffer: Arc<AllocatedBuffer>,
    current_materials_state: StateType,

    objects: HashMap<String, MeshDefinition>,
}

impl Manager {
    fn memory_pool_size(frames_in_flight: u32) -> u64 {
        (1024u64 * 1024u64 * 128u64) + (frames_in_flight as u64 * 8192u64)
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

        let materials_buffer = Buffer::new(
            device.clone(),
            ConcreteBufferDescriptor::new(
                BufferUsage::from([BufferUseAs::TransferSrc].as_slice()),
                (SIZEOF_MATERIAL_DEFINITION as u64) * (MAX_MATERIALS as u64),
            ),
            None,
            Some(format!("resource_management.current_materials_buffer").as_str()),
        )?;

        let memory_heap = MemoryHeap::new(
            device.clone(),
            ConcreteMemoryHeapDescriptor::new(
                MemoryType::DeviceLocal(Some(MemoryHostVisibility::MemoryHostVisibile {
                    cached: false,
                })),
                total_size,
            ),
            MemoryRequirements::try_from(
                [
                    &buffer as &dyn MemoryRequiring,
                    &materials_buffer as &dyn MemoryRequiring,
                ]
                .as_slice(),
            )?,
        )?;

        let memory_pool = MemoryPool::new(
            memory_heap,
            Arc::new(DefaultAllocator::new(total_size)),
            MemoryPoolFeatures::from([].as_slice()),
        )?;

        let current_materials_buffer = AllocatedBuffer::new(memory_pool.clone(), materials_buffer)?;

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

        let materials = 0;
        let mut materials_unallocated_buffer: smallvec::SmallVec<
            [Buffer; MAX_FRAMES_IN_FLIGHT_NO_MALLOC],
        > = smallvec::smallvec![];
        for index in 0..(frames_in_flight as usize) {
            materials_unallocated_buffer.push(Buffer::new(
                device.clone(),
                ConcreteBufferDescriptor::new(
                    BufferUsage::from(
                        [BufferUseAs::TransferDst, BufferUseAs::UniformBuffer].as_slice(),
                    ),
                    (frames_in_flight as u64) * (MAX_MATERIALS as u64),
                ),
                None,
                Some(format!("resource_management.materials_buffer[{index}]").as_str()),
            )?);
        }

        let materials_size = (SIZEOF_MATERIAL_DEFINITION as u64)
            * (MAX_MATERIALS as u64)
            * (frames_in_flight as u64);
        let block_size = 128u64;
        let materials_allocator = DefaultAllocator::with_blocksize(
            block_size,
            ((materials_size / block_size) + 1u64) + (frames_in_flight as u64),
        );

        let materials_memory_heap = MemoryHeap::new(
            device.clone(),
            ConcreteMemoryHeapDescriptor::new(
                MemoryType::DeviceLocal(None),
                materials_allocator.total_size(),
            ),
            MemoryRequirements::try_from(materials_unallocated_buffer.as_slice())?,
        )?;

        let materials_memory_pool = MemoryPool::new(
            materials_memory_heap,
            Arc::new(materials_allocator),
            MemoryPoolFeatures::from([].as_slice()),
        )?;

        let current_materials_state = 0;
        let mut materials_buffer: MaterialsInFlightType = smallvec::smallvec![];
        for buffer in materials_unallocated_buffer.into_iter() {
            let material_buffer = AllocatedBuffer::new(materials_memory_pool.clone(), buffer)?;
            materials_buffer.push((current_materials_state, material_buffer));
        }

        let objects = HashMap::new();

        Ok(Self {
            queue_family,

            memory_pool,

            mesh_manager,
            texture_manager,

            materials,
            materials_buffer,
            current_materials_buffer,
            current_materials_state,

            objects,
        })
    }

    pub fn load_object(&mut self, file: PathBuf) -> RenderingResult<()> {
        let device = self.queue_family.get_parent_device();

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
            indexes: Option<Arc<AllocatedBuffer>>,
        }

        let mut textures: HashMap<String, TextureDecl> = HashMap::new();
        let mut materials: HashMap<String, MaterialDecl> = HashMap::new();
        let mut models: HashMap<String, ModelDecl> = HashMap::new();

        if !file.exists() {
            panic!("File doesn't exists!");
        }

        if !file.is_file() {
            panic!("Not a file");
        }

        let file = std::fs::File::open(file).unwrap();
        let mut archive = Archive::new(file);

        for file in archive.entries()? {
            let loaded_resources = self.texture_manager.wait_load_nonblock()?;
            if loaded_resources > 0 {
                println!("Texture manager has completed loading {loaded_resources} texture(s)");
            }

            let mut file = file?;

            let path = file.header().path()?;

            let path_str = path.to_string_lossy();
            let path_trimmed = match path_str.starts_with("./") {
                true => String::from(&path_str[2..]),
                false => String::from(path_str),
            };
            let splitted_path = path_trimmed.split('/').collect::<Vec<_>>();
            if splitted_path.is_empty() {
                println!("Invalid file name {:?}", file.header().path()?);
            }

            let obj_type = *(splitted_path.first().unwrap());
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
                            None => Default::default(),
                        };

                        match *property {
                            "data" => {
                                let texture_size = file.header().size()?;
                                if texture_size
                                    >= self.memory_pool.get_parent_memory_heap().total_size()
                                {
                                    return Err(RenderingError::ResourceError(
                                        ResourceError::ResourceTooLarge,
                                    ));
                                }

                                // Allocate the buffer that will be used to upload the vertex data to the vulkan device
                                let buffer = Buffer::new(
                                    device.clone(),
                                    ConcreteBufferDescriptor::new(
                                        BufferUsage::from([BufferUseAs::TransferSrc].as_slice()),
                                        texture_size,
                                    ),
                                    None,
                                    Some("resource_management.vertex_buffer"),
                                )?;

                                //println!("allocating texture size: {texture_size}");
                                let buffer =
                                    AllocatedBuffer::new(self.memory_pool.clone(), buffer).unwrap();

                                // Fill the buffer with actual data from the file
                                {
                                    let mut mem_map = MemoryMap::new(self.memory_pool.clone())?;
                                    let slice = mem_map.as_mut_slice::<u32>(
                                        buffer.clone() as Arc<dyn MemoryPoolBacked>
                                    )?;
                                    let len = std::mem::size_of_val(slice);
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
                    }
                    "materials" => {
                        let material_name = String::from(*obj_name);

                        let Some(property) = splitted_path.get(2) else {
                            // this is the directory definition
                            continue;
                        };

                        let mut material_decl = match materials.get(&material_name) {
                            Some(decl) => decl.clone(),
                            None => Default::default(),
                        };

                        match *property {
                            "diffuse_texture" => {
                                let Some(linkname) = file.link_name()? else {
                                    return Err(RenderingError::ResourceError(
                                        ResourceError::InvalidObjectFormat,
                                    ));
                                };

                                let mut name = String::new();
                                for n in linkname.to_string_lossy().split("/") {
                                    name = String::from(n);
                                }

                                material_decl.diffuse_texture.replace(name);
                            }
                            "displacement_texture" => {
                                let Some(linkname) = file.link_name()? else {
                                    return Err(RenderingError::ResourceError(
                                        ResourceError::InvalidObjectFormat,
                                    ));
                                };

                                let mut name = String::new();
                                for n in linkname.to_string_lossy().split("/") {
                                    name = String::from(n);
                                }

                                material_decl.displacement_texture.replace(name);
                            }
                            "reflection_texture" => {
                                let Some(linkname) = file.link_name()? else {
                                    return Err(RenderingError::ResourceError(
                                        ResourceError::InvalidObjectFormat,
                                    ));
                                };

                                let mut name = String::new();
                                for n in linkname.to_string_lossy().split("/") {
                                    name = String::from(n);
                                }

                                material_decl.reflection_texture.replace(name);
                            }
                            "normal_texture" => {
                                let Some(linkname) = file.link_name()? else {
                                    return Err(RenderingError::ResourceError(
                                        ResourceError::InvalidObjectFormat,
                                    ));
                                };

                                let mut name = String::new();
                                for n in linkname.to_string_lossy().split("/") {
                                    name = String::from(n);
                                }

                                material_decl.normal_texture.replace(name);
                            }
                            "" => continue,
                            _ => println!(
                                "WARNING: unrecognised property for material {material_name}: {property}"
                            ),
                        };

                        materials.insert(material_name, material_decl);
                    }
                    "models" => {
                        let model_name = String::from(*obj_name);

                        let Some(property) = splitted_path.get(2) else {
                            // this is the directory definition
                            continue;
                        };

                        let mut model_decl = match models.get(&model_name) {
                            Some(decl) => decl.clone(),
                            None => Default::default(),
                        };

                        match *property {
                            "material" => {
                                let Some(linkname) = file.link_name()? else {
                                    return Err(RenderingError::ResourceError(
                                        ResourceError::InvalidObjectFormat,
                                    ));
                                };

                                let mut name = String::new();
                                for n in linkname.to_string_lossy().split("/") {
                                    name = String::from(n);
                                }

                                model_decl.material_name.replace(name);
                            }
                            "indexes" => {
                                let indexes_size = file.header().size()?;

                                // Allocate the buffer that will be used to upload the index data to the vulkan device
                                let buffer = Buffer::new(
                                    device.clone(),
                                    ConcreteBufferDescriptor::new(
                                        BufferUsage::from([BufferUseAs::TransferSrc].as_slice()),
                                        indexes_size,
                                    ),
                                    None,
                                    Some("resource_management.vertex_buffer"),
                                )?;

                                //println!("allocating index buffer size: {indexes_size}");
                                let buffer =
                                    AllocatedBuffer::new(self.memory_pool.clone(), buffer).unwrap();

                                // Fill the buffer with actual data from the file
                                {
                                    let mut mem_map = MemoryMap::new(self.memory_pool.clone())?;
                                    let slice = mem_map.as_mut_slice::<u32>(
                                        buffer.clone() as Arc<dyn MemoryPoolBacked>
                                    )?;
                                    let len = std::mem::size_of_val(slice);
                                    let ptr = slice.as_mut_ptr() as *mut u8;
                                    let slice_u8 =
                                        unsafe { std::slice::from_raw_parts_mut(ptr, len) };
                                    file.read_exact(slice_u8).unwrap();
                                }

                                model_decl.indexes.replace(buffer);
                            }
                            "" => continue,
                            _ => println!(
                                "WARNING: unrecognised property for model {model_name}: {property}"
                            ),
                        };

                        models.insert(model_name, model_decl);
                    }
                    _ => {
                        print!("WARNING: unrecognised object type");
                    }
                },
                None => {
                    if obj_type == "vertex_buffer" {
                        // Allocate the buffer that will be used to upload the vertex data to the vulkan device
                        let buffer = Buffer::new(
                            device.clone(),
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
                            let len = std::mem::size_of_val(slice);
                            let ptr = slice.as_mut_ptr() as *mut u8;
                            let slice_u8 = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
                            file.read_exact(slice_u8).unwrap();
                        }
                    }
                }
            };
        }

        let mut loaded_textures: HashMap<String, LoadedTexture> = HashMap::new();
        for (k, v) in textures.into_iter() {
            let texture_id = match (&v.width, &v.height, &v.miplevel, &v.data) {
                (Some(width), Some(height), Some(miplevel), Some(data)) => {
                    let texture_id = self.texture_manager.load(
                        ImageFormat::from(CommonImageFormat::bc7_srgb_block),
                        Image2DDimensions::new(*width, *height),
                        *miplevel,
                        data.clone(),
                    )?;

                    println!("Loaded texture {k} at id {texture_id}");

                    texture_id
                }
                _ => {
                    return Err(crate::rendering::RenderingError::ResourceError(
                        ResourceError::IncompleteTexture(k.clone()),
                    ));
                }
            };

            loaded_textures.insert(
                k,
                LoadedTexture {
                    texture: texture_id,
                },
            );
        }

        let mut loaded_materials: HashMap<String, LoadedMaterial> = HashMap::new();
        for (k, v) in materials.into_iter() {
            let diffuse_texture = match &v.diffuse_texture {
                Some(texture_name) => match loaded_textures.get(texture_name) {
                    Some(texture) => texture.to_owned(),
                    None => {
                        println!(
                            "WARNING: no matching texture '{texture_name}' loaded for material {k}"
                        );

                        LoadedTexture {
                            texture: self.texture_manager.stub_texture_index(),
                        }
                    }
                },
                None => LoadedTexture {
                    texture: self.texture_manager.stub_texture_index(),
                },
            };

            let normal_texture = match &v.normal_texture {
                Some(texture_name) => match loaded_textures.get(texture_name) {
                    Some(texture) => texture.to_owned(),
                    None => {
                        println!(
                            "WARNING: no matching texture '{texture_name}' loaded for material {k}"
                        );

                        LoadedTexture {
                            texture: self.texture_manager.stub_texture_index(),
                        }
                    }
                },
                None => LoadedTexture {
                    texture: self.texture_manager.stub_texture_index(),
                },
            };

            let mut material_index = 0u8;
            while material_index < (MAX_MATERIALS as u8) + 1u8 {
                if material_index == (MAX_MATERIALS as u8) {
                    // TODO: no more space available for materials :(
                    todo!()
                }

                let bitshifted_index = 1u128 << material_index;
                if (self.materials & bitshifted_index) == 0u128 {
                    // write material to the buffer memory
                    let mut mem_map =
                        MemoryMap::new(self.current_materials_buffer.get_backing_memory_pool())?;

                    let mapped_materials = mem_map.as_mut_slice::<MaterialGPU>(
                        self.current_materials_buffer.clone() as Arc<dyn MemoryPoolBacked>,
                    )?;

                    mapped_materials[material_index as usize] = MaterialGPU {
                        diffuse_texture_index: diffuse_texture.texture,
                        normal_texture_index: self.texture_manager.stub_texture_index(),
                        reflection_texture_index: self.texture_manager.stub_texture_index(),
                        di_texture_index: self.texture_manager.stub_texture_index(),
                    };

                    self.materials |= bitshifted_index;

                    break;
                }

                material_index += 1;
            }

            loaded_materials.insert(
                k,
                LoadedMaterial {
                    material_index,
                    diffuse_texture,
                    normal_texture,
                },
            );
        }

        println!();

        Ok(())
    }
}
