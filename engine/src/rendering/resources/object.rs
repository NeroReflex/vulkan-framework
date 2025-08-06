use std::{
    collections::HashMap,
    io::Read,
    ops::DerefMut,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use tar::Archive;

use crate::{
    EmbeddedAssets,
    rendering::{
        MAX_MESHES, RenderingError, RenderingResult,
        resources::{ResourceError, SIZEOF_MATERIAL_DEFINITION, materials::MaterialManager},
    },
};

use super::{mesh::MeshManager, texture::TextureManager};

use vulkan_framework::{
    acceleration_structure::{
        VertexIndexing,
        bottom_level::{
            BottomLevelAccelerationStructureIndexBuffer,
            BottomLevelAccelerationStructureTransformBuffer,
            BottomLevelAccelerationStructureVertexBuffer, BottomLevelTrianglesGroupDecl,
            BottomLevelVerticesTopologyDecl,
        },
    },
    buffer::{
        AllocatedBuffer, Buffer, BufferTrait, BufferUsage, BufferUseAs, ConcreteBufferDescriptor,
    },
    command_buffer::CommandBufferRecorder,
    descriptor_set_layout::DescriptorSetLayout,
    device::DeviceOwned,
    graphics_pipeline::{AttributeType, IndexType},
    image::{CommonImageFormat, Image2DDimensions, ImageFormat},
    memory_heap::{MemoryHostVisibility, MemoryType},
    memory_management::{MemoryManagementTagSize, MemoryManagementTags, MemoryManagerTrait},
    memory_pool::{MemoryMap, MemoryPoolBacked, MemoryPoolFeatures},
    pipeline_layout::PipelineLayout,
    queue_family::QueueFamily,
    shader_stage_access::ShaderStagesAccess,
};

#[derive(Debug, Default, Copy, Clone)]
struct LoadedTexture {
    texture: u32,
}

#[derive(Debug, Default, Copy, Clone)]

struct LoadedMaterial {
    material_index: u32,
    diffuse_texture: LoadedTexture,
    normal_texture: LoadedTexture,
}

#[derive(Debug, Default, Copy, Clone)]

struct LoadedMesh {
    mesh_index: u32,
    material: LoadedMaterial,
}

struct MeshDefinition {
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

type LoadedMeshesType = smallvec::SmallVec<[Option<MeshDefinition>; MAX_MESHES as usize]>;

pub struct Manager {
    debug_name: String,

    queue_family: Arc<QueueFamily>,

    memory_manager: Arc<Mutex<dyn MemoryManagerTrait>>,

    mesh_manager: MeshManager,
    texture_manager: TextureManager,
    material_manager: MaterialManager,

    current_mesh_to_material_map: Arc<AllocatedBuffer>,

    objects: LoadedMeshesType,
}

impl Manager {
    #[inline]
    pub fn vertex_buffer_position_stride() -> u32 {
        (4u32 * 3u32) + (4u32 * 3u32) + (4u32 * 2u32)
    }

    #[inline]
    pub fn vertex_buffer_normals_stride() -> u32 {
        (4u32 * 3u32) + (4u32 * 3u32) + (4u32 * 2u32)
    }

    #[inline]
    pub fn vertex_buffer_texture_uv_stride() -> u32 {
        (4u32 * 3u32) + (4u32 * 3u32) + (4u32 * 2u32)
    }

    #[inline]
    fn leftover_memory(frames_in_flight: u32) -> u64 {
        (1024u64 * 1024u64 * 128u64) + (frames_in_flight as u64 * 8192u64)
    }

    #[inline]
    pub fn textures_descriptor_set_layout(&self) -> Arc<DescriptorSetLayout> {
        self.texture_manager.descriptor_set_layout()
    }

    #[inline]
    pub fn materials_descriptor_set_layout(&self) -> Arc<DescriptorSetLayout> {
        self.material_manager.descriptor_set_layout()
    }

    pub fn new(
        queue_family: Arc<QueueFamily>,
        memory_manager: Arc<Mutex<dyn MemoryManagerTrait>>,
        frames_in_flight: u32,
        debug_name: String,
    ) -> RenderingResult<Self> {
        let device = queue_family.get_parent_device();

        let buffer = Buffer::new(
            device.clone(),
            ConcreteBufferDescriptor::new(
                BufferUsage::from([BufferUseAs::TransferSrc].as_slice()),
                1024u64 * 1024u64 * 24u64, // 24MiB
            ),
            None,
            Some(format!("{debug_name}->resource_management.stub_image_buffer").as_str()),
        )?;

        let current_mesh_to_material_map = Buffer::new(
            device.clone(),
            ConcreteBufferDescriptor::new(
                BufferUsage::from([BufferUseAs::TransferSrc].as_slice()),
                (MAX_MESHES as u64) * 4u64,
            ),
            None,
            Some(
                format!("{debug_name}->resource_management.current_mesh_to_material_map").as_str(),
            ),
        )?;

        // allocate resources
        let mut memory_allocator = memory_manager.lock().unwrap();
        let alloc_result = memory_allocator
            .allocate_resources(
                &MemoryType::DeviceLocal(Some(MemoryHostVisibility::MemoryHostVisibile {
                    cached: false,
                })),
                &MemoryPoolFeatures::default(),
                vec![buffer.into()],
                MemoryManagementTags::default()
                    .with_name("temp".to_string())
                    .with_size(MemoryManagementTagSize::MediumSmall),
            )
            .inspect_err(|err| println!("Unable to allocate buffer for the stub image: {err}"))?;
        assert_eq!(alloc_result.len(), 1_usize);
        let stub_image_data = alloc_result.first().unwrap().buffer();

        let alloc_result = memory_allocator
            .allocate_resources(
                &MemoryType::DeviceLocal(Some(MemoryHostVisibility::MemoryHostVisibile {
                    cached: false,
                })),
                &MemoryPoolFeatures::default(),
                vec![current_mesh_to_material_map.into()],
                MemoryManagementTags::default()
                    .with_name("temp".to_string())
                    .with_size(MemoryManagementTagSize::MediumSmall),
            )
            .inspect_err(|err| {
                println!(
                    "Unable to allocate buffer for instance to material associative map: {err}"
                )
            })?;
        assert_eq!(alloc_result.len(), 1_usize);
        let current_mesh_to_material_map = alloc_result.first().unwrap().buffer();
        drop(memory_allocator);

        {
            let mut mem_map = MemoryMap::new(stub_image_data.get_backing_memory_pool())?;
            let temp = EmbeddedAssets::get("stub.dds").unwrap();
            let stub_emedded_data = temp.data.as_ref();
            let size = stub_emedded_data.len();
            let stub_data = mem_map.as_mut_slice_with_size::<u8>(
                stub_image_data.clone() as Arc<dyn MemoryPoolBacked>,
                size as u64,
            )?;
            stub_data.copy_from_slice(stub_emedded_data);
        }

        let mesh_manager = MeshManager::new(
            queue_family.clone(),
            memory_manager.clone(),
            frames_in_flight,
            format!("{debug_name}->mesh_manager"),
        )?;
        let texture_manager = TextureManager::new(
            queue_family.clone(),
            memory_manager.clone(),
            stub_image_data.clone(),
            frames_in_flight,
            format!("{debug_name}->texture_manager"),
        )?;
        let material_manager = MaterialManager::new(
            queue_family.clone(),
            frames_in_flight,
            format!("{debug_name}->material_manager"),
        )?;

        let mut objects = LoadedMeshesType::with_capacity(MAX_MESHES as usize);
        for _ in 0..objects.capacity() {
            objects.push(None);
        }

        Ok(Self {
            debug_name,

            queue_family,

            memory_manager,

            mesh_manager,
            texture_manager,
            material_manager,

            current_mesh_to_material_map,

            objects,
        })
    }

    pub fn load_object(&mut self, file: PathBuf) -> RenderingResult<usize> {
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

        #[derive(Default)]
        struct ModelDecl {
            material_name: Option<String>,
            indexes: Option<BottomLevelAccelerationStructureIndexBuffer>,
            transform: Option<BottomLevelAccelerationStructureTransformBuffer>,
        }

        let mut textures: HashMap<String, TextureDecl> = HashMap::new();
        let mut materials: HashMap<String, MaterialDecl> = HashMap::new();
        let mut models: HashMap<String, ModelDecl> = HashMap::new();
        let mut vertex_buffer: Option<(BottomLevelVerticesTopologyDecl, Arc<AllocatedBuffer>)> =
            Option::None;

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
                                // Allocate the buffer that will be used to upload the vertex data to the vulkan device
                                let buffer = Buffer::new(
                                    device.clone(),
                                    ConcreteBufferDescriptor::new(
                                        BufferUsage::from([BufferUseAs::TransferSrc].as_slice()),
                                        file.header().size()?,
                                    ),
                                    None,
                                    Some(
                                        format!("resource_management.texture[{texture_name}]")
                                            .as_str(),
                                    ),
                                )?;

                                let buffer = {
                                    let mut allocator = self.memory_manager.lock().unwrap();
                                    let alloc_result = allocator.allocate_resources(
                                        &MemoryType::DeviceLocal(Some(
                                            MemoryHostVisibility::MemoryHostVisibile {
                                                cached: false,
                                            },
                                        )),
                                        &MemoryPoolFeatures::default(),
                                        vec![buffer.into()],
                                        MemoryManagementTags::default()
                                            .with_name("temp".to_string())
                                            .with_size(MemoryManagementTagSize::MediumSmall),
                                    )?;
                                    assert_eq!(alloc_result.len(), 1_usize);
                                    alloc_result[0].buffer()
                                };

                                // Fill the buffer with actual data from the file
                                {
                                    let mut mem_map =
                                        MemoryMap::new(buffer.get_backing_memory_pool())?;
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

                        let mut material_decl = match materials.get(&material_name) {
                            Some(decl) => decl.clone(),
                            None => Default::default(),
                        };

                        if let Some(property) = splitted_path.get(2) {
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
                            }
                        };

                        materials.insert(material_name, material_decl);
                    }
                    "models" => {
                        let model_name = String::from(*obj_name);

                        let Some(property) = splitted_path.get(2) else {
                            // this is the directory definition
                            continue;
                        };

                        let mut model_decl = models.remove(&model_name).unwrap_or_default();

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
                            // TODO: "transform" => { /* for the initial model matrix */}
                            "indexes" => {
                                let size = file.header().size()?;
                                let indexes_size = size;

                                let vertex_count =
                                    indexes_size as usize / std::mem::size_of::<u32>();
                                let triangle_count = vertex_count as u64 / 3u64;

                                let triangles_decl = BottomLevelTrianglesGroupDecl::new(
                                    VertexIndexing::UInt32,
                                    triangle_count as u32,
                                );

                                // Allocate the buffer that will be used to upload the index data to the vulkan device
                                let index_buffer = {
                                    let mut mem_manager_guard = self.memory_manager.lock().unwrap();
                                    self.mesh_manager.create_index_buffer(
                                        mem_manager_guard.deref_mut(),
                                        triangles_decl,
                                        BufferUsage::from(
                                            [BufferUseAs::IndexBuffer, BufferUseAs::UniformBuffer]
                                                .as_slice(),
                                        ),
                                        None,
                                    )?
                                };

                                // Fill the buffer with actual data from the file
                                {
                                    let mut mem_map = MemoryMap::new(
                                        index_buffer.buffer().get_backing_memory_pool().clone(),
                                    )?;
                                    let slice = mem_map.as_mut_slice_with_size::<u8>(
                                        index_buffer.buffer().clone() as Arc<dyn MemoryPoolBacked>,
                                        index_buffer.buffer().size(),
                                    )?;
                                    assert_eq!(size as usize, slice.len());
                                    file.read_exact(slice).unwrap();
                                }

                                model_decl.indexes.replace(index_buffer);
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
                        let vertex_stride = 5u64 * 4u64;
                        let vertex_size = (3u64 * 4u64) + vertex_stride;

                        let size = file.header().size()?;

                        let vertex_count = size / vertex_size;
                        if (size % vertex_size) != 0u64 {
                            panic!("AAAAAAAHHH");
                        }

                        let vertices_topology = BottomLevelVerticesTopologyDecl::new(
                            vertex_count as u32,
                            AttributeType::Vec3,
                            vertex_stride,
                        );

                        // Allocate the buffer that will be used to upload the vertex data to the vulkan device
                        let buffer = {
                            let mut mem_manager_guard = self.memory_manager.lock().unwrap();
                            self.mesh_manager
                                .create_vertex_buffer(
                                    mem_manager_guard.deref_mut(),
                                    vertices_topology,
                                    BufferUsage::from(
                                        [BufferUseAs::VertexBuffer, BufferUseAs::StorageBuffer]
                                            .as_slice(),
                                    ),
                                    None,
                                )
                                .inspect_err(|err| {
                                    println!("Unable to create vertex buffer: {err}")
                                })?
                        };

                        // Fill the buffer with actual data from the file
                        {
                            let mut mem_map =
                                MemoryMap::new(buffer.1.get_backing_memory_pool().clone())?;
                            let slice = mem_map.as_mut_slice_with_size::<u8>(
                                buffer.1.clone() as Arc<dyn MemoryPoolBacked>,
                                buffer.1.size(),
                            )?;
                            assert_eq!(size as usize, slice.len());
                            file.read_exact(slice).unwrap();
                        }

                        vertex_buffer.replace(buffer);
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

            let material_def_size = std::mem::size_of::<MaterialGPU>();
            assert_eq!(SIZEOF_MATERIAL_DEFINITION, material_def_size);
            let material_buffer = Buffer::new(
                device.clone(),
                ConcreteBufferDescriptor::new(
                    [BufferUseAs::TransferSrc].as_slice().into(),
                    material_def_size as u64,
                ),
                None,
                Some(""),
            )?;

            let material_buffer = {
                let mut allocator = self.memory_manager.lock().unwrap();
                let alloc_result = allocator
                    .allocate_resources(
                        &MemoryType::DeviceLocal(Some(MemoryHostVisibility::MemoryHostVisibile {
                            cached: false,
                        })),
                        &MemoryPoolFeatures::default(),
                        vec![material_buffer.into()],
                        MemoryManagementTags::default()
                            .with_name("temp".to_string())
                            .with_size(MemoryManagementTagSize::MediumSmall),
                    )
                    .inspect_err(|err| {
                        println!("Unable to allocate buffer for material definition: {err}")
                    })?;
                assert_eq!(alloc_result.len(), 1_usize);
                alloc_result[0].buffer()
            };

            {
                // write material to the buffer memory
                let mut mem_map = MemoryMap::new(material_buffer.get_backing_memory_pool())?;

                let mapped_materials = mem_map.as_mut_slice::<MaterialGPU>(
                    material_buffer.clone() as Arc<dyn MemoryPoolBacked>,
                )?;

                mapped_materials[0] = MaterialGPU {
                    diffuse_texture_index: diffuse_texture.texture,
                    normal_texture_index: self.texture_manager.stub_texture_index(),
                    reflection_texture_index: self.texture_manager.stub_texture_index(),
                    di_texture_index: self.texture_manager.stub_texture_index(),
                };
            }

            let Ok(material_index) = self.material_manager.load(material_buffer) else {
                panic!("MATERIAL NOT LOADED");
            };

            loaded_materials.insert(
                k,
                LoadedMaterial {
                    material_index,
                    diffuse_texture,
                    normal_texture,
                },
            );
        }

        // TODO: allow for multiple instances.
        let Some((vertices_topology, vertex_buffer)) = vertex_buffer.take() else {
            return Err(RenderingError::ResourceError(
                ResourceError::MissingVertexBuffer,
            ));
        };

        let mut loaded_models: HashMap<String, LoadedMesh> = HashMap::new();
        for (k, mut v) in models.into_iter() {
            let Some(index_buffer) = v.indexes.take() else {
                println!("WARNING: model {k} is missing its index buffer and will be skipped");
                continue;
            };

            let transform_buffer = {
                let mut memory_manager = self.memory_manager.lock().unwrap();
                match v.transform.take() {
                    Some(transform) => transform,
                    None => self.mesh_manager.create_transform_buffer(
                        memory_manager.deref_mut(),
                        BufferUsage::default(),
                        None,
                    )?,
                }
            };

            let Some(material_name) = v.material_name else {
                println!(
                    "WARNING: model {k} is missing its material definition and will be skipped"
                );
                continue;
            };

            let Some(material) = loaded_materials.get(&material_name) else {
                println!(
                    "WARNING: model {k} has an invalid material definition ({material_name}) and will be skipped"
                );
                continue;
            };

            let material = *material;
            let material_index: u32 = material.material_index;

            let mesh_index = self.mesh_manager.load(
                BottomLevelAccelerationStructureVertexBuffer::new(
                    vertices_topology,
                    vertex_buffer.clone(),
                )?,
                index_buffer,
                transform_buffer,
            )?;

            loaded_models.insert(
                k,
                LoadedMesh {
                    mesh_index,
                    material,
                },
            );

            // register into the GPU the mesh->material association
            {
                let mut mem_map = MemoryMap::new(
                    self.current_mesh_to_material_map
                        .get_backing_memory_pool()
                        .clone(),
                )?;
                let slice = mem_map.as_mut_slice_with_size::<u32>(
                    self.current_mesh_to_material_map.clone() as Arc<dyn MemoryPoolBacked>,
                    self.current_mesh_to_material_map.size(),
                )?;
                assert_eq!(MAX_MESHES as usize, slice.len());
                slice[mesh_index as usize] = material_index;
            }
        }

        let blas_created = self.mesh_manager.wait_load_nonblock()?;
        println!("During loading resources GPU has already created {blas_created} BLAS");

        let mesh = MeshDefinition {
            meshes: loaded_models.iter().map(|mesh| *mesh.1).collect(),
        };

        for allocation_index in 0..self.objects.len() {
            if self.objects[allocation_index].is_some() {
                continue;
            }

            self.objects[allocation_index] = Some(mesh);

            return Ok(allocation_index);
        }

        todo!()
    }

    /// Performs a guided rendering.
    /// When called inside a rendering recording function it will update a push constant
    /// containing a single u32 to the specified offset and stage, bind the vertex buffer,
    /// bind the index buffers and dispatch relevants draw calls.
    ///
    /// This function avoids rendering assets that are not completely loaded in GPU memory.
    #[inline]
    pub fn guided_rendering(
        &self,
        recorder: &mut CommandBufferRecorder,
        current_frame: usize,
        pipeline_layout: Arc<PipelineLayout>,
        textures_descriptor_set_binding: u32,
        materials_descriptor_set_binding: u32,
        push_constant_offset: u32,
        push_constant_size: u32,
        push_constant_stages: ShaderStagesAccess,
    ) {
        // update materials descriptor sets (to make them relevants to this frame) and bind them
        recorder.bind_descriptor_sets_for_graphics_pipeline(
            pipeline_layout.clone(),
            textures_descriptor_set_binding,
            [self.texture_manager.texture_descriptor_set(current_frame)].as_slice(),
        );

        recorder.bind_descriptor_sets_for_graphics_pipeline(
            pipeline_layout.clone(),
            materials_descriptor_set_binding,
            [self.material_manager.material_descriptor_set(current_frame)].as_slice(),
        );

        // now try to render every object that has its resources completely loaded in GPU memory
        for obj_index in 0..self.objects.len() {
            let Some(loaded_obj) = &self.objects[obj_index] else {
                continue;
            };

            // just an optimization: avoid issuing lots of useless bind_vertex_buffers calls
            let mut last_bound_vertex_buffer = MAX_MESHES as u64;

            for mesh_index in 0..loaded_obj.meshes.len() {
                let loaded_mesh: &LoadedMesh = &loaded_obj.meshes[mesh_index];

                let Some(blas) = self
                    .mesh_manager
                    .fetch_loaded(loaded_mesh.mesh_index as usize)
                else {
                    println!(
                        "skipping drawing of a mesh due to a bound blas not being fully loaded"
                    );
                    continue;
                };

                if !self
                    .texture_manager
                    .is_loaded(loaded_mesh.material.diffuse_texture.texture as usize)
                {
                    println!(
                        "skipping drawing of a mesh due to a bound texture not being fully loaded"
                    );
                    continue;
                }

                if !self
                    .texture_manager
                    .is_loaded(loaded_mesh.material.normal_texture.texture as usize)
                {
                    println!(
                        "skipping drawing of a mesh due to a bound texture not being fully loaded"
                    );
                    continue;
                }

                // This is the material ID to be given to mesh rendering so that it can find
                // on the material buffer, and with that address textures
                let material_id = loaded_mesh.material.material_index;

                if !self.material_manager.is_loaded(material_id as usize) {
                    println!(
                        "skipping drawing of a mesh due to a bound material not being fully loaded"
                    );
                    continue;
                }

                recorder.push_constant(
                    pipeline_layout.clone(),
                    push_constant_stages,
                    push_constant_offset,
                    unsafe {
                        ::core::slice::from_raw_parts(
                            (&material_id as *const _) as *const u8,
                            push_constant_size as usize,
                        )
                    },
                );

                let vertex_buffer = blas.vertex_buffer();
                if vertex_buffer.buffer().native_handle() != last_bound_vertex_buffer {
                    last_bound_vertex_buffer = vertex_buffer.buffer().native_handle();
                    recorder.bind_vertex_buffers(
                        0,
                        [
                            (0u64, vertex_buffer.buffer() as Arc<dyn BufferTrait>),
                            (4u64 * 3u64, vertex_buffer.buffer() as Arc<dyn BufferTrait>),
                            (
                                4u64 * (3u64 + 2u64),
                                vertex_buffer.buffer() as Arc<dyn BufferTrait>,
                            ),
                        ]
                        .as_slice(),
                    );
                }

                recorder.bind_index_buffer(0, blas.index_buffer().buffer(), IndexType::UInt32);

                // TODO: for now drawing one instance, but in the future allows multiple instances to be rendered
                recorder.draw_indexed(blas.max_primitives_count() * 3u32, 1, 0, 0, 0);
            }
        }
    }

    #[inline]
    pub fn wait_blocking(&mut self) -> RenderingResult<()> {
        self.texture_manager.wait_load_blocking()?;
        self.mesh_manager.wait_load_blocking()?;
        self.material_manager.wait_load_blocking()?;
        Ok(())
    }

    #[inline]
    pub fn wait_nonblocking(&mut self) -> RenderingResult<()> {
        self.texture_manager.wait_load_nonblock()?;
        self.mesh_manager.wait_load_nonblock()?;
        self.material_manager.wait_load_nonblock()?;
        Ok(())
    }
}
