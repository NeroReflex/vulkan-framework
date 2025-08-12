use std::{
    collections::HashMap,
    io::Read,
    mem::MaybeUninit,
    ops::DerefMut,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use tar::Archive;

use crate::{
    EmbeddedAssets,
    core::texture::directdraw_surface::{DDSHeader, DDSHeaderDXT10, DirectDrawSurface},
    rendering::{
        MAX_MESHES, RenderingError, RenderingResult,
        resources::{ResourceError, SIZEOF_MATERIAL_DEFINITION, materials::MaterialManager},
    },
};

use super::{mesh::MeshManager, texture::TextureManager};

use vulkan_framework::{
    acceleration_structure::{
        AllowedBuildingDevice, VertexIndexing,
        bottom_level::{
            BottomLevelAccelerationStructureIndexBuffer,
            BottomLevelAccelerationStructureTransformBuffer,
            BottomLevelAccelerationStructureVertexBuffer, BottomLevelTrianglesGroupDecl,
            BottomLevelVerticesTopologyDecl,
        },
        top_level::{
            TopLevelAccelerationStructure, TopLevelAccelerationStructureInstanceBuffer,
            TopLevelBLASGroupDecl,
        },
    },
    buffer::{
        AllocatedBuffer, Buffer, BufferSubresourceRange, BufferTrait, BufferUsage, BufferUseAs,
        ConcreteBufferDescriptor,
    },
    command_buffer::{CommandBufferRecorder, CommandBufferTrait, PrimaryCommandBuffer},
    command_pool::CommandPool,
    descriptor_set_layout::DescriptorSetLayout,
    device::DeviceOwned,
    fence::{Fence, FenceWaiter},
    graphics_pipeline::{AttributeType, IndexType},
    image::Image2DDimensions,
    memory_barriers::{BufferMemoryBarrier, MemoryAccessAs},
    memory_heap::{MemoryHostVisibility, MemoryType},
    memory_management::{MemoryManagementTagSize, MemoryManagementTags, MemoryManagerTrait},
    memory_pool::{MemoryMap, MemoryPoolBacked, MemoryPoolFeatures},
    pipeline_layout::PipelineLayout,
    pipeline_stage::{PipelineStage, PipelineStageAccelerationStructureKHR},
    queue::Queue,
    queue_family::{QueueFamily, QueueFamilyOwned},
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
    reflection_texture: LoadedTexture,
    displacement_texture: LoadedTexture,
}

#[derive(Debug, Default, Copy, Clone)]

struct LoadedMesh {
    mesh_index: u32,
    material: LoadedMaterial,
}

struct MeshDefinition {
    meshes: Vec<LoadedMesh>,
    load_transform: vulkan_framework::ash::vk::TransformMatrixKHR,
}

pub type InstanceDataType = vulkan_framework::ash::vk::TransformMatrixKHR;

struct MeshInstances {
    instances: Vec<InstanceDataType>,
}

#[repr(C, packed)]
#[derive(Debug, Copy, Clone)]
pub struct MaterialGPU {
    pub diffuse_texture_index: u32,
    pub normal_texture_index: u32,
    pub reflection_texture_index: u32,
    pub displacement_texture_index: u32,
}

struct TLASStatus {
    tlas: Arc<TopLevelAccelerationStructure>,
    loading: Option<FenceWaiter>,
}

impl TLASStatus {
    pub fn new(tlas: Arc<TopLevelAccelerationStructure>, loading: FenceWaiter) -> Self {
        Self {
            tlas,
            loading: Some(loading),
        }
    }

    pub fn tlas(&self) -> Arc<TopLevelAccelerationStructure> {
        self.tlas.clone()
    }

    pub fn wait_nonblocking(&mut self) -> RenderingResult<()> {
        let Some(fence_waiter) = self.loading.take() else {
            return Ok(());
        };

        if fence_waiter.complete()? {
            drop(fence_waiter)
        } else {
            self.loading.replace(fence_waiter);
        }

        Ok(())
    }

    pub fn wait_blocking(&mut self) -> RenderingResult<()> {
        drop(self.loading.take());

        Ok(())
    }
}

type LoadedMeshesType =
    smallvec::SmallVec<[Option<(MeshDefinition, MeshInstances)>; MAX_MESHES as usize]>;

pub struct Manager {
    debug_name: String,

    queue_family: Arc<QueueFamily>,

    command_pool: Arc<CommandPool>,

    memory_manager: Arc<Mutex<dyn MemoryManagerTrait>>,

    mesh_manager: MeshManager,
    texture_manager: TextureManager,
    material_manager: MaterialManager,

    current_mesh_to_material_map: Arc<AllocatedBuffer>,

    tlas_loading_queue: Arc<Queue>,
    current_tlas: Option<TLASStatus>,

    objects: LoadedMeshesType,
}

impl Manager {
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

        let command_pool = CommandPool::new(queue_family.clone(), Some("tlas_command_pool"))?;

        let temp = EmbeddedAssets::get("stub.dds").unwrap();
        let stub_emedded_data = temp.data.as_ref();
        let stub_data_buffer = Buffer::new(
            device.clone(),
            ConcreteBufferDescriptor::new(
                BufferUsage::from([BufferUseAs::TransferSrc].as_slice()),
                stub_emedded_data.len() as u64, // 24MiB
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
                vec![stub_data_buffer.into()],
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
            let mem_map = MemoryMap::new(stub_image_data.get_backing_memory_pool())?;
            let mut range =
                mem_map.range::<u8>(stub_image_data.clone() as Arc<dyn MemoryPoolBacked>)?;
            let stub_data = range.as_mut_slice();
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
            memory_manager.clone(),
            frames_in_flight,
            format!("{debug_name}->material_manager"),
        )?;

        let mut objects = LoadedMeshesType::with_capacity(MAX_MESHES as usize);
        for _ in 0..objects.capacity() {
            objects.push(None);
        }

        let tlas_loading_queue = Queue::new(queue_family.clone(), Some("tlas_loading_queue"))?;
        let current_tlas = None;

        Ok(Self {
            debug_name,

            command_pool,

            queue_family,

            memory_manager,

            mesh_manager,
            texture_manager,
            material_manager,

            current_mesh_to_material_map,

            tlas_loading_queue,
            current_tlas,

            objects,
        })
    }

    pub fn load_object(
        &mut self,
        file: PathBuf,
        transform_data: vulkan_framework::ash::vk::TransformMatrixKHR,
    ) -> RenderingResult<usize> {
        let device = self.queue_family.get_parent_device();

        #[derive(Default, Clone)]
        struct TextureDecl {
            width: Option<u32>,
            height: Option<u32>,
            miplevel: Option<u32>,
            data: Option<Arc<AllocatedBuffer>>,
            format: Option<vulkan_framework::ash::vk::Format>,
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
                                let total_size = file.header().size()?;

                                // a DDS fine begins with 0x44 0x44 0x53 0x20
                                let mut header = [0x00u8, 0x00u8, 0x00u8, 0x00u8];
                                let mut read_size = 0u64;
                                file.read_exact(&mut header).unwrap();
                                read_size += 4;

                                if header[0] == 0x44
                                    && header[1] == 0x44
                                    && header[2] == 0x53
                                    && header[3] == 0x20
                                {
                                    let mut dds_uninitialized_header =
                                        MaybeUninit::<DDSHeader>::uninit();
                                    let dds_header_slice = unsafe {
                                        std::slice::from_raw_parts_mut(
                                            dds_uninitialized_header.as_mut_ptr()
                                                as *mut std::ffi::c_void
                                                as *mut u8,
                                            std::mem::size_of::<DDSHeader>(),
                                        )
                                    };
                                    file.read_exact(dds_header_slice).unwrap();
                                    read_size += std::mem::size_of::<DDSHeader>() as u64;
                                    let dds_header =
                                        unsafe { dds_uninitialized_header.assume_init() };

                                    let dds_dxt10_header = if dds_header
                                        .is_followed_by_dxt10_header()
                                    {
                                        let mut dxt10_uninitialized_header =
                                            MaybeUninit::<DDSHeaderDXT10>::uninit();
                                        let dx10_header_slice = unsafe {
                                            std::slice::from_raw_parts_mut(
                                                dxt10_uninitialized_header.as_mut_ptr()
                                                    as *mut std::ffi::c_void
                                                    as *mut u8,
                                                std::mem::size_of::<DDSHeaderDXT10>(),
                                            )
                                        };
                                        assert_eq!(
                                            std::mem::size_of::<DDSHeaderDXT10>(),
                                            dx10_header_slice.len()
                                        );
                                        file.read_exact(dx10_header_slice).unwrap();
                                        read_size += std::mem::size_of::<DDSHeaderDXT10>() as u64;
                                        Some(unsafe { dxt10_uninitialized_header.assume_init() })
                                    } else {
                                        None
                                    };

                                    let surface_header =
                                        DirectDrawSurface::new(dds_header, dds_dxt10_header);

                                    texture_decl.format = Some(surface_header.vulkan_format());
                                    texture_decl.height = Some(surface_header.height());
                                    texture_decl.width = Some(surface_header.width());
                                    texture_decl.miplevel = Some(surface_header.mip_map_count());
                                } else {
                                    panic!("Only DDS is supported for now");
                                }

                                // Allocate the buffer that will be used to upload the vertex data to the vulkan device
                                let buffer = Buffer::new(
                                    device.clone(),
                                    ConcreteBufferDescriptor::new(
                                        BufferUsage::from([BufferUseAs::TransferSrc].as_slice()),
                                        total_size - read_size,
                                    ),
                                    None,
                                    Some(
                                        format!(
                                            "{}->resource_management.texture[{texture_name}]",
                                            self.debug_name
                                        )
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
                                    let mem_map = MemoryMap::new(buffer.get_backing_memory_pool())?;
                                    let mut range = mem_map
                                        .range::<u8>(buffer.clone() as Arc<dyn MemoryPoolBacked>)?;
                                    let slice = range.as_mut_slice();
                                    file.read_exact(slice).unwrap();
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

                                if (vertex_count as u64 % 3u64) != 0 {
                                    panic!("wrong vertex count");
                                }

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
                                        BufferUsage::from([BufferUseAs::IndexBuffer].as_slice()),
                                        None,
                                    )?
                                };

                                // Fill the buffer with actual data from the file
                                {
                                    let mem_map = MemoryMap::new(
                                        index_buffer.buffer().get_backing_memory_pool().clone(),
                                    )?;
                                    let mut range = mem_map
                                        .range::<u8>(index_buffer.buffer().clone()
                                            as Arc<dyn MemoryPoolBacked>)?;
                                    let slice = range.as_mut_slice();
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
                            let mem_map =
                                MemoryMap::new(buffer.1.get_backing_memory_pool().clone())?;
                            let mut range = mem_map
                                .range::<u8>(buffer.1.clone() as Arc<dyn MemoryPoolBacked>)?;
                            let slice = range.as_mut_slice();
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
            let texture_id = match (&v.width, &v.height, &v.miplevel, &v.format, &v.data) {
                (Some(width), Some(height), Some(miplevel), Some(format), Some(data)) => {
                    //println!("Loaded texture {k} at id {texture_id}");

                    self.texture_manager.load(
                        format.into(),
                        Image2DDimensions::new(*width, *height),
                        *miplevel,
                        data.clone(),
                    )?
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

            let displacement_texture = match &v.displacement_texture {
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

            let reflection_texture = match &v.reflection_texture {
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

            let material_def_size = SIZEOF_MATERIAL_DEFINITION;
            assert_eq!(SIZEOF_MATERIAL_DEFINITION, material_def_size);

            let Ok(material_index) = self.material_manager.load(MaterialGPU {
                diffuse_texture_index: diffuse_texture.texture,
                normal_texture_index: normal_texture.texture,
                reflection_texture_index: reflection_texture.texture,
                displacement_texture_index: displacement_texture.texture,
            }) else {
                panic!("MATERIAL NOT LOADED");
            };

            loaded_materials.insert(
                k,
                LoadedMaterial {
                    material_index,
                    diffuse_texture,
                    reflection_texture,
                    normal_texture,
                    displacement_texture,
                },
            );
        }

        let transform_shared_buffer = {
            let mut memory_manager = self.memory_manager.lock().unwrap();

            self.mesh_manager
                .create_transform_buffer(
                    memory_manager.deref_mut(),
                    &transform_data,
                    [BufferUseAs::VertexBuffer].as_slice().into(),
                    None,
                )?
                .buffer()
        };

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

            let transform_buffer = BottomLevelAccelerationStructureTransformBuffer::new(
                transform_shared_buffer.clone(),
            )?;

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

            // TODO: there can be the case where the mesh to matial of the previous frame is being copied
            // to the frame-specific buffer while we modify the mapping loading the model for next frames:
            // to avoid the risk of such race condition it would be best to clone the buffer into a new one
            // and use that new buffer from this moment on.

            // Register into the GPU the mesh->material association
            {
                let mem_map = MemoryMap::new(
                    self.current_mesh_to_material_map
                        .get_backing_memory_pool()
                        .clone(),
                )?;
                let mut mem_range = mem_map.range::<u32>(
                    self.current_mesh_to_material_map.clone() as Arc<dyn MemoryPoolBacked>,
                )?;
                let slice = mem_range.as_mut_slice();
                assert_eq!(MAX_MESHES as usize, slice.len());
                slice[mesh_index as usize] = material_index;
            }
        }

        let blas_created = self.mesh_manager.wait_load_nonblock()?;
        println!("During loading resources GPU has already created {blas_created} BLAS");

        let mesh = MeshDefinition {
            meshes: loaded_models.iter().map(|mesh| *mesh.1).collect(),
            load_transform: transform_data,
        };

        for allocation_index in 0..self.objects.len() {
            if self.objects[allocation_index].is_some() {
                continue;
            }

            self.objects[allocation_index] = Some((mesh, MeshInstances { instances: vec![] }));

            return Ok(allocation_index);
        }

        // TODO: I could not find a spot to placing the model
        todo!()
    }

    pub fn add_instance(
        &mut self,
        object: usize,
        instance: InstanceDataType,
    ) -> RenderingResult<()> {
        let max_instances: usize = self
            .objects
            .iter()
            .map(|loaded_obj| match loaded_obj {
                Some((obj_meshes, obj_instances)) => {
                    obj_meshes.meshes.len() * obj_instances.instances.len()
                }
                None => 0_usize,
            })
            .sum();

        let Some(loaded_mesh) = self.objects.get_mut(object) else {
            panic!()
            //return;
        };

        let Some((mesh_model, mesh_instances)) = loaded_mesh else {
            panic!()
            //return;
        };

        let meshes_count = mesh_model.meshes.len();
        mesh_instances.instances.push(instance);

        assert!((max_instances + meshes_count) < (u32::MAX as usize));

        let max_instances = max_instances as u32 + meshes_count as u32;
        let blas_decl = TopLevelBLASGroupDecl::new();
        let instance_unallocated_buffer = Buffer::new(
            self.queue_family.get_parent_device(),
            TopLevelAccelerationStructureInstanceBuffer::template(
                &blas_decl,
                max_instances,
                [BufferUseAs::VertexBuffer].as_slice().into(),
            ),
            None,
            Some("instance_buffer"),
        )?;

        let tlas = {
            let mut mem_manager = self.memory_manager.lock().unwrap();
            let instance_allocated_data = mem_manager.allocate_resources(
                &MemoryType::DeviceLocal(Some(MemoryHostVisibility::visible(false))),
                &MemoryPoolFeatures::new(true),
                vec![instance_unallocated_buffer.into()],
                MemoryManagementTags::default()
                    .with_name("temp".to_string())
                    .with_size(MemoryManagementTagSize::MediumSmall),
            )?;

            let tlas_buffer = TopLevelAccelerationStructureInstanceBuffer::new(
                blas_decl,
                max_instances,
                instance_allocated_data[0].buffer(),
            )?;

            TopLevelAccelerationStructure::new(
                &mut *mem_manager,
                AllowedBuildingDevice::DeviceOnly,
                tlas_buffer,
                MemoryManagementTags::default()
                    .with_name("temp".to_string())
                    .with_size(MemoryManagementTagSize::MediumSmall),
                None,
                Some("tlas"),
            )
        }?;

        let current_tlas_ref = tlas.as_ref();

        // wait for all meshes to be fully loaded: we will need them for the TLAS construction
        self.mesh_manager.wait_load_blocking()?;

        // now recreate instances of the TLAS
        {
            let mem_map = MemoryMap::new(
                current_tlas_ref
                    .instance_buffer()
                    .buffer()
                    .get_backing_memory_pool(),
            )?;
            let mut range = mem_map
                .range::<vulkan_framework::ash::vk::AccelerationStructureInstanceKHR>(
                    current_tlas_ref.instance_buffer().buffer() as Arc<dyn MemoryPoolBacked>,
                )?;

            let slice = range.as_mut_slice();

            let mut instance_num = 0_usize;
            for obj in self.objects.iter() {
                let Some((obj_mesh, obj_instances)) = obj else {
                    continue;
                };

                for mesh in obj_mesh.meshes.iter() {
                    let mesh_index = mesh.mesh_index;
                    for obj_instance in obj_instances.instances.iter() {
                        slice[instance_num] =
                            vulkan_framework::ash::vk::AccelerationStructureInstanceKHR {
                                transform: *obj_instance,
                                instance_shader_binding_table_record_offset_and_flags:
                                    vulkan_framework::ash::vk::Packed24_8::new(0, 0x01), // VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR
                                instance_custom_index_and_mask:
                                    vulkan_framework::ash::vk::Packed24_8::new(0x00, 0xFF),
                                acceleration_structure_reference:
                                    vulkan_framework::ash::vk::AccelerationStructureReferenceKHR {
                                        device_handle: self
                                            .mesh_manager
                                            .fetch_loaded(mesh_index as usize)
                                            .as_ref()
                                            .unwrap()
                                            .device_addr(),
                                    },
                            };

                        instance_num += 1;
                    }
                }
            }
        }

        let fence = Fence::new(
            self.command_pool
                .get_parent_queue_family()
                .get_parent_device(),
            false,
            Some("successive_tlas_commandl_buffer"),
        )?;

        let command_buffer = PrimaryCommandBuffer::new(
            self.command_pool.clone(),
            Some("successive_tlas_commandl_buffer"),
        )?;

        command_buffer.record_one_time_submit(|recorder| {
            recorder.buffer_barriers(
                [BufferMemoryBarrier::new(
                    [PipelineStage::Host].as_slice().into(),
                    [MemoryAccessAs::HostWrite].as_slice().into(),
                    [PipelineStage::AccelerationStructureKHR(
                        PipelineStageAccelerationStructureKHR::Build,
                    )]
                    .as_slice()
                    .into(),
                    [MemoryAccessAs::MemoryRead].as_slice().into(),
                    BufferSubresourceRange::new(
                        tlas.instance_buffer().buffer(),
                        0,
                        tlas.instance_buffer().buffer().size(),
                    ),
                    self.queue_family.clone(),
                    self.queue_family.clone(),
                )]
                .as_slice(),
            );

            recorder.build_tlas(tlas.clone(), 0, max_instances);

            recorder.buffer_barriers(
                [BufferMemoryBarrier::new(
                    [PipelineStage::AccelerationStructureKHR(
                        PipelineStageAccelerationStructureKHR::Build,
                    )]
                    .as_slice()
                    .into(),
                    [MemoryAccessAs::MemoryRead].as_slice().into(),
                    [PipelineStage::BottomOfPipe].as_slice().into(),
                    [].as_slice().into(),
                    BufferSubresourceRange::new(
                        tlas.instance_buffer().buffer(),
                        0,
                        tlas.instance_buffer().buffer().size(),
                    ),
                    self.queue_family.clone(),
                    self.queue_family.clone(),
                )]
                .as_slice(),
            );
        })?;

        let fence_waiter = self.tlas_loading_queue.submit(
            [command_buffer as Arc<dyn CommandBufferTrait>].as_slice(),
            [].as_slice(),
            [].as_slice(),
            fence,
        )?;

        // WARNING: overwriting the old value will also drop the FenceWaiter that might have
        // been there, meaning this is a blocking instruction that might wait
        // for the GPU to finish the previous operation.
        self.current_tlas = Some(TLASStatus::new(tlas, fence_waiter));

        Ok(())
    }

    pub fn update_buffers(
        &self,
        recorder: &mut CommandBufferRecorder,
        current_frame: usize,
        queue_family: Arc<QueueFamily>,
    ) {
        self.material_manager.update_buffers(
            recorder,
            current_frame,
            self.current_mesh_to_material_map.clone(),
            queue_family,
        );
    }

    /// Performs a guided rendering.
    /// When called inside a rendering recording function it will update a push constant
    /// containing a mat3x4 and a u32 to the specified offset and stage, bind the vertex buffer,
    /// bind the index buffers and dispatch relevants draw calls.
    ///
    /// The mat3x4 is the transformation matrix of the mesh, and the u32 is the mesh index.
    /// 
    /// This function avoids rendering assets that are not completely loaded in GPU memory.
    pub fn guided_rendering(
        &self,
        recorder: &mut CommandBufferRecorder,
        current_frame: usize,
        pipeline_layout: Arc<PipelineLayout>,
        textures_descriptor_set_binding: u32,
        materials_descriptor_set_binding: u32,
        push_constant_offset: u32,
        push_constant_stages: ShaderStagesAccess,
    ) {
        // do not render anything if the TLAS does not exists
        let Some(tlas) = &self.current_tlas else {
            return;
        };

        // bind the updated materials descriptor sets (update happens by calling update_buffers):
        // WARNING: the update MUST have been happened before this method,
        // and proper barriers MUST have been placed already
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

        let mut drawn = 0u64;

        // now try to render every object that has its resources completely loaded in GPU memory
        for obj_index in 0..self.objects.len() {
            let Some((loaded_obj_mesh, loaded_obj_instances)) = &self.objects[obj_index] else {
                continue;
            };

            // If this mesh has no instances skip the rendering
            let instance_count = loaded_obj_instances.instances.len() as u32;
            if instance_count == 0 {
                continue;
            }

            let transform_data = loaded_obj_mesh.load_transform.clone();

            // just an optimization: avoid issuing lots of useless bind_vertex_buffers calls
            let mut last_bound_vertex_buffer = MAX_MESHES as u64;

            for mesh_index in 0..loaded_obj_mesh.meshes.len() {
                let loaded_mesh: &LoadedMesh = &loaded_obj_mesh.meshes[mesh_index];

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
                        "skipping drawing of a mesh due to a bound (diffuse) texture not being fully loaded"
                    );
                    continue;
                }

                if !self
                    .texture_manager
                    .is_loaded(loaded_mesh.material.normal_texture.texture as usize)
                {
                    println!(
                        "skipping drawing of a mesh due to a bound (normal) texture not being fully loaded"
                    );
                    continue;
                }

                if !self
                    .texture_manager
                    .is_loaded(loaded_mesh.material.reflection_texture.texture as usize)
                {
                    println!(
                        "skipping drawing of a mesh due to a bound (reflection) texture not being fully loaded"
                    );
                    continue;
                }

                if !self
                    .texture_manager
                    .is_loaded(loaded_mesh.material.displacement_texture.texture as usize)
                {
                    println!(
                        "skipping drawing of a mesh due to a bound (displacement) texture not being fully loaded"
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

                let transform_size = core::mem::size_of_val(&transform_data) as u32;
                recorder.push_constant(
                    pipeline_layout.clone(),
                    push_constant_stages,
                    push_constant_offset,
                    unsafe {
                        ::core::slice::from_raw_parts(
                            (&transform_data as *const _) as *const u8,
                            transform_size as usize,
                        )
                    },
                );

                let mesh_id_size = core::mem::size_of_val(&loaded_mesh.mesh_index) as u32;
                recorder.push_constant(
                    pipeline_layout.clone(),
                    push_constant_stages,
                    push_constant_offset + transform_size,
                    unsafe {
                        ::core::slice::from_raw_parts(
                            (&loaded_mesh.mesh_index as *const _) as *const u8,
                            mesh_id_size as usize,
                        )
                    },
                );

                let vertex_buffer = blas.vertex_buffer();
                if vertex_buffer.buffer().native_handle() != last_bound_vertex_buffer {
                    last_bound_vertex_buffer = vertex_buffer.buffer().native_handle();
                    recorder.bind_vertex_buffers(
                        0,
                        [(0u64, vertex_buffer.buffer() as Arc<dyn BufferTrait>)].as_slice(),
                    );
                    recorder.bind_vertex_buffers(
                        1,
                        [(
                            drawn
                                * (instance_count as u64)
                                * (core::mem::size_of::<
                                    vulkan_framework::ash::vk::AccelerationStructureInstanceKHR,
                                >() as u64),
                            tlas.tlas().instance_buffer().buffer() as Arc<dyn BufferTrait>,
                        )]
                        .as_slice(),
                    );
                }

                recorder.bind_index_buffer(0, blas.index_buffer().buffer(), IndexType::UInt32);

                // TODO: for now drawing one instance, but in the future allows multiple instances to be rendered
                recorder.draw_indexed(blas.max_primitives_count() * 3u32, instance_count, 0, 0, 0);
                drawn += instance_count as u64;
            }
        }
    }

    #[inline]
    pub fn wait_blocking(&mut self) -> RenderingResult<()> {
        self.texture_manager.wait_load_blocking()?;
        self.mesh_manager.wait_load_blocking()?;
        self.material_manager.wait_load_blocking()?;
        if let Some(tlas_status) = &mut self.current_tlas {
            tlas_status.wait_blocking()?;
        };

        Ok(())
    }

    #[inline]
    pub fn wait_nonblocking(&mut self) -> RenderingResult<()> {
        self.texture_manager.wait_load_nonblock()?;
        self.mesh_manager.wait_load_nonblock()?;
        self.material_manager.wait_load_nonblock()?;
        if let Some(tlas_status) = &mut self.current_tlas {
            tlas_status.wait_nonblocking()?
        };

        Ok(())
    }

    #[inline]
    pub fn tlas(&self) -> Arc<TopLevelAccelerationStructure> {
        self.current_tlas.as_ref().unwrap().tlas()
    }
}
