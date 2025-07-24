use vulkan_framework::{device::*, instance::*, queue_family::*};

fn main() {
    let engine_name = String::from("None");
    let app_name = String::from("hello_device");

    let device_extensions: Vec<String> = vec![];

    let Ok(instance) = Instance::new(
        [String::from("VK_LAYER_KHRONOS_validation")].as_slice(),
        [String::from("VK_EXT_debug_utils")].as_slice(),
        &engine_name,
        &app_name,
    ) else {
        panic!("Error creating vulkan instance");
    };

    println!("Vulkan instance created");

    let Ok(_device) = Device::new(
        instance,
        [ConcreteQueueFamilyDescriptor::new(
            vec![QueueFamilySupportedOperationType::Compute].as_ref(),
            [1.0f32].as_slice(),
        )]
        .as_slice(),
        device_extensions.as_slice(),
        Some("Opened Device"),
    ) else {
        panic!("Error opening a suitable device");
    };

    println!("Device opened successfully");
}
