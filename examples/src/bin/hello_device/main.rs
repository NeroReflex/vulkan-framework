use std::rc::Rc;

use vulkan_framework;

fn main() {
    let instance_extensions = Vec::<String>::new();
    let engine_name = String::from("None");
    let app_name = String::from("hello_device");
    let api_version = vulkan_framework::instance::InstanceAPIVersion::Version1_0;

    let device_extensions: Vec<String> = vec![];
    let device_layers: Vec<String> = vec![];
    let required_queues: Vec<vulkan_framework::queue_family::ConcreteQueueFamilyDescriptor> = vec![
        vulkan_framework::queue_family::ConcreteQueueFamilyDescriptor::new(
            [vulkan_framework::queue_family::QueueFamilySupportedOperationType::Compute].as_slice(),
            [1.0f32].as_slice(),
        ),
    ];

    if let Ok(instance) = vulkan_framework::instance::Instance::new(
        instance_extensions.as_slice(),
        &engine_name,
        &app_name,
        &api_version,
        true,
    ) {
        println!("Vulkan instance created");

        if let Ok(_device) = vulkan_framework::device::Device::new(
            Rc::downgrade(&instance),
            required_queues.as_slice().as_ref(),
            device_extensions.as_slice().as_ref(),
            device_layers.as_slice().as_ref(),
            { todo!() },
        ) {
            println!("Device opened successfully");
        } else {
            println!("Error opening a suitable device");
        }
    } else {
        println!("Error creating vulkan instance");
    }
}
