use vulkan_framework::{device::*, instance::*, queue_family::*};

fn main() {
    let instance_extensions = Vec::<String>::new();
    let engine_name = String::from("None");
    let app_name = String::from("hello_device");
    let api_version = InstanceAPIVersion::Version1_0;

    let device_extensions: Vec<String> = vec![];
    let device_layers: Vec<String> = vec![];

    if let Ok(instance) = Instance::new(
        instance_extensions.as_slice(),
        &engine_name,
        &app_name,
        &api_version,
        false,
    ) {
        println!("Vulkan instance created");

        let required_queues = vec![ConcreteQueueFamilyDescriptor::new(
            vec![QueueFamilySupportedOperationType::Compute].as_ref(),
            [1.0f32].as_slice(),
        )];

        if let Ok(_device) = Device::new(
            &instance,
            required_queues,
            [].as_ref(),
            device_extensions.as_slice().as_ref(),
            device_layers.as_slice().as_ref(),
            Some("Opened Device"),
        ) {
            println!("Device opened successfully");
        } else {
            println!("Error opening a suitable device");
        }
    } else {
        println!("Error creating vulkan instance");
    }
}
