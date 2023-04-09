use vulkan_framework;

fn main() {
    let instance_extensions = Vec::<String>::new();
    let engine_name = String::from("None");
    let app_name = String::from("hello_triangle");
    let api_version = vulkan_framework::instance::InstanceAPIVersion::Version1_0;

    if let Ok(_instance) = vulkan_framework::instance::Instance::new(
        instance_extensions.as_slice(),
        &engine_name,
        &app_name,
        &api_version,
        false
    ) {
        println!("Vulkan instance created");
    } else {
        println!("Error creating vulkan instance");
    }
}
