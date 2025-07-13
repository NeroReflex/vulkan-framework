use vulkan_framework::{self, instance::Instance};

fn main() {
    let engine_name = String::from("None");
    let app_name = String::from("hello_triangle");

    if let Ok(_instance) = Instance::new(
        [String::from("VK_LAYER_KHRONOS_validation")].as_slice(),
        [String::from("VK_EXT_debug_utils")].as_slice(),
        &engine_name,
        &app_name,
    ) {
        println!("Vulkan instance created");
    } else {
        println!("Error creating vulkan instance");
    }
}
