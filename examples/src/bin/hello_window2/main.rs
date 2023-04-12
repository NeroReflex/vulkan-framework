use vulkan_framework::{self, instance::Instance};

fn main() {
    let mut instance_extensions = Vec::<String>::new();
    let engine_name = String::from("None");
    let app_name = String::from("hello_window2");
    let api_version = vulkan_framework::instance::InstanceAPIVersion::Version1_0;

    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    // this parenthesis contains the window handle and closes before calling deinitialize(). This is important as Window::drop() MUST be called BEFORE calling deinitialize()!!!
    // create a sdl2 window
    let window = video_subsystem
        .window("Window", 800, 600)
        .vulkan()
        .build()
        .unwrap();

    for required_instance_extension_name in window.vulkan_instance_extensions().unwrap() {
        instance_extensions.push(String::from(required_instance_extension_name));
    }

    let instance = match Instance::new(
        [String::from("VK_LAYER_KHRONOS_validation")].as_slice(),
        instance_extensions.as_slice(),
        &engine_name,
        &app_name,
        &api_version,
    ) {
        Ok(a) => a,
        Err(_) => return (),
    };

    let raw_surface_khr = window
        .vulkan_create_surface(instance.native_handle() as sdl2::video::VkInstance)
        .unwrap();

    let _surface = vulkan_framework::surface::Surface::from_raw(instance, raw_surface_khr);
}
