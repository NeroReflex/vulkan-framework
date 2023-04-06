use ash;
use vulkan_framework;

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

    let instance = match vulkan_framework::instance::Instance::new(
        instance_extensions.as_slice(),
        &engine_name,
        &app_name,
        &api_version,
        true,
        true,
    ) {
        Ok(a) => a,
        Err(_) => return (),
    };

    let _ = window
        .vulkan_create_surface(
            ash::vk::Handle::as_raw(instance.native_handle().handle().clone())
                as sdl2::video::VkInstance,
        )
        .unwrap();

    // validation layers error. Fuck.
    drop(window);
    drop(video_subsystem);
    drop(sdl_context);
    drop(instance);
}
