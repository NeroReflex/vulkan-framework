use std::time::{Duration, Instant};

use renderdoc::{RenderDoc, V141};

use artrtic::{
    core::camera::{HEAD_DOWN, spectator::SpectatorCamera},
    rendering::system::System,
};

const DEFAULT_WINDOW_WIDTH: u32 = 1280;
const DEFAULT_WINDOW_HEIGHT: u32 = 720;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app_name = String::from("ArtRTic");

    let sdl_context = sdl2::init().unwrap();
    let sdl_mouse = sdl_context.mouse();

    let mut renderer = System::new(
        app_name,
        sdl_context.video().unwrap(),
        DEFAULT_WINDOW_WIDTH,
        DEFAULT_WINDOW_HEIGHT,
    )
    .map_err(|err| panic!("{err}"))
    .unwrap();

    // a test call
    renderer.test();

    let mut rd: RenderDoc<V141> = RenderDoc::new().expect("Unable to connect");

    let (major, minor, patch) = rd.get_api_version();
    println!("RenderDoc API {major}.{minor}.{patch}");

    let mut start_time = Instant::now();
    let mut frame_count = 0;
    let mut event_pump = sdl_context.event_pump().unwrap();

    let hdr = artrtic::core::hdr::HDR::default();
    let mut camera = SpectatorCamera::new(
        glm::vec3(152.0, 650.0, -8.5),
        HEAD_DOWN,
        1.0,
        10000.0,
        -17.2499943,
        -0.0299999565,
        65.0,
    );

    //sdl_mouse.capture(true);
    let move_units_per_second = 275.0;
    let mouse_sensitivity = 0.015;

    let mouse_state = event_pump.mouse_state();
    let mut mouse_pos = glm::vec2(mouse_state.x() as f32, mouse_state.y() as f32);

    //let mut total_elapsed_time_in_seconds = 0.0;

    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                sdl2::event::Event::Quit { .. }
                | sdl2::event::Event::KeyDown {
                    keycode: Some(sdl2::keyboard::Keycode::Escape),
                    ..
                } => {
                    break 'running;
                }
                _ => {}
            }
        }

        let delta_time = (start_time.elapsed().as_micros() as f32) * 1000000.0;

        let new_mouse_state = event_pump.mouse_state();
        let new_mouse_pos = glm::vec2(new_mouse_state.x() as f32, new_mouse_state.y() as f32);

        let orientation_change = new_mouse_pos - mouse_pos;

        // Update the mouse position
        mouse_pos = new_mouse_pos;

        camera.apply_horizontal_rotation(orientation_change.x * mouse_sensitivity);
        camera.apply_vertical_rotation(orientation_change.y * mouse_sensitivity);

        renderer.render(&camera, &hdr).unwrap();
        frame_count += 1;

        // Check if one second has passed
        if start_time.elapsed() >= Duration::from_millis(1000) {
            println!("FPS: {}", frame_count);
            frame_count = 0;
            start_time = Instant::now();
        }
    }

    Ok(())
}
