use std::time::{Duration, Instant};

use artrtic::{
    core::camera::{CameraTrait, HEAD_DOWN, spectator::SpectatorCamera},
    rendering::system::System,
};
use sdl2::keyboard::Scancode;

const DEFAULT_WINDOW_WIDTH: u32 = 1280;
const DEFAULT_WINDOW_HEIGHT: u32 = 720;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app_name = String::from("ArtRTic");

    let sdl_context = sdl2::init().unwrap();

    let mut preferred_frames_in_flight = 1u32;
    #[cfg(debug_assertions)]
    {
        preferred_frames_in_flight = 6;
    }

    let mut renderer = System::new(
        app_name,
        sdl_context.video().unwrap(),
        DEFAULT_WINDOW_WIDTH,
        DEFAULT_WINDOW_HEIGHT,
        preferred_frames_in_flight
    )
    .map_err(|err| panic!("{err}"))
    .unwrap();

    // a test call
    renderer.test();

    let mut start_time = Instant::now();
    let mut frame_count = 0;
    let mut event_pump = sdl_context.event_pump().unwrap();
    let mut last_frame_time = Instant::now();

    let hdr = artrtic::core::hdr::HDR::default();
    let mut camera = SpectatorCamera::new(
        glm::vec3(152.0, 650.0, -8.5),
        HEAD_DOWN,
        1.0,
        10000.0,
        -17.249_994,
        -0.029_999_956,
        65.0,
    );

    //sdl_mouse.capture(true);
    let move_units_per_second = 275.0;
    let mouse_sensitivity_per_millisecond = 0.0015;

    let mouse_state = event_pump.mouse_state();
    let mut mouse_pos = glm::vec2(mouse_state.x() as f32, mouse_state.y() as f32);

    //let mut total_elapsed_time_in_seconds = 0.0;

    'running: loop {
        let coeff = last_frame_time.elapsed().as_millis() as f32;
        last_frame_time = Instant::now();
        let mous_coeff = mouse_sensitivity_per_millisecond * coeff;
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

        // Update camera position
        {
            let move_quantity = move_units_per_second * (coeff / 1000.0);
            let new_keyboard_state = event_pump.keyboard_state();
            {
                if new_keyboard_state.is_scancode_pressed(Scancode::W) {
                    camera.apply_movement(camera.orientation(), move_quantity);
                }

                if new_keyboard_state.is_scancode_pressed(Scancode::S) {
                    camera.apply_movement(camera.orientation(), -1.0 * move_quantity);
                }

                if new_keyboard_state.is_scancode_pressed(Scancode::D) {
                    camera.apply_movement(
                        glm::normalize(glm::cross(
                            glm::Vec3::new(0.0, 1.0, 0.0),
                            camera.orientation(),
                        )),
                        move_quantity,
                    );
                }

                if new_keyboard_state.is_scancode_pressed(Scancode::A) {
                    camera.apply_movement(
                        glm::normalize(glm::cross(
                            glm::Vec3::new(0.0, 1.0, 0.0),
                            camera.orientation(),
                        )),
                        -1.0 * move_quantity,
                    );
                }
            }
        }

        // Update the mouse position
        {
            let new_mouse_state = event_pump.mouse_state();
            let new_mouse_pos = glm::vec2(new_mouse_state.x() as f32, new_mouse_state.y() as f32);
            let orientation_change = (new_mouse_pos - mouse_pos) * mous_coeff;
            mouse_pos = new_mouse_pos;
            camera.apply_horizontal_rotation(orientation_change.x);
            camera.apply_vertical_rotation(orientation_change.y);
        }

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
