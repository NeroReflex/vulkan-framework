use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use artrtic::{
    core::camera::{CameraTrait, HEAD_DOWN, spectator::SpectatorCamera},
    rendering::system::System,
};
use sdl2::keyboard::Scancode;

const DEFAULT_WINDOW_WIDTH: u32 = 1280;
const DEFAULT_WINDOW_HEIGHT: u32 = 720;

#[cfg(debug_assertions)]
const PREFERRED_FRAMES_IN_FLIGHT: u32 = 6u32;

#[cfg(not(debug_assertions))]
const PREFERRED_FRAMES_IN_FLIGHT: u32 = 1u32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app_name = String::from("ArtRTic");

    let sdl_context = sdl2::init().unwrap();
    let sdl_mouse = sdl_context.mouse();
    // Relative mode breaks ImGui hit-testing (mouse position stops updating). Toggle per frame.
    sdl_mouse.set_relative_mouse_mode(false);

    let preferred_frames = std::env::var("ART_RTIC_FRAMES_IN_FLIGHT")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|count| *count > 0)
        .unwrap_or(PREFERRED_FRAMES_IN_FLIGHT);
    // A bounded smoke test exits normally, including GPU/resource teardown.
    let max_frames = std::env::var("ART_RTIC_MAX_FRAMES")
        .ok()
        .and_then(|value| value.parse::<u64>().ok());
    let mut rendered_frames = 0u64;

    let mut renderer = System::new(
        app_name,
        sdl_context.video().unwrap(),
        DEFAULT_WINDOW_WIDTH,
        DEFAULT_WINDOW_HEIGHT,
        preferred_frames,
    )
    .map_err(|err| panic!("{err}"))
    .unwrap();

    // a test call
    renderer.test();

    let mut locked = false;

    let mut prev_frame_lock_key_was_pressed = false;

    let mut start_time = Instant::now();
    let mut frame_count = 0;
    let mut event_pump = sdl_context.event_pump().unwrap();
    let mut last_frame_time = Instant::now();
    let bench_wander = std::env::var("ART_RTIC_BENCH_WANDER")
        .ok()
        .is_some_and(|value| value != "0");
    let mut frame_times_s = Vec::<f32>::new();

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

    // set initial camera
    renderer.change_camera(Arc::new(camera.clone()));

    let move_units_per_second = 275.0;
    let mouse_sensitivity_per_millisecond = 0.0015;

    let mouse_state = event_pump.mouse_state();
    let mut mouse_pos = glm::vec2(mouse_state.x() as f32, mouse_state.y() as f32);
    let mut mouse_rel = glm::vec2(0.0f32, 0.0f32);

    //let mut total_elapsed_time_in_seconds = 0.0;

    'running: loop {
        let coeff = last_frame_time.elapsed().as_millis() as f32;
        last_frame_time = Instant::now();
        let mous_coeff = mouse_sensitivity_per_millisecond * coeff;
        for event in event_pump.poll_iter() {
            if let sdl2::event::Event::MouseMotion { xrel, yrel, .. } = event {
                mouse_rel.x += xrel as f32;
                mouse_rel.y += yrel as f32;
            }
            if let Some(ui) = renderer.ui_layer_mut() {
                ui.handle_event(&event);
            }
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

        renderer.build_ui_frame(&event_pump.mouse_state());

        let ui_wants_mouse = renderer
            .ui_layer_mut()
            .is_some_and(|ui| ui.wants_capture_mouse());
        let ui_camera_locked = renderer
            .ui_layer_mut()
            .is_some_and(|ui| ui.camera_locked);

        // Never use SDL relative mouse mode: it freezes/warps coordinates and breaks ImGui
        // window dragging (Scene panel flies off-screen on title-bar click). FPS look uses
        // accumulated MouseMotion xrel/yrel instead.
        let fps_mouse = !bench_wander && !locked && !ui_wants_mouse && !ui_camera_locked;

        // Update camera position
        {
            if let Some((dx, dy, forward, strafe)) = artrtic::preview::take_look() {
                camera.apply_horizontal_rotation(dx);
                camera.apply_vertical_rotation(dy);
                camera.apply_movement(camera.orientation(), forward);
                camera.apply_movement(glm::cross(camera.orientation(), camera.head()), strafe);
            }
            let move_quantity = move_units_per_second * (coeff / 1000.0);
            let new_keyboard_state = event_pump.keyboard_state();

            if !prev_frame_lock_key_was_pressed
                && new_keyboard_state.is_scancode_pressed(Scancode::Space)
            {
                locked = !locked;
                prev_frame_lock_key_was_pressed = true;
                //sdl_mouse.set_relative_mouse_mode(locked);
            } else if !new_keyboard_state.is_scancode_pressed(Scancode::Space) {
                prev_frame_lock_key_was_pressed = false;
            }

            if bench_wander {
                // Stay in the atrium: lock pitch, yaw and strafe on XZ, never look at sky.
                let t = rendered_frames as f32 * (1.0 / 60.0);
                let step = move_units_per_second * (1.0 / 60.0);
                camera.set_vertical_angle(-0.03);
                camera.apply_horizontal_rotation((t * 0.85).sin() * 0.035);
                let mut forward = camera.orientation();
                forward.y = 0.0;
                let forward_len = glm::length(forward);
                if forward_len > 1e-4 {
                    forward = glm::normalize(forward);
                    camera.apply_movement(forward, step * (0.65 + 0.35 * (t * 0.31).sin()));
                }
                let strafe = glm::normalize(glm::cross(
                    glm::Vec3::new(0.0, 1.0, 0.0),
                    forward,
                ));
                camera.apply_movement(strafe, step * 0.45 * (t * 0.23).cos());
                renderer.change_camera(Arc::new(camera.clone()));
            } else if !locked && !ui_wants_mouse && !ui_camera_locked {
                if new_keyboard_state.is_scancode_pressed(Scancode::W) {
                    camera.apply_movement(camera.orientation(), move_quantity);

                    renderer.change_camera(Arc::new(camera.clone()));
                }

                if new_keyboard_state.is_scancode_pressed(Scancode::S) {
                    camera.apply_movement(camera.orientation(), -move_quantity);

                    renderer.change_camera(Arc::new(camera.clone()));
                }

                if new_keyboard_state.is_scancode_pressed(Scancode::D) {
                    camera.apply_movement(
                        glm::normalize(glm::cross(
                            glm::Vec3::new(0.0, 1.0, 0.0),
                            camera.orientation(),
                        )),
                        move_quantity,
                    );

                    renderer.change_camera(Arc::new(camera.clone()));
                }

                if new_keyboard_state.is_scancode_pressed(Scancode::A) {
                    camera.apply_movement(
                        glm::normalize(glm::cross(
                            glm::Vec3::new(0.0, 1.0, 0.0),
                            camera.orientation(),
                        )),
                        -move_quantity,
                    );

                    renderer.change_camera(Arc::new(camera.clone()));
                }
            }
        }

        // Update the mouse position
        {
            let orientation_change = if fps_mouse {
                let delta = mouse_rel * mous_coeff;
                mouse_rel = glm::vec2(0.0, 0.0);
                delta
            } else {
                let new_mouse_state = event_pump.mouse_state();
                let new_mouse_pos =
                    glm::vec2(new_mouse_state.x() as f32, new_mouse_state.y() as f32);
                let delta = (new_mouse_pos - mouse_pos) * mous_coeff;
                mouse_pos = new_mouse_pos;
                delta
            };

            if !bench_wander
                && !locked
                && !ui_wants_mouse
                && !ui_camera_locked
                && (orientation_change.x != 0.0 || orientation_change.y != 0.0)
            {
                camera.apply_horizontal_rotation(orientation_change.x);
                camera.apply_vertical_rotation(orientation_change.y);
                renderer.change_camera(Arc::new(camera.clone()));
            }
        }

        let gpu_start = Instant::now();
        renderer.render(&hdr)?;
        let frame_s = gpu_start.elapsed().as_secs_f32();
        // Skip scene upload / pipeline warmup.
        if rendered_frames >= 90 {
            frame_times_s.push(frame_s);
        }
        frame_count += 1;
        rendered_frames += 1;
        if max_frames.is_some_and(|limit| rendered_frames >= limit) {
            println!("Completed bounded run: {rendered_frames} frames");
            break 'running;
        }

        // Check if one second has passed
        if start_time.elapsed() >= Duration::from_millis(1000) {
            println!("FPS: {}", frame_count);
            frame_count = 0;
            start_time = Instant::now();
        }
    }

    print_frame_stats(&frame_times_s);
    Ok(())
}

fn print_frame_stats(times: &[f32]) {
    if times.is_empty() {
        return;
    }
    let mut sorted = times.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let sum: f32 = sorted.iter().sum();
    let pct = |p: f32| sorted[(((n - 1) as f32) * p) as usize];
    println!(
        "Frame times ({n} frames after warmup): min {:.2}ms avg {:.2}ms p95 {:.2}ms p99 {:.2}ms max {:.2}ms ({:.1} FPS avg)",
        sorted[0] * 1000.0,
        (sum / n as f32) * 1000.0,
        pct(0.95) * 1000.0,
        pct(0.99) * 1000.0,
        sorted[n - 1] * 1000.0,
        n as f32 / sum
    );
}
