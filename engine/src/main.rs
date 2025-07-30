use std::time::{Duration, Instant};

use artrtic::{core::hdr::HDR, rendering::system::System};

const DEFAULT_WINDOW_WIDTH: u32 = 1280;
const DEFAULT_WINDOW_HEIGHT: u32 = 720;

fn lcg(seed: u64) -> u64 {
    const A: u64 = 1664525;
    const C: u64 = 1013904223;
    const M: u64 = 1 << 32; // 2^32

    (A.wrapping_mul(seed).wrapping_add(C)) % M
}

fn random_float(seed: &mut u64, lower_bound: f64, upper_bound: f64) -> f64 {
    *seed = lcg(*seed);
    let range = upper_bound - lower_bound;
    lower_bound + (*seed as f64 / (u64::MAX as f64)) * range
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app_name = String::from("ArtRTic");

    let sdl_context = sdl2::init().unwrap();

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

    let mut start_time = Instant::now();
    let mut frame_count = 0;
    let mut event_pump = sdl_context.event_pump().unwrap();

    let mut seed = 12345;
    let mut hdr = artrtic::core::hdr::HDR::default();

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

        renderer.render(&hdr).unwrap();
        frame_count += 1;

        // Check if one second has passed
        if start_time.elapsed() >= Duration::from_millis(1000) {
            hdr = artrtic::core::hdr::HDR::new(
                random_float(&mut seed, 0.01, 4.0) as f32,
                random_float(&mut seed, 0.01, 4.0) as f32,
            );

            println!("FPS: {}", frame_count);
            frame_count = 0;
            start_time = Instant::now();
        }
    }

    Ok(())
}
