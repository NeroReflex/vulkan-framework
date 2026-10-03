//! Localhost preview. The engine blits the GI image to a small buffer, JPEG
//! encodes it, and sends it to one websocket client. Input comes back as a
//! single text line: `look dx dy forward strafe`.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Mutex, OnceLock};
use std::thread;

struct PreviewState {
    jpeg: Vec<u8>,
    look: Option<(f32, f32, f32, f32)>,
}

fn state() -> &'static Mutex<PreviewState> {
    static STATE: OnceLock<Mutex<PreviewState>> = OnceLock::new();
    STATE.get_or_init(|| {
        Mutex::new(PreviewState {
            jpeg: Vec::new(),
            look: None,
        })
    })
}

pub fn enabled() -> bool {
    std::env::var("ART_RTIC_PREVIEW")
        .ok()
        .is_some_and(|value| value != "0")
}

pub fn start() {
    if !enabled() {
        return;
    }
    thread::spawn(|| {
        let listener = match TcpListener::bind("127.0.0.1:9761") {
            Ok(listener) => listener,
            Err(err) => {
                eprintln!("preview listen failed: {err}");
                return;
            }
        };
        eprintln!("preview websocket on ws://127.0.0.1:9761");
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            let _ = serve(stream);
        }
    });
}

pub fn publish_rgba(width: u32, height: u32, rgba32f: &[u8]) {
    let pixels = (width as usize) * (height as usize);
    if rgba32f.len() < pixels * 16 {
        return;
    }
    let mut rgb = Vec::with_capacity(pixels * 3);
    for pixel in 0..pixels {
        let offset = pixel * 16;
        for channel in 0..3 {
            let bits = u32::from_le_bytes(
                rgba32f[offset + channel * 4..offset + channel * 4 + 4]
                    .try_into()
                    .unwrap(),
            );
            let value = f32::from_bits(bits);
            let mapped = (value / (1.0 + value.abs())).clamp(0.0, 1.0);
            rgb.push((mapped * 255.0) as u8);
        }
    }
    let mut jpeg = Vec::new();
    let encoder = jpeg_encoder::Encoder::new(&mut jpeg, 70);
    if encoder
        .encode(&rgb, width as u16, height as u16, jpeg_encoder::ColorType::Rgb)
        .is_err()
    {
        return;
    }
    if let Ok(mut preview) = state().lock() {
        preview.jpeg = jpeg;
    }
}

pub fn take_look() -> Option<(f32, f32, f32, f32)> {
    state().lock().ok()?.look.take()
}

fn serve(mut stream: TcpStream) -> std::io::Result<()> {
    let mut header = [0u8; 2048];
    let read = stream.read(&mut header)?;
    let text = String::from_utf8_lossy(&header[..read]);
    let key = text
        .lines()
        .find_map(|line| line.split_once("Sec-WebSocket-Key:"))
        .map(|(_, value)| value.trim())
        .unwrap_or("");
    let accept = websocket_accept(key);
    let response = format!(
        "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n"
    );
    stream.write_all(response.as_bytes())?;
    stream.set_nonblocking(true)?;
    loop {
        let mut incoming = [0u8; 4096];
        match stream.read(&mut incoming) {
            Ok(0) => break,
            Ok(n) => store_client_text(&incoming[..n]),
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(_) => break,
        }
        let jpeg = state().lock().ok().map(|preview| preview.jpeg.clone()).unwrap_or_default();
        if !jpeg.is_empty() {
            write_binary_frame(&mut stream, &jpeg)?;
        }
        thread::sleep(std::time::Duration::from_millis(33));
    }
    Ok(())
}

fn store_client_text(frame: &[u8]) {
    if frame.len() < 6 || frame[0] & 0x0F != 0x1 {
        return;
    }
    let masked = frame[1] & 0x80 != 0;
    let mut len = (frame[1] & 0x7F) as usize;
    let mut offset = 2;
    if len == 126 {
        if frame.len() < 4 {
            return;
        }
        len = u16::from_be_bytes([frame[2], frame[3]]) as usize;
        offset = 4;
    }
    if !masked || frame.len() < offset + 4 + len {
        return;
    }
    let mask = [frame[offset], frame[offset + 1], frame[offset + 2], frame[offset + 3]];
    offset += 4;
    let text: String = frame[offset..offset + len]
        .iter()
        .enumerate()
        .map(|(index, byte)| (byte ^ mask[index % 4]) as char)
        .collect();
    let mut parts = text.split_whitespace();
    if parts.next() != Some("look") {
        return;
    }
    let Some(dx) = parts.next().and_then(|v| v.parse().ok()) else { return };
    let Some(dy) = parts.next().and_then(|v| v.parse().ok()) else { return };
    let Some(forward) = parts.next().and_then(|v| v.parse().ok()) else { return };
    let Some(strafe) = parts.next().and_then(|v| v.parse().ok()) else { return };
    if let Ok(mut preview) = state().lock() {
        preview.look = Some((dx, dy, forward, strafe));
    }
}

fn write_binary_frame(stream: &mut TcpStream, payload: &[u8]) -> std::io::Result<()> {
    let mut header = Vec::with_capacity(10 + payload.len());
    header.push(0x82);
    if payload.len() < 126 {
        header.push(payload.len() as u8);
    } else if payload.len() <= u16::MAX as usize {
        header.push(126);
        header.extend_from_slice(&(payload.len() as u16).to_be_bytes());
    } else {
        header.push(127);
        header.extend_from_slice(&(payload.len() as u64).to_be_bytes());
    }
    header.extend_from_slice(payload);
    stream.write_all(&header)
}

fn websocket_accept(key: &str) -> String {
    sha1_base64(&format!("{key}258EAFA5-E914-47DA-95CA-C5AB0DC85B11"))
}

fn sha1_base64(text: &str) -> String {
    let hash = sha1(text.as_bytes());
    base64(&hash)
}

fn sha1(message: &[u8]) -> [u8; 20] {
    let mut data = message.to_vec();
    let bit_len = (message.len() as u64) * 8;
    data.push(0x80);
    while (data.len() % 64) != 56 {
        data.push(0);
    }
    data.extend_from_slice(&bit_len.to_be_bytes());
    let mut h = [
        0x67452301u32,
        0xEFCDAB89,
        0x98BADCFE,
        0x10325476,
        0xC3D2E1F0,
    ];
    for chunk in data.chunks(64) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(chunk[i * 4..i * 4 + 4].try_into().unwrap());
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let (mut a, mut b, mut c, mut d, mut e) = (h[0], h[1], h[2], h[3], h[4]);
        for i in 0..80 {
            let (f, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5A827999),
                20..=39 => (b ^ c ^ d, 0x6ED9EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDC),
                _ => (b ^ c ^ d, 0xCA62C1D6),
            };
            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(w[i]);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
    }
    let mut out = [0u8; 20];
    for (index, word) in h.iter().enumerate() {
        out[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

fn base64(bytes: &[u8]) -> String {
    const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    let mut index = 0;
    while index + 3 <= bytes.len() {
        let n = ((bytes[index] as u32) << 16) | ((bytes[index + 1] as u32) << 8) | bytes[index + 2] as u32;
        out.push(TABLE[((n >> 18) & 63) as usize] as char);
        out.push(TABLE[((n >> 12) & 63) as usize] as char);
        out.push(TABLE[((n >> 6) & 63) as usize] as char);
        out.push(TABLE[(n & 63) as usize] as char);
        index += 3;
    }
    if index < bytes.len() {
        let remain = bytes.len() - index;
        let mut n = (bytes[index] as u32) << 16;
        if remain == 2 {
            n |= (bytes[index + 1] as u32) << 8;
        }
        out.push(TABLE[((n >> 18) & 63) as usize] as char);
        out.push(TABLE[((n >> 12) & 63) as usize] as char);
        if remain == 2 {
            out.push(TABLE[((n >> 6) & 63) as usize] as char);
            out.push('=');
        } else {
            out.push('=');
            out.push('=');
        }
    }
    out
}
