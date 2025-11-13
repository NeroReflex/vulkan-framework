use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;
use std::sync::mpsc::{self, Sender, Receiver};
use std::thread;

use quick_xml::events::Event;
use quick_xml::Reader;

#[derive(Debug, Clone)]
pub struct MeasurementResult {
    pub red: u8,
    pub green: u8,
    pub blue: u8,
    pub x: Option<f64>,
    pub y: Option<f64>,
    pub y_lum: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct MeasureRequest {
    pub red: u8,
    pub green: u8,
    pub blue: u8,
}

pub struct ColourSpaceClient {
    stream: TcpStream,
}

impl ColourSpaceClient {
    pub fn connect(addr: &str) -> std::io::Result<Self> {
        let stream = TcpStream::connect(addr)?;
        stream.set_read_timeout(Some(Duration::from_secs(5)))?;
        stream.set_write_timeout(Some(Duration::from_secs(5)))?;
        Ok(Self { stream })
    }

    fn send_xml(&mut self, xml: &str) -> std::io::Result<()> {
        let count = xml.chars().count();
        let header = format!("{}", count);
        self.stream.write_all(header.as_bytes())?;
        self.stream.write_all(xml.as_bytes())?;
        self.stream.flush()?;
        Ok(())
    }

    fn read_message(&mut self) -> std::io::Result<String> {
        let mut buf = [0u8; 1];
        let mut digits = Vec::new();
        loop {
            let n = self.stream.read(&mut buf)?;
            if n == 0 {
                return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "stream closed"));
            }
            let b = buf[0];
            if (b as char).is_ascii_digit() {
                digits.push(b);
                if digits.len() > 16 {
                    return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "header too long"));
                }
                continue;
            } else {
                let header = String::from_utf8(digits.clone()).map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid header"))?;
                let len: usize = header.parse().map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid header parse"))?;
                // Skip optional whitespace characters (spaces, tabs, newlines, CR) between header and payload
                let mut first_payload_byte: Option<u8> = None;
                if !(b as char).is_whitespace() {
                    first_payload_byte = Some(b);
                } else {
                    // consume further bytes until a non-whitespace or EOF
                    loop {
                        let n = self.stream.read(&mut buf)?;
                        if n == 0 {
                            return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "stream closed while skipping whitespace"));
                        }
                        let nb = buf[0];
                        if !(nb as char).is_whitespace() {
                            first_payload_byte = Some(nb);
                            break;
                        }
                    }
                }

                let mut payload = Vec::with_capacity(len + 1);
                if let Some(fp) = first_payload_byte {
                    payload.push(fp);
                    let mut to_read = len.saturating_sub(1);
                    while to_read > 0 {
                        let mut chunk = vec![0u8; to_read.min(4096)];
                        let r = self.stream.read(&mut chunk)?;
                        if r == 0 { break; }
                        payload.extend_from_slice(&chunk[..r]);
                        to_read = to_read.saturating_sub(r);
                    }
                } else {
                    // No payload byte found - treat as error
                    return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "no payload after header"));
                }

                return String::from_utf8(payload).map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid utf8 payload"));
            }
        }
    }

    pub fn init_profile(&mut self) -> std::io::Result<()> {
        let xml = r#"<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n<CS_RMC version=1>\n<command>\ninit profile\n</command>\n</CS_RMC>"#;
        self.send_xml(xml)
    }

    pub fn measure(&mut self, r: u8, g: u8, b: u8) -> std::io::Result<MeasurementResult> {
        let xml = format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n<CS_RMC version=1>\n<measurement>\n<red>{}</red>\n<green>{}</green>\n<blue>{}</blue>\n</measurement>\n</CS_RMC>",
            r, g, b
        );
        self.send_xml(&xml)?;
        let msg = self.read_message()?;
        let mut reader = Reader::from_str(&msg);
        reader.trim_text(true);
        let mut buf = Vec::new();
        let mut in_result = false;
        let mut cur_elem = String::new();
    let mut res = MeasurementResult { red: r, green: g, blue: b, x: None, y: None, y_lum: None };
        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(e)) => {
                    let name = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    cur_elem = name.clone();
                    if name == "result" { in_result = true; }
                }
                Ok(Event::Text(e)) => {
                    if !in_result { continue; }
                    let txt = e.unescape().unwrap_or_default().into_owned();
                    match cur_elem.as_str() {
                        "red" => if let Ok(v) = txt.parse::<u8>() { res.red = v },
                        "green" => if let Ok(v) = txt.parse::<u8>() { res.green = v },
                        "blue" => if let Ok(v) = txt.parse::<u8>() { res.blue = v },
                        "x" => if let Ok(v) = txt.parse::<f64>() { res.x = Some(v) },
                        "y" => if let Ok(v) = txt.parse::<f64>() { res.y = Some(v) },
                        "Y" => if let Ok(v) = txt.parse::<f64>() { res.y_lum = Some(v) },
                        _ => {}
                    }
                }
                Ok(Event::Eof) => break,
                Err(e) => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("xml parse error: {}", e))),
                _ => {}
            }
            buf.clear();
        }
        Ok(res)
    }
}

/// Spawn a background worker thread that keeps a connection and performs measurements.
/// Returns (request_sender, response_receiver).
pub fn spawn_worker(addr: &str) -> std::io::Result<(Sender<MeasureRequest>, Receiver<std::result::Result<MeasurementResult, String>>)> {
    let (tx_req, rx_req): (Sender<MeasureRequest>, Receiver<MeasureRequest>) = mpsc::channel();
    let (tx_resp, rx_resp): (Sender<std::result::Result<MeasurementResult, String>>, Receiver<std::result::Result<MeasurementResult, String>>) = mpsc::channel();
    let addr = addr.to_owned();
    thread::spawn(move || {
        match ColourSpaceClient::connect(&addr) {
            Ok(mut client) => {
                let _ = client.init_profile();
                // process requests until sender is dropped
                while let Ok(req) = rx_req.recv() {
                    match client.measure(req.red, req.green, req.blue) {
                        Ok(res) => { let _ = tx_resp.send(Ok(res)); }
                        Err(e) => { let _ = tx_resp.send(Err(e.to_string())); }
                    }
                }
            }
            Err(e) => {
                let _ = tx_resp.send(Err(format!("connect error: {}", e)));
            }
        }
    });
    Ok((tx_req, rx_resp))
}
