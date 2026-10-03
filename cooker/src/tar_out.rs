use std::io::Write;
use std::path::Path;

pub fn append_bytes<W: Write>(builder: &mut tar::Builder<W>, path: &str, bytes: &[u8]) {
    let mut header = tar::Header::new_ustar();
    header.set_path(path).unwrap();
    header.set_size(bytes.len() as u64);
    header.set_mode(0o644);
    header.set_cksum();
    builder.append(&header, bytes).unwrap();
}

pub fn append_symlink<W: Write>(builder: &mut tar::Builder<W>, path: &str, target: &str) {
    let mut header = tar::Header::new_ustar();
    header.set_path(path).unwrap();
    header.set_entry_type(tar::EntryType::Symlink);
    header.set_size(0);
    header.set_mode(0o777);
    builder
        .append_link(&mut header, Path::new(path), Path::new(target))
        .unwrap();
}
