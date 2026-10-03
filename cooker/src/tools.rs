//! Locate `compressonatorcli` / `toktx` without requiring a global PATH.
//!
//! Search order: `ARTRTIC_TOOLS_DIR`, directory containing this executable,
//! `~/.local/share/artrtic/bin`, common Compressonator install folders, then PATH.

use std::path::{Path, PathBuf};
use std::process::Command;

pub fn tools_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("ARTRTIC_TOOLS_DIR") {
        let path = PathBuf::from(dir);
        if path.is_dir() {
            return Some(path);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            if parent.is_dir() {
                return Some(parent.to_path_buf());
            }
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        let share = PathBuf::from(home).join(".local/share/artrtic/bin");
        if share.is_dir() {
            return Some(share);
        }
    }
    None
}

pub fn resolve_tool(name: &str) -> Option<PathBuf> {
    let candidates = tool_search_dirs();
    for dir in candidates {
        let path = dir.join(name);
        if is_runnable(&path) {
            return Some(path);
        }
    }
    which_on_path(name)
}

fn tool_search_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(dir) = tools_dir() {
        dirs.push(dir);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            dirs.push(parent.to_path_buf());
        }
    }
    if let Ok(dir) = std::env::var("ARTRTIC_TOOLS_DIR") {
        dirs.push(PathBuf::from(dir));
    }
    if let Some(home) = std::env::var_os("HOME") {
        let home = PathBuf::from(home);
        dirs.push(home.join(".local/share/artrtic/bin"));
        let bin = home.join(".bin");
        if bin.is_dir() {
            if let Ok(entries) = std::fs::read_dir(&bin) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() && path.file_name().is_some_and(|n| {
                        n.to_string_lossy().starts_with("compressonatorcli")
                    }) {
                        dirs.push(path);
                    }
                }
            }
        }
    }
    dirs
}

fn is_runnable(path: &Path) -> bool {
    path.is_file() && std::fs::metadata(path).is_ok_and(|meta| {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            meta.permissions().mode() & 0o111 != 0
        }
        #[cfg(not(unix))]
        {
            true
        }
    })
}

fn which_on_path(tool: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(tool);
        if is_runnable(&candidate) {
            return Some(candidate);
        }
    }
    None
}

pub fn require_tools(names: &[&str]) -> Result<(), String> {
    let mut missing = Vec::new();
    for name in names {
        if resolve_tool(name).is_none() {
            missing.push(*name);
        }
    }
    if missing.is_empty() {
        Ok(())
    } else {
        let hint = tools_dir()
            .map(|dir| format!(" (also tried {})", dir.display()))
            .unwrap_or_default();
        Err(format!(
            "{}{} — place them next to artrtic-cook, set ARTRTIC_TOOLS_DIR, or use PATH",
            missing.join(", "),
            hint
        ))
    }
}

pub fn run_tool(tool: &str, args: &[&str]) -> std::io::Result<std::process::ExitStatus> {
    let path = resolve_tool(tool).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("{tool} not found"),
        )
    })?;
    run_at_path(&path, args)
}

pub fn run_at_path(path: &Path, args: &[&str]) -> std::io::Result<std::process::ExitStatus> {
    let use_bash = path
        .extension()
        .is_none_or(|ext| ext != "exe")
        && std::fs::read(path)
            .ok()
            .and_then(|bytes| bytes.first().copied())
            .is_some_and(|byte| byte == b'#');
    if use_bash {
        let mut command = Command::new("bash");
        command.arg(path);
        command.args(args);
        command.status()
    } else {
        Command::new(path).args(args).status()
    }
}
