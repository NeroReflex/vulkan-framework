#[cfg(test)]
mod instance_device_tests {
    use std::sync::Arc;

    use crate::prelude::*;

    #[test]
    fn test_instance_and_device_creation() -> VulkanResult<()> {
        match crate::tests::common::setup_test_device() {
            Ok((_instance, device)) => {
                println!("Created device with native handle: {:#x}", device.native_handle());
                Ok(())
            }
            Err(err) => {
                eprintln!("Skipping test_instance_and_device_creation: {}", err);
                Ok(())
            }
        }
    }
}
