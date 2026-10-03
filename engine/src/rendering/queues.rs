/// Map `unique` hardware queues onto an upload slot and `frames` in-flight slots.
///
/// Rendering slots take distinct queues first so frames can overlap. When the
/// family has a spare queue (`unique > frames`) that spare is dedicated to
/// uploads and initialization. Otherwise uploads share queue 0 with whatever
/// frames round-robin onto it. A single queue is shared by everything.
pub(crate) fn map_queues_to_frames(unique: usize, frames: usize) -> (usize, Vec<usize>) {
    assert!(unique >= 1, "at least one hardware queue is required");
    assert!(frames >= 1, "at least one frame in flight is required");

    if unique > frames {
        let upload = unique - 1;
        let frame_indices = (0..frames).collect();
        (upload, frame_indices)
    } else {
        let frame_indices = (0..frames).map(|index| index % unique).collect();
        (0, frame_indices)
    }
}
