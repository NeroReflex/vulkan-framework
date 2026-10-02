use crate::rendering::surface::SurfaceHelper;

#[test]
fn image_limits_and_frame_slots() {
    assert_eq!(SurfaceHelper::choose_frame_counts(6, 2, 0), Some((6, 7)));
    assert_eq!(SurfaceHelper::choose_frame_counts(6, 2, 3), Some((2, 3)));
    assert_eq!(SurfaceHelper::choose_frame_counts(1, 3, 3), Some((1, 3)));
    assert_eq!(SurfaceHelper::choose_frame_counts(6, 1, 1), Some((1, 1)));
    assert_eq!(SurfaceHelper::choose_frame_counts(0, 2, 0), None);
    assert_eq!(SurfaceHelper::choose_frame_counts(1, 3, 2), None);
    assert_eq!(
        SurfaceHelper::choose_frame_counts(u32::MAX, 2, 0),
        Some((u32::MAX - 1, u32::MAX))
    );
}
