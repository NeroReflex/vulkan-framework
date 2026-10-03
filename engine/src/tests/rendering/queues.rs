use crate::rendering::queues::map_queues_to_frames;

#[test]
fn one_queue_is_shared_by_uploads_and_every_frame() {
    assert_eq!(map_queues_to_frames(1, 6), (0, vec![0, 0, 0, 0, 0, 0]));
}

#[test]
fn fewer_queues_than_frames_round_robin_and_share_uploads() {
    assert_eq!(map_queues_to_frames(2, 6), (0, vec![0, 1, 0, 1, 0, 1]));
    assert_eq!(map_queues_to_frames(3, 4), (0, vec![0, 1, 2, 0]));
}

#[test]
fn matching_counts_give_each_frame_its_own_queue() {
    assert_eq!(
        map_queues_to_frames(6, 6),
        (0, vec![0, 1, 2, 3, 4, 5])
    );
}

#[test]
fn a_spare_queue_is_dedicated_to_uploads() {
    assert_eq!(map_queues_to_frames(7, 6), (6, vec![0, 1, 2, 3, 4, 5]));
    assert_eq!(map_queues_to_frames(3, 2), (2, vec![0, 1]));
}
