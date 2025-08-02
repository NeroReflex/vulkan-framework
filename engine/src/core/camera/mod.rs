pub mod spectator;

pub const HEAD_DOWN: glm::Vec3 = glm::Vec3 {
    x: 0.0,
    y: -1.0,
    z: 0.0,
};
pub const HEAD_UP: glm::Vec3 = glm::Vec3 {
    x: 0.0,
    y: 1.0,
    z: 0.0,
};

pub trait CameraTrait {
    fn position(&self) -> glm::Vec3;

    fn head(&self) -> glm::Vec3;

    fn near_plane(&self) -> f32;

    fn far_plane(&self) -> f32;

    fn horizontal_angle(&self) -> f32;

    fn vertical_angle(&self) -> f32;

    fn orientation(&self) -> glm::Vec3 {
        let vert = self.vertical_angle();
        let horiz = self.horizontal_angle();
        glm::normalize(glm::vec3(
            glm::cos(vert) * glm::sin(horiz),
            glm::sin(vert),
            glm::cos(vert) * glm::cos(horiz),
        ))
    }

    fn view_matrix(&self) -> glm::Mat4 {
        let position = self.position();
        glm::ext::look_at(position, position + self.orientation(), self.head())
    }

    fn projection_matrix(&self, width: u32, height: u32) -> glm::Mat4;
}
