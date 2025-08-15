#[derive(Copy, Clone)]
pub struct DirectionalLight {
    direction: glm::Vec3,
    albedo: glm::Vec3,
}

impl DirectionalLight {
    pub fn new(direction: glm::Vec3, albedo: glm::Vec3) -> Self {
        Self { direction, albedo }
    }

    pub fn direction(&self) -> glm::Vec3 {
        glm::normalize(self.direction)
    }

    pub fn albedo(&self) -> glm::Vec3 {
        self.albedo
    }
}
