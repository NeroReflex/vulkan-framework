use crate::core::camera::CameraTrait;

pub struct SpectatorCamera {
    position: glm::Vec3,

    head: glm::Vec3,

    near_plane: f32,
    far_plane: f32,

    horiz_angle: f32,
    vert_angle: f32,

    field_of_view: f32,
}

impl SpectatorCamera {
    pub fn set_horizontal_angle(&mut self, angle: f32) {
        self.horiz_angle = angle;
    }

    pub fn set_vertical_angle(&mut self, angle: f32) {
        self.vert_angle = angle;
    }

    pub fn apply_horizontal_rotation(&mut self, amount: f32) {
        self.horiz_angle += amount;
    }

    pub fn apply_vertical_rotation(&mut self, amount: f32) {
        self.vert_angle += amount;
    }

    pub fn new(
        position: glm::Vec3,
        head: glm::Vec3,
        near_plane: f32,
        far_plane: f32,
        horiz_angle: f32,
        vert_angle: f32,
        fov_deg: f32,
    ) -> Self {
        let field_of_view = glm::radians(fov_deg);

        Self {
            position,

            head,

            near_plane,
            far_plane,

            horiz_angle,
            vert_angle,

            field_of_view,
        }
    }
}

impl CameraTrait for SpectatorCamera {
    fn position(&self) -> glm::Vec3 {
        self.position
    }

    fn head(&self) -> glm::Vec3 {
        self.head
    }

    fn near_plane(&self) -> f32 {
        self.near_plane
    }

    fn far_plane(&self) -> f32 {
        self.far_plane
    }

    fn horizontal_angle(&self) -> f32 {
        self.horiz_angle
    }

    fn vertical_angle(&self) -> f32 {
        self.vert_angle
    }

    fn projection_matrix(&self, width: u32, height: u32) -> glm::Mat4 {
        let aspect = (width as f32) / (height as f32);
        glm::ext::perspective(
            self.field_of_view,
            aspect,
            self.near_plane(),
            self.far_plane(),
        )
    }
}
