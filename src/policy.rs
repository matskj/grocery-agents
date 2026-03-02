use crate::model::{Action, GameState};

#[derive(Debug, Default)]
pub struct Policy;

impl Policy {
    pub fn new() -> Self {
        Self
    }

    pub fn decide(&self, _state: &GameState) -> Action {
        Action::idle()
    }
}
