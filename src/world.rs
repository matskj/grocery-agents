use crate::model::GameState;

#[derive(Debug, Default)]
pub struct World {
    state: GameState,
}

impl World {
    pub fn new(state: GameState) -> Self {
        Self { state }
    }

    pub fn state(&self) -> &GameState {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut GameState {
        &mut self.state
    }
}
