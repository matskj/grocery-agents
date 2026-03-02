use crate::model::{Action, BotState, GameState};

#[derive(Debug, Default)]
pub struct Policy;

impl Policy {
    pub fn new() -> Self {
        Self
    }

    pub fn decide(&self, _state: &GameState, bot: &BotState) -> Action {
        Action::wait(bot.id.clone())
    }
}
