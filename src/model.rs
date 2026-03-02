use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeContext {
    pub token: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GameState {
    pub tick: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    pub kind: String,
}

impl Action {
    pub fn idle() -> Self {
        Self {
            kind: "idle".to_owned(),
        }
    }
}
