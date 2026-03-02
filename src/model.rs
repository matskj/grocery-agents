use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeContext {
    pub token: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GameState {
    #[serde(default)]
    pub tick: u64,
    #[serde(default)]
    pub grid: Grid,
    #[serde(default)]
    pub bots: Vec<BotState>,
    #[serde(default)]
    pub items: Vec<Item>,
    #[serde(default)]
    pub orders: Vec<Order>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GameOver {
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub final_score: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Grid {
    #[serde(default)]
    pub width: i32,
    #[serde(default)]
    pub height: i32,
    #[serde(default)]
    pub walls: Vec<[i32; 2]>,
    #[serde(default)]
    pub drop_off_tiles: Vec<[i32; 2]>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BotState {
    pub id: String,
    #[serde(default)]
    pub x: i32,
    #[serde(default)]
    pub y: i32,
    #[serde(default)]
    pub carrying: Vec<String>,
    #[serde(default = "default_capacity")]
    pub capacity: usize,
}

fn default_capacity() -> usize {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Item {
    pub id: String,
    #[serde(default)]
    pub x: i32,
    #[serde(default)]
    pub y: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Order {
    pub id: String,
    pub item_id: String,
    #[serde(default)]
    pub status: OrderStatus,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OrderStatus {
    #[default]
    Pending,
    InProgress,
    Delivered,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Action {
    Move { bot_id: String, dx: i32, dy: i32 },
    PickUp { bot_id: String, item_id: String },
    DropOff { bot_id: String, order_id: String },
    Wait { bot_id: String },
}

impl Action {
    pub fn wait(bot_id: impl Into<String>) -> Self {
        Self::Wait {
            bot_id: bot_id.into(),
        }
    }

    pub fn bot_id(&self) -> &str {
        match self {
            Self::Move { bot_id, .. }
            | Self::PickUp { bot_id, .. }
            | Self::DropOff { bot_id, .. }
            | Self::Wait { bot_id } => bot_id,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ActionEnvelope {
    pub actions: Vec<Action>,
}
