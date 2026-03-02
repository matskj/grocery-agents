use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeContext {
    pub token: String,
    pub ws_url: Option<String>,
    pub session: SessionMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SessionMetadata {
    pub team_id: Option<String>,
    pub map_id: Option<String>,
    pub difficulty: Option<String>,
    pub map_seed: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GameState {
    #[serde(default)]
    pub tick: u64,
    #[serde(default)]
    pub score: i64,
    #[serde(default)]
    pub active_order_index: i64,
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
    3
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Item {
    pub id: String,
    #[serde(default)]
    pub kind: String,
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

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WireServerMessage {
    GameState(WireGameState),
    GameOver(WireGameOver),
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct WireGameState {
    #[serde(default)]
    pub round: u64,
    #[serde(default)]
    pub score: i64,
    #[serde(default)]
    pub active_order_index: i64,
    #[serde(default)]
    pub grid: WireGrid,
    #[serde(default)]
    pub bots: Vec<WireBotState>,
    #[serde(default)]
    pub items: Vec<WireItem>,
    #[serde(default)]
    pub orders: Vec<WireOrder>,
    #[serde(default)]
    pub drop_off: Option<[i32; 2]>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct WireGrid {
    #[serde(default)]
    pub width: i32,
    #[serde(default)]
    pub height: i32,
    #[serde(default)]
    pub walls: Vec<[i32; 2]>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct WireBotState {
    #[serde(default)]
    pub id: WireBotId,
    #[serde(default)]
    pub position: [i32; 2],
    #[serde(default)]
    pub inventory: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct WireItem {
    #[serde(default)]
    pub id: String,
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub position: [i32; 2],
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct WireOrder {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub items_required: Vec<String>,
    #[serde(default)]
    pub items_delivered: Vec<String>,
    #[serde(default)]
    pub complete: bool,
    #[serde(default)]
    pub status: WireOrderStatus,
}

#[derive(Debug, Clone, Copy, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum WireOrderStatus {
    Active,
    Preview,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct WireGameOver {
    #[serde(default)]
    pub score: Option<i64>,
    #[serde(default)]
    pub final_score: Option<i64>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub rounds_used: Option<u64>,
    #[serde(default)]
    pub items_delivered: Option<u64>,
    #[serde(default)]
    pub orders_completed: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WireActionEnvelope {
    pub actions: Vec<WireAction>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WireAction {
    pub bot: WireBotId,
    pub action: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub item_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(untagged)]
pub enum WireBotId {
    Int(i64),
    Str(String),
    #[default]
    Empty,
}

impl WireBotId {
    fn as_key(&self) -> String {
        match self {
            Self::Int(v) => v.to_string(),
            Self::Str(v) => v.clone(),
            Self::Empty => String::new(),
        }
    }

    fn from_bot_key(bot_id: &str) -> Self {
        bot_id
            .parse::<i64>()
            .map(Self::Int)
            .unwrap_or_else(|_| Self::Str(bot_id.to_owned()))
    }
}

impl GameState {
    pub fn from_wire(wire: WireGameState) -> Self {
        let mut drop_off_tiles = Vec::new();
        if let Some(tile) = wire.drop_off {
            drop_off_tiles.push(tile);
        }

        let bots = wire
            .bots
            .into_iter()
            .map(|bot| BotState {
                id: bot.id.as_key(),
                x: bot.position[0],
                y: bot.position[1],
                carrying: bot.inventory,
                capacity: 3,
            })
            .collect();

        let items = wire
            .items
            .into_iter()
            .map(|item| Item {
                id: item.id.clone(),
                kind: if item.kind.is_empty() {
                    item.id
                } else {
                    item.kind
                },
                x: item.position[0],
                y: item.position[1],
            })
            .collect();

        let mut orders = Vec::new();
        for order in wire.orders {
            if order.complete {
                continue;
            }
            let status = match order.status {
                WireOrderStatus::Active => OrderStatus::InProgress,
                WireOrderStatus::Preview => OrderStatus::Pending,
                WireOrderStatus::Unknown => OrderStatus::Pending,
            };

            let missing = missing_item_counts(order.items_required, order.items_delivered);
            for (item_id, count) in missing {
                for idx in 0..count {
                    orders.push(Order {
                        id: format!("{}:{}:{}", order.id, item_id, idx),
                        item_id: item_id.clone(),
                        status,
                    });
                }
            }
        }

        Self {
            tick: wire.round,
            score: wire.score,
            active_order_index: wire.active_order_index,
            grid: Grid {
                width: wire.grid.width,
                height: wire.grid.height,
                walls: wire.grid.walls,
                drop_off_tiles,
            },
            bots,
            items,
            orders,
        }
    }
}

impl GameOver {
    pub fn from_wire(wire: WireGameOver) -> Self {
        let final_score = wire.final_score.or(wire.score).unwrap_or_default();
        let reason = wire.reason.or_else(|| {
            if wire.rounds_used.is_none()
                && wire.items_delivered.is_none()
                && wire.orders_completed.is_none()
            {
                None
            } else {
                Some(format!(
                    "rounds_used={:?}, items_delivered={:?}, orders_completed={:?}",
                    wire.rounds_used, wire.items_delivered, wire.orders_completed
                ))
            }
        });

        Self {
            reason,
            final_score,
        }
    }
}

pub fn to_wire_action_envelope(actions: &[Action]) -> WireActionEnvelope {
    let actions = actions
        .iter()
        .map(|action| match action {
            Action::Move { bot_id, dx, dy } => WireAction {
                bot: WireBotId::from_bot_key(bot_id),
                action: move_action(*dx, *dy),
                item_id: None,
            },
            Action::PickUp { bot_id, item_id } => WireAction {
                bot: WireBotId::from_bot_key(bot_id),
                action: "pick_up",
                item_id: Some(item_id.clone()),
            },
            Action::DropOff { bot_id, .. } => WireAction {
                bot: WireBotId::from_bot_key(bot_id),
                action: "drop_off",
                item_id: None,
            },
            Action::Wait { bot_id } => WireAction {
                bot: WireBotId::from_bot_key(bot_id),
                action: "wait",
                item_id: None,
            },
        })
        .collect();

    WireActionEnvelope { actions }
}

fn move_action(dx: i32, dy: i32) -> &'static str {
    match (dx, dy) {
        (1, 0) => "move_right",
        (-1, 0) => "move_left",
        (0, 1) => "move_down",
        (0, -1) => "move_up",
        _ => "wait",
    }
}

fn missing_item_counts(required: Vec<String>, delivered: Vec<String>) -> HashMap<String, u16> {
    let mut needed: HashMap<String, u16> = HashMap::new();
    for item in required {
        *needed.entry(item).or_insert(0) += 1;
    }
    for item in delivered {
        if let Some(count) = needed.get_mut(&item) {
            if *count > 0 {
                *count -= 1;
            }
        }
    }
    needed.retain(|_, count| *count > 0);
    needed
}

#[cfg(test)]
mod tests {
    use super::{
        to_wire_action_envelope, Action, GameState, WireActionEnvelope, WireGameState, WireGrid,
        WireItem, WireOrder, WireOrderStatus,
    };

    #[test]
    fn converts_wire_game_state() {
        let wire = WireGameState {
            round: 12,
            score: 7,
            active_order_index: 2,
            grid: WireGrid {
                width: 12,
                height: 10,
                walls: vec![[1, 1]],
            },
            bots: vec![],
            items: vec![WireItem {
                id: "inst_1".to_owned(),
                kind: "apple".to_owned(),
                position: [3, 3],
            }],
            orders: vec![WireOrder {
                id: "order_1".to_owned(),
                items_required: vec!["apple".to_owned(), "apple".to_owned()],
                items_delivered: vec!["apple".to_owned()],
                complete: false,
                status: WireOrderStatus::Active,
            }],
            drop_off: Some([0, 0]),
        };

        let state = GameState::from_wire(wire);
        assert_eq!(state.tick, 12);
        assert_eq!(state.score, 7);
        assert_eq!(state.active_order_index, 2);
        assert_eq!(state.items.len(), 1);
        assert_eq!(state.orders.len(), 1);
        assert_eq!(state.grid.drop_off_tiles, vec![[0, 0]]);
    }

    #[test]
    fn serializes_wire_actions() {
        let envelope: WireActionEnvelope = to_wire_action_envelope(&[
            Action::Move {
                bot_id: "1".to_owned(),
                dx: 1,
                dy: 0,
            },
            Action::PickUp {
                bot_id: "2".to_owned(),
                item_id: "i9".to_owned(),
            },
        ]);

        let json = serde_json::to_value(&envelope).expect("serialize envelope");
        assert_eq!(json["actions"][0]["action"], "move_right");
        assert_eq!(json["actions"][1]["action"], "pick_up");
        assert_eq!(json["actions"][1]["item_id"], "i9");
    }
}
