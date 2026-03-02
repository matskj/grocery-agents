use std::{sync::OnceLock, time::Duration};

use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpStream;
use tokio::time::timeout;
use tokio_tungstenite::{
    connect_async,
    tungstenite::{self, Message},
    MaybeTlsStream, WebSocketStream,
};
use tracing::{info, warn};

use crate::{
    dispatcher::Dispatcher,
    model::{Action, ActionEnvelope, BotState, GameOver, GameState, RuntimeContext},
    policy::Policy,
};

type WsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;
const ROUND_PLANNING_BUDGET: Duration = Duration::from_millis(1_400);

#[derive(Debug, serde::Deserialize)]
#[serde(untagged)]
enum ServerMessage {
    GameStateEnvelope { game_state: GameState },
    GameState(GameState),
    GameOverEnvelope { game_over: GameOver },
    GameOver(GameOver),
}

pub async fn connect(token: &str) -> Result<WsStream, tungstenite::Error> {
    let base_url =
        std::env::var("GROCERY_WS_URL").unwrap_or_else(|_| "ws://localhost:8765/ws".to_owned());
    let separator = if base_url.contains('?') { '&' } else { '?' };
    let url = format!("{base_url}{separator}token={token}");
    let (stream, _) = connect_async(url).await?;
    Ok(stream)
}

pub async fn run_game_loop(
    ctx: RuntimeContext,
    mut policy: Policy,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut socket = connect(&ctx.token).await?;
    let dispatcher = Dispatcher::new();

    info!("connected websocket, entering receive loop");

    while let Some(frame) = socket.next().await {
        match frame {
            Ok(Message::Text(text)) => {
                let msg = match serde_json::from_str::<ServerMessage>(&text) {
                    Ok(m) => m,
                    Err(err) => {
                        warn!(error = %err, payload = %text, "failed to parse server message");
                        continue;
                    }
                };

                match msg {
                    ServerMessage::GameStateEnvelope { game_state }
                    | ServerMessage::GameState(game_state) => {
                        let planned = timeout(
                            ROUND_PLANNING_BUDGET,
                            plan_round_actions(&mut policy, &dispatcher, &game_state),
                        )
                        .await
                        .unwrap_or_else(|_| {
                            warn!(
                                tick = game_state.tick,
                                "planning timeout, falling back to wait actions"
                            );
                            if bot_debug_enabled() {
                                warn!(
                                    tick = game_state.tick,
                                    budget_ms = ROUND_PLANNING_BUDGET.as_millis(),
                                    bot_count = game_state.bots.len(),
                                    "time-budget event: fallback wait envelope emitted"
                                );
                            }
                            fallback_wait_actions(&game_state)
                        });

                        let envelope = ActionEnvelope { actions: planned };
                        let payload = serde_json::to_string(&envelope)?;
                        socket.send(Message::Text(payload.into())).await?;
                        info!(tick = game_state.tick, "sent round action envelope");
                    }
                    ServerMessage::GameOverEnvelope { game_over }
                    | ServerMessage::GameOver(game_over) => {
                        info!(
                            final_score = game_over.final_score,
                            reason = ?game_over.reason,
                            "game over received"
                        );
                        break;
                    }
                }
            }
            Ok(Message::Ping(payload)) => {
                socket.send(Message::Pong(payload)).await?;
            }
            Ok(Message::Close(frame)) => {
                info!(?frame, "server closed websocket");
                break;
            }
            Ok(Message::Binary(_)) | Ok(Message::Pong(_)) => {}
            Ok(Message::Frame(_)) => {}
            Err(err) => {
                warn!(error = %err, "websocket receive error, terminating loop");
                break;
            }
        }
    }

    Ok(())
}

async fn plan_round_actions(
    policy: &mut Policy,
    dispatcher: &Dispatcher,
    state: &GameState,
) -> Vec<Action> {
    let proposed = policy.decide_round(state);
    state
        .bots
        .iter()
        .zip(proposed.into_iter())
        .map(|(bot, action)| {
            let dispatched = dispatcher.dispatch(action);
            validate_action(dispatched, state, bot)
        })
        .collect()
}

fn fallback_wait_actions(state: &GameState) -> Vec<Action> {
    if bot_debug_enabled() {
        warn!(
            tick = state.tick,
            bot_count = state.bots.len(),
            "safety fallback activated: forcing wait actions"
        );
    }
    state
        .bots
        .iter()
        .map(|bot| Action::wait(bot.id.clone()))
        .collect()
}

fn validate_action(action: Action, state: &GameState, bot: &BotState) -> Action {
    if action.bot_id() != bot.id {
        if bot_debug_enabled() {
            warn!(tick = state.tick, bot_id = %bot.id, "fallback event: action bot_id mismatch; waiting");
        }
        return Action::wait(bot.id.clone());
    }

    match action {
        Action::Move { bot_id, dx, dy } => {
            let nx = bot.x + dx;
            let ny = bot.y + dy;
            let in_bounds = nx >= 0 && ny >= 0 && nx < state.grid.width && ny < state.grid.height;
            let blocked = state
                .grid
                .walls
                .iter()
                .any(|wall| wall[0] == nx && wall[1] == ny);
            if in_bounds && !blocked {
                Action::Move { bot_id, dx, dy }
            } else {
                if bot_debug_enabled() {
                    warn!(tick = state.tick, bot_id = %bot.id, nx, ny, in_bounds, blocked, "safety fallback: invalid move converted to wait");
                }
                Action::wait(bot.id.clone())
            }
        }
        Action::PickUp { bot_id, item_id } => {
            let has_capacity = bot.carrying.len() < bot.capacity;
            let maybe_item = state.items.iter().find(|item| item.id == item_id);
            let adjacent = maybe_item
                .map(|item| (item.x - bot.x).abs() + (item.y - bot.y).abs() == 1)
                .unwrap_or(false);
            if has_capacity && adjacent {
                Action::PickUp { bot_id, item_id }
            } else {
                if bot_debug_enabled() {
                    warn!(tick = state.tick, bot_id = %bot.id, has_capacity, adjacent, "safety fallback: invalid pickup converted to wait");
                }
                Action::wait(bot.id.clone())
            }
        }
        Action::DropOff { bot_id, order_id } => {
            let on_drop_off = state
                .grid
                .drop_off_tiles
                .iter()
                .any(|tile| tile[0] == bot.x && tile[1] == bot.y);
            if on_drop_off {
                Action::DropOff { bot_id, order_id }
            } else {
                if bot_debug_enabled() {
                    warn!(tick = state.tick, bot_id = %bot.id, "safety fallback: invalid dropoff converted to wait");
                }
                Action::wait(bot.id.clone())
            }
        }
        Action::Wait { bot_id } => Action::Wait { bot_id },
    }
}

fn bot_debug_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("BOT_DEBUG")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false)
    })
}
