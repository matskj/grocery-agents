use std::time::Duration;

use futures_util::SinkExt;
use tokio::time::sleep;
use tokio_tungstenite::tungstenite::Message;
use tracing::info;

use crate::{dispatcher::Dispatcher, model::RuntimeContext, motion, policy::Policy, world::World};

pub async fn run_game_loop(
    ctx: RuntimeContext,
    policy: Policy,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut world = World::new(Default::default());
    let dispatcher = Dispatcher::new();

    info!(token_len = ctx.token.len(), "starting game loop");

    for _ in 0..5 {
        motion::advance(&mut world);
        let action = policy.decide(world.state());
        let action = dispatcher.dispatch(action);

        let payload = serde_json::to_string(&action)?;
        let mut sink = futures_util::sink::drain();
        sink.send(Message::Text(payload.into())).await?;

        info!(tick = world.state().tick, "processed tick");
        sleep(Duration::from_millis(50)).await;
    }

    Ok(())
}
