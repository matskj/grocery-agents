use std::{
    pin::Pin,
    task::{Context, Poll},
    time::Duration,
};

use futures_util::{Sink, SinkExt};
use tokio::time::sleep;
use tokio_tungstenite::tungstenite::Message;
use tracing::{debug, info};

use crate::{dispatcher::Dispatcher, model::RuntimeContext, motion, policy::Policy, world::World};

#[derive(Default)]
struct LoggingSink;

impl Sink<Message> for LoggingSink {
    type Error = std::io::Error;

    fn poll_ready(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn start_send(self: Pin<&mut Self>, item: Message) -> Result<(), Self::Error> {
        debug!(payload = ?item, "queued outbound action");
        Ok(())
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn poll_close(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }
}

pub async fn run_game_loop(
    ctx: RuntimeContext,
    policy: Policy,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut sink = LoggingSink;
    run_game_loop_with_sink(ctx, policy, Some(&mut sink)).await
}

pub async fn run_game_loop_with_sink<S>(
    ctx: RuntimeContext,
    policy: Policy,
    mut websocket_sink: Option<&mut S>,
) -> Result<(), Box<dyn std::error::Error>>
where
    S: Sink<Message> + Unpin,
    S::Error: std::error::Error + Send + Sync + 'static,
{
    let sink = websocket_sink
        .as_deref_mut()
        .ok_or_else(|| std::io::Error::other("websocket sink is not configured"))?;

    let mut world = World::new(Default::default());
    let dispatcher = Dispatcher::new();

    info!(token_len = ctx.token.len(), "starting game loop");

    for _ in 0..5 {
        motion::advance(&mut world);
        let action = policy.decide(world.state());
        let action = dispatcher.dispatch(action);

        let payload = serde_json::to_string(&action)?;
        sink.send(Message::Text(payload.into())).await?;

        info!(tick = world.state().tick, "processed tick");
        sleep(Duration::from_millis(50)).await;
    }

    Ok(())
}
