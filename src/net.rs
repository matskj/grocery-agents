use crate::{model::RuntimeContext, policy::Policy};

pub async fn run_game_loop(
    _ctx: RuntimeContext,
    _policy: Policy,
) -> Result<(), Box<dyn std::error::Error>> {
    Err(
        std::io::Error::other("websocket sink is not configured; refusing to run with a drop sink")
            .into(),
    )
}
