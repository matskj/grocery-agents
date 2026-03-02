mod dispatcher;
mod dist;
mod model;
mod motion;
mod net;
mod policy;
mod world;

use clap::Parser;
use model::RuntimeContext;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Debug, Parser)]
#[command(name = "grocery-agents")]
struct Cli {
    #[arg(long, env = "GROCERY_TOKEN")]
    token: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();
    let ctx = RuntimeContext { token: cli.token };
    let policy = policy::Policy::new();

    net::run_game_loop(ctx, policy).await
}
