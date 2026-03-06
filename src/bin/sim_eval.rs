use std::{path::PathBuf, process::ExitCode};

use clap::Parser;
use grocery_agents::{
    config::PolicyMode,
    sim::eval::{run_batch_eval, EvalRequest},
};

#[derive(Debug, Parser)]
#[command(name = "sim_eval")]
struct Cli {
    #[arg(long, default_value = "logs")]
    logs_dir: PathBuf,
    #[arg(long, default_value_t = 20)]
    episodes: usize,
    #[arg(long, value_enum, default_value_t = PolicyMode::Auto)]
    policy: PolicyMode,
    #[arg(long)]
    difficulty: Option<String>,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long)]
    out: Option<PathBuf>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let request = EvalRequest {
        logs_dir: &cli.logs_dir,
        episodes: cli.episodes,
        policy: cli.policy,
        difficulty: cli.difficulty.as_deref(),
        seed: cli.seed,
    };

    let report = match run_batch_eval(request) {
        Ok(report) => report,
        Err(err) => {
            eprintln!("sim_eval failed: {err}");
            return ExitCode::from(1);
        }
    };

    let json = match serde_json::to_string_pretty(&report) {
        Ok(s) => s,
        Err(err) => {
            eprintln!("failed to serialize report: {err}");
            return ExitCode::from(1);
        }
    };

    if let Some(out) = cli.out {
        if let Err(err) = std::fs::write(&out, &json) {
            eprintln!("failed to write report to {}: {err}", out.display());
            return ExitCode::from(1);
        }
        println!("wrote report: {}", out.display());
    } else {
        println!("{json}");
    }

    ExitCode::SUCCESS
}
