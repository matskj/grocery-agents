use std::{
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::model::{Action, GameState};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayTickFrame {
    pub tick: u64,
    pub game_state: GameState,
    pub actions: Vec<Action>,
    pub team_summary: serde_json::Value,
    pub tick_outcome: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayRun {
    pub run_id: String,
    pub path: PathBuf,
    pub mode: Option<String>,
    pub frames: Vec<ReplayTickFrame>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayRunMeta {
    pub run_id: String,
    pub file_name: String,
    pub mode: Option<String>,
    pub ticks: usize,
    pub final_score: i64,
}

pub fn load_run(path: &Path) -> Result<ReplayRun, Box<dyn std::error::Error>> {
    let raw = fs::read_to_string(path)?;
    let mut run_id = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("run")
        .to_owned();
    let mut mode = None::<String>;
    let mut frames = Vec::<ReplayTickFrame>::new();

    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let event = value
            .get("event")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        let data = value
            .get("data")
            .cloned()
            .unwrap_or(serde_json::Value::Null);

        if event == "log_opened" {
            if let Some(id) = value.get("run_id").and_then(serde_json::Value::as_str) {
                run_id = id.to_owned();
            }
        }

        if event == "game_mode" {
            mode = data
                .get("mode")
                .and_then(serde_json::Value::as_str)
                .map(|s| s.to_owned());
        }

        if event != "tick" {
            continue;
        }

        let tick = data
            .get("tick")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let Some(game_state_value) = data.get("game_state") else {
            continue;
        };
        let Ok(game_state) = serde_json::from_value::<GameState>(game_state_value.clone()) else {
            continue;
        };
        let actions = data
            .get("actions")
            .cloned()
            .and_then(|v| serde_json::from_value::<Vec<Action>>(v).ok())
            .unwrap_or_default();

        let team_summary = data
            .get("team_summary")
            .cloned()
            .unwrap_or(serde_json::json!({}));
        let tick_outcome = data
            .get("tick_outcome")
            .cloned()
            .unwrap_or(serde_json::json!({}));

        frames.push(ReplayTickFrame {
            tick,
            game_state,
            actions,
            team_summary,
            tick_outcome,
        });
    }

    Ok(ReplayRun {
        run_id,
        path: path.to_path_buf(),
        mode,
        frames,
    })
}

pub fn list_runs(logs_dir: &Path) -> Result<Vec<ReplayRunMeta>, Box<dyn std::error::Error>> {
    let mut files = fs::read_dir(logs_dir)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|s| s.to_str())
                .map(|name| name.starts_with("run-") && name.ends_with(".jsonl"))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    files.sort();

    let mut out = Vec::new();
    for file in files {
        let run = load_run(&file)?;
        let final_score = run.frames.last().map(|f| f.game_state.score).unwrap_or(0);
        out.push(ReplayRunMeta {
            run_id: run.run_id,
            file_name: file
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_owned(),
            mode: run.mode,
            ticks: run.frames.len(),
            final_score,
        });
    }
    out.sort_by(|a, b| a.file_name.cmp(&b.file_name));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::load_run;

    #[test]
    fn parses_tick_frames_from_jsonl() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!("replay-parse-{suffix}.jsonl"));
        let content = vec![
            serde_json::json!({"event":"log_opened","run_id":"run-x"}).to_string(),
            serde_json::json!({"event":"game_mode","data":{"mode":"easy"}}).to_string(),
            serde_json::json!({
                "event":"tick",
                "data":{
                    "tick":0,
                    "game_state":{"tick":0,"score":0,"active_order_index":0,"grid":{"width":2,"height":2,"walls":[],"drop_off_tiles":[[0,0]]},"bots":[],"items":[],"orders":[]},
                    "actions":[],
                    "team_summary":{},
                    "tick_outcome":{"delta_score":0}
                }
            })
            .to_string(),
        ]
        .join("\n");
        fs::write(&path, content).expect("fixture");

        let run = load_run(&path).expect("parse run");
        assert_eq!(run.run_id, "run-x");
        assert_eq!(run.mode.as_deref(), Some("easy"));
        assert_eq!(run.frames.len(), 1);
        let _ = fs::remove_file(path);
    }
}
