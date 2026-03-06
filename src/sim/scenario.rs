use serde::{Deserialize, Serialize};

use crate::{difficulty::infer_difficulty, model::GameState, replay::ReplayRun};

use super::{orders::OrderGenerator, state::SimState};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForkRequest {
    pub run_id: String,
    pub tick: u64,
    pub seed: u64,
}

pub fn sim_from_replay_fork(run: &ReplayRun, tick: u64, seed: u64) -> Option<SimState> {
    let frame = run.frames.iter().find(|f| f.tick == tick)?;
    let difficulty = infer_difficulty(&frame.game_state);
    let generator = OrderGenerator::for_difficulty(difficulty);
    Some(SimState::new(frame.game_state.clone(), seed, generator))
}

pub fn sim_from_state(state: GameState, seed: u64) -> SimState {
    let difficulty = infer_difficulty(&state);
    let generator = OrderGenerator::for_difficulty(difficulty);
    SimState::new(state, seed, generator)
}

#[cfg(test)]
mod tests {
    use crate::{
        model::{GameState, Grid},
        replay::{ReplayRun, ReplayTickFrame},
    };

    use super::sim_from_replay_fork;

    #[test]
    fn can_fork_from_matching_tick() {
        let state = GameState {
            tick: 10,
            grid: Grid {
                width: 12,
                height: 10,
                drop_off_tiles: vec![[0, 0]],
                ..Grid::default()
            },
            ..GameState::default()
        };
        let run = ReplayRun {
            run_id: "r1".to_owned(),
            path: std::path::PathBuf::from("logs/run-r1.jsonl"),
            mode: Some("easy".to_owned()),
            frames: vec![ReplayTickFrame {
                tick: 10,
                game_state: state.clone(),
                actions: vec![],
                team_summary: serde_json::json!({}),
                tick_outcome: serde_json::json!({}),
            }],
        };

        let sim = sim_from_replay_fork(&run, 10, 123).expect("fork");
        assert_eq!(sim.game_state.tick, state.tick);
        assert_eq!(sim.game_state.grid.width, 12);
    }
}
