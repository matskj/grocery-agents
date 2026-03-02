use std::{collections::HashMap, fs, sync::OnceLock};

use serde::Deserialize;

use crate::model::GameState;

#[derive(Debug, Clone, Deserialize, Default)]
pub struct PolicyArtifact {
    #[serde(default)]
    pub schema_version: String,
    #[serde(default)]
    pub modes: HashMap<String, ModeModel>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct ModeModel {
    #[serde(default)]
    pub weights: HashMap<String, f64>,
    #[serde(default)]
    pub ordering_weights: HashMap<String, f64>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CandidateFeatures {
    pub dist_to_nearest_active_item: f64,
    pub dist_to_dropoff: f64,
    pub inventory_util: f64,
    pub local_congestion: f64,
    pub teammate_proximity: f64,
    pub order_urgency: f64,
    pub blocked_ticks: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct OrderingFeatures {
    pub carrying_active: f64,
    pub queue_role_lead: f64,
    pub queue_role_courier: f64,
    pub blocked_ticks: f64,
    pub local_conflict_count: f64,
    pub dist_to_goal: f64,
    pub dropoff_watchdog_pressure: f64,
    pub choke_occupancy: f64,
}

pub fn detect_mode_label(state: &GameState) -> &'static str {
    match (state.bots.len(), state.grid.width, state.grid.height) {
        (1, 12, 10) => "easy",
        (3, 16, 12) => "medium",
        (5, 22, 14) => "hard",
        (10, 28, 18) => "expert",
        _ => "custom",
    }
}

pub fn maybe_score_pick(mode: &str, features: CandidateFeatures) -> Option<f64> {
    let artifact = load_artifact();
    let model = artifact.modes.get(mode)?;
    Some(score_with_weights(&model.weights, features))
}

pub fn maybe_score_ordering(mode: &str, features: OrderingFeatures) -> Option<f64> {
    let artifact = load_artifact();
    let model = artifact.modes.get(mode)?;
    Some(score_with_ordering_weights(&model.ordering_weights, features))
}

fn score_with_weights(weights: &HashMap<String, f64>, features: CandidateFeatures) -> f64 {
    let mut score = *weights.get("bias").unwrap_or(&0.0);
    score += features.dist_to_nearest_active_item
        * *weights.get("dist_to_nearest_active_item").unwrap_or(&0.0);
    score += features.dist_to_dropoff * *weights.get("dist_to_dropoff").unwrap_or(&0.0);
    score += features.inventory_util * *weights.get("inventory_util").unwrap_or(&0.0);
    score += features.local_congestion * *weights.get("local_congestion").unwrap_or(&0.0);
    score += features.teammate_proximity * *weights.get("teammate_proximity").unwrap_or(&0.0);
    score += features.order_urgency * *weights.get("order_urgency").unwrap_or(&0.0);
    score += features.blocked_ticks * *weights.get("blocked_ticks").unwrap_or(&0.0);
    score
}

fn score_with_ordering_weights(weights: &HashMap<String, f64>, features: OrderingFeatures) -> f64 {
    let mut score = *weights.get("bias").unwrap_or(&0.0);
    score += features.carrying_active * *weights.get("carrying_active").unwrap_or(&0.0);
    score += features.queue_role_lead * *weights.get("queue_role_lead").unwrap_or(&0.0);
    score += features.queue_role_courier * *weights.get("queue_role_courier").unwrap_or(&0.0);
    score += features.blocked_ticks * *weights.get("blocked_ticks").unwrap_or(&0.0);
    score += features.local_conflict_count * *weights.get("local_conflict_count").unwrap_or(&0.0);
    score += features.dist_to_goal * *weights.get("dist_to_goal").unwrap_or(&0.0);
    score += features.dropoff_watchdog_pressure
        * *weights
            .get("dropoff_watchdog_pressure")
            .unwrap_or(&0.0);
    score += features.choke_occupancy * *weights.get("choke_occupancy").unwrap_or(&0.0);
    score
}

fn load_artifact() -> &'static PolicyArtifact {
    static MODEL: OnceLock<PolicyArtifact> = OnceLock::new();
    MODEL.get_or_init(|| {
        let path = match std::env::var("POLICY_ARTIFACT_PATH") {
            Ok(v) => v,
            Err(_) => return PolicyArtifact::default(),
        };
        let content = match fs::read_to_string(path) {
            Ok(v) => v,
            Err(_) => return PolicyArtifact::default(),
        };
        serde_json::from_str::<PolicyArtifact>(&content).unwrap_or_default()
    })
}

#[cfg(test)]
mod tests {
    use crate::model::{BotState, GameState, Grid};

    use super::detect_mode_label;

    fn state(width: i32, height: i32, bots: usize) -> GameState {
        GameState {
            grid: Grid {
                width,
                height,
                ..Grid::default()
            },
            bots: (0..bots)
                .map(|idx| BotState {
                    id: idx.to_string(),
                    x: 0,
                    y: 0,
                    carrying: Vec::new(),
                    capacity: 3,
                })
                .collect(),
            ..GameState::default()
        }
    }

    #[test]
    fn detects_standard_modes() {
        assert_eq!(detect_mode_label(&state(12, 10, 1)), "easy");
        assert_eq!(detect_mode_label(&state(16, 12, 3)), "medium");
        assert_eq!(detect_mode_label(&state(22, 14, 5)), "hard");
        assert_eq!(detect_mode_label(&state(28, 18, 10)), "expert");
    }

    #[test]
    fn detects_custom_mode() {
        assert_eq!(detect_mode_label(&state(13, 11, 4)), "custom");
    }
}
