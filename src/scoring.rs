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
    #[serde(default)]
    pub feature_columns: Vec<String>,
    #[serde(default)]
    pub runtime_feature_columns: Vec<String>,
    #[serde(default)]
    pub normalization: NormalizationModel,
    #[serde(default)]
    pub heads: HashMap<String, HeadModel>,
    #[serde(default)]
    pub calibration: CalibrationModel,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct NormalizationModel {
    #[serde(default)]
    pub mean: Vec<f64>,
    #[serde(default)]
    pub std: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HeadModel {
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub bias: f64,
    #[serde(default)]
    pub weights: Vec<f64>,
    #[serde(default = "default_clip")]
    pub clip: f64,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
}

impl Default for HeadModel {
    fn default() -> Self {
        Self {
            kind: String::new(),
            bias: 0.0,
            weights: Vec::new(),
            clip: default_clip(),
            temperature: default_temperature(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct CalibrationModel {
    #[serde(default = "default_temperature")]
    pub pickup_temp: f64,
    #[serde(default = "default_temperature")]
    pub dropoff_temp: f64,
}

impl Default for CalibrationModel {
    fn default() -> Self {
        Self {
            pickup_temp: default_temperature(),
            dropoff_temp: default_temperature(),
        }
    }
}

fn default_clip() -> f64 {
    8.0
}

fn default_temperature() -> f64 {
    1.0
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CandidateFeatures {
    pub dist_to_nearest_active_item: f64,
    pub dist_to_dropoff: f64,
    pub inventory_util: f64,
    pub queue_distance: f64,
    pub local_congestion: f64,
    pub local_conflict_count: f64,
    pub teammate_proximity: f64,
    pub order_urgency: f64,
    pub blocked_ticks: f64,
    pub queue_role_lead: f64,
    pub queue_role_courier: f64,
    pub queue_role_collector: f64,
    pub queue_role_yield: f64,
    pub serviceable_dropoff: f64,
    pub stand_failure_count_recent: f64,
    pub stand_success_count_recent: f64,
    pub stand_cooldown_ticks_remaining: f64,
    pub kind_failure_count_recent: f64,
    pub repeated_same_stand_no_delta_streak: f64,
    pub contention_at_stand_proxy: f64,
    pub time_since_last_conversion_tick: f64,
    pub last_conversion_was_pickup: f64,
    pub last_conversion_was_dropoff: f64,
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

#[derive(Debug, Clone, Copy, Default)]
pub struct PickScore {
    pub pickup_prob: f64,
    pub dropoff_prob: f64,
    pub ordering_score: f64,
    pub combined_expected_score: f64,
    pub legacy_pick_score: f64,
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

pub fn maybe_score_pick(mode: &str, features: CandidateFeatures) -> Option<PickScore> {
    let artifact = load_artifact();
    let model = artifact.modes.get(mode)?;
    Some(score_pick_with_model(model, features))
}

pub fn maybe_score_ordering(mode: &str, features: OrderingFeatures) -> Option<f64> {
    let artifact = load_artifact();
    let model = artifact.modes.get(mode)?;
    Some(score_with_ordering_weights(
        &model.ordering_weights,
        features,
    ))
}

fn score_pick_with_model(model: &ModeModel, features: CandidateFeatures) -> PickScore {
    let legacy_pick_score = score_with_weights(&model.weights, features);
    let Some(normalized_features) = normalized_feature_vector(model, features) else {
        let legacy = clamp(legacy_pick_score, -200.0, 200.0);
        return PickScore {
            pickup_prob: 1.0,
            dropoff_prob: 1.0,
            ordering_score: legacy,
            combined_expected_score: legacy,
            legacy_pick_score: legacy,
        };
    };

    let pickup_head = model.heads.get("pickup").cloned().unwrap_or_default();
    let dropoff_head = model.heads.get("dropoff").cloned().unwrap_or_default();
    let ordering_head = model.heads.get("ordering").cloned().unwrap_or_default();

    let pickup_temp = if pickup_head.temperature > 0.0 {
        pickup_head.temperature
    } else {
        model.calibration.pickup_temp
    };
    let dropoff_temp = if dropoff_head.temperature > 0.0 {
        dropoff_head.temperature
    } else {
        model.calibration.dropoff_temp
    };
    let pickup_prob =
        head_probability(&pickup_head, &normalized_features, pickup_temp).unwrap_or(1.0);
    let dropoff_prob =
        head_probability(&dropoff_head, &normalized_features, dropoff_temp).unwrap_or(1.0);
    let ordering_score =
        head_linear_score(&ordering_head, &normalized_features).unwrap_or_else(|| {
            if legacy_pick_score.abs() > 1e-9 {
                legacy_pick_score
            } else {
                features.order_urgency
            }
        });
    let value_proxy = if ordering_score.abs() > 1e-9 {
        ordering_score.max(0.0)
    } else {
        features.order_urgency.max(0.0)
    };
    let eta_proxy = clamp(
        features.dist_to_nearest_active_item + 0.7 * features.dist_to_dropoff,
        0.0,
        160.0,
    );
    let combined = value_proxy * pickup_prob * dropoff_prob / (eta_proxy + 1.0);

    PickScore {
        pickup_prob: clamp(pickup_prob, 0.0, 1.0),
        dropoff_prob: clamp(dropoff_prob, 0.0, 1.0),
        ordering_score: clamp(ordering_score, -1_000.0, 1_000.0),
        combined_expected_score: clamp(combined, -1_000.0, 1_000.0),
        legacy_pick_score: clamp(legacy_pick_score, -1_000.0, 1_000.0),
    }
}

fn normalized_feature_vector(model: &ModeModel, features: CandidateFeatures) -> Option<Vec<f64>> {
    let columns = model_feature_columns(model);
    if columns.is_empty() {
        return None;
    }
    let mut out = Vec::with_capacity(columns.len());
    for (idx, col) in columns.iter().enumerate() {
        let raw = candidate_feature_value(col, features);
        let mean = model.normalization.mean.get(idx).copied().unwrap_or(0.0);
        let std = model.normalization.std.get(idx).copied().unwrap_or(1.0);
        let safe_std = if std.abs() <= 1e-9 { 1.0 } else { std };
        out.push((raw - mean) / safe_std);
    }
    Some(out)
}

fn model_feature_columns(model: &ModeModel) -> &[String] {
    if !model.runtime_feature_columns.is_empty() {
        &model.runtime_feature_columns
    } else {
        &model.feature_columns
    }
}

fn candidate_feature_value(name: &str, features: CandidateFeatures) -> f64 {
    match name {
        "dist_to_nearest_active_item" => features.dist_to_nearest_active_item,
        "dist_to_dropoff" => features.dist_to_dropoff,
        "inventory_util" => features.inventory_util,
        "queue_distance" => features.queue_distance,
        "local_congestion" => features.local_congestion,
        "local_conflict_count" => features.local_conflict_count,
        "teammate_proximity" => features.teammate_proximity,
        "order_urgency" => features.order_urgency,
        "blocked_ticks" => features.blocked_ticks,
        "queue_role_lead" => features.queue_role_lead,
        "queue_role_courier" => features.queue_role_courier,
        "queue_role_collector" => features.queue_role_collector,
        "queue_role_yield" => features.queue_role_yield,
        "serviceable_dropoff" => features.serviceable_dropoff,
        "stand_failure_count_recent" => features.stand_failure_count_recent,
        "stand_success_count_recent" => features.stand_success_count_recent,
        "stand_cooldown_ticks_remaining" => features.stand_cooldown_ticks_remaining,
        "kind_failure_count_recent" => features.kind_failure_count_recent,
        "repeated_same_stand_no_delta_streak" => features.repeated_same_stand_no_delta_streak,
        "contention_at_stand_proxy" => features.contention_at_stand_proxy,
        "time_since_last_conversion_tick" => features.time_since_last_conversion_tick,
        "last_conversion_was_pickup" => features.last_conversion_was_pickup,
        "last_conversion_was_dropoff" => features.last_conversion_was_dropoff,
        _ => 0.0,
    }
}

fn head_linear_score(head: &HeadModel, x: &[f64]) -> Option<f64> {
    if head.weights.len() != x.len() {
        return None;
    }
    let mut value = head.bias;
    for (w, xv) in head.weights.iter().zip(x.iter()) {
        value += *w * *xv;
    }
    let clip = if head.clip <= 0.0 {
        default_clip()
    } else {
        head.clip
    };
    Some(clamp(value, -clip, clip))
}

fn head_probability(head: &HeadModel, x: &[f64], temperature: f64) -> Option<f64> {
    let linear = head_linear_score(head, x)?;
    let temp = clamp(temperature, 0.25, 8.0);
    Some(sigmoid(linear / temp))
}

fn sigmoid(v: f64) -> f64 {
    1.0 / (1.0 + (-clamp(v, -24.0, 24.0)).exp())
}

fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}

fn score_with_weights(weights: &HashMap<String, f64>, features: CandidateFeatures) -> f64 {
    let mut score = *weights.get("bias").unwrap_or(&0.0);
    score += features.dist_to_nearest_active_item
        * *weights.get("dist_to_nearest_active_item").unwrap_or(&0.0);
    score += features.dist_to_dropoff * *weights.get("dist_to_dropoff").unwrap_or(&0.0);
    score += features.inventory_util * *weights.get("inventory_util").unwrap_or(&0.0);
    score += features.queue_distance * *weights.get("queue_distance").unwrap_or(&0.0);
    score += features.local_congestion * *weights.get("local_congestion").unwrap_or(&0.0);
    score += features.local_conflict_count * *weights.get("local_conflict_count").unwrap_or(&0.0);
    score += features.teammate_proximity * *weights.get("teammate_proximity").unwrap_or(&0.0);
    score += features.order_urgency * *weights.get("order_urgency").unwrap_or(&0.0);
    score += features.blocked_ticks * *weights.get("blocked_ticks").unwrap_or(&0.0);
    score += features.queue_role_lead * *weights.get("queue_role_lead").unwrap_or(&0.0);
    score += features.queue_role_courier * *weights.get("queue_role_courier").unwrap_or(&0.0);
    score += features.queue_role_collector * *weights.get("queue_role_collector").unwrap_or(&0.0);
    score += features.queue_role_yield * *weights.get("queue_role_yield").unwrap_or(&0.0);
    score += features.serviceable_dropoff * *weights.get("serviceable_dropoff").unwrap_or(&0.0);
    score += features.stand_failure_count_recent
        * *weights.get("stand_failure_count_recent").unwrap_or(&0.0);
    score += features.stand_success_count_recent
        * *weights.get("stand_success_count_recent").unwrap_or(&0.0);
    score += features.stand_cooldown_ticks_remaining
        * *weights
            .get("stand_cooldown_ticks_remaining")
            .unwrap_or(&0.0);
    score += features.kind_failure_count_recent
        * *weights.get("kind_failure_count_recent").unwrap_or(&0.0);
    score += features.repeated_same_stand_no_delta_streak
        * *weights
            .get("repeated_same_stand_no_delta_streak")
            .unwrap_or(&0.0);
    score += features.contention_at_stand_proxy
        * *weights.get("contention_at_stand_proxy").unwrap_or(&0.0);
    score += features.time_since_last_conversion_tick
        * *weights
            .get("time_since_last_conversion_tick")
            .unwrap_or(&0.0);
    score += features.last_conversion_was_pickup
        * *weights.get("last_conversion_was_pickup").unwrap_or(&0.0);
    score += features.last_conversion_was_dropoff
        * *weights.get("last_conversion_was_dropoff").unwrap_or(&0.0);
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
        * *weights.get("dropoff_watchdog_pressure").unwrap_or(&0.0);
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

    use super::{
        detect_mode_label, score_pick_with_model, CandidateFeatures, HeadModel, ModeModel,
    };

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

    #[test]
    fn v1_fallback_uses_legacy_score() {
        let mut model = ModeModel::default();
        model.weights.insert("bias".to_owned(), 7.0);
        model
            .weights
            .insert("dist_to_nearest_active_item".to_owned(), -1.0);
        let features = CandidateFeatures {
            dist_to_nearest_active_item: 2.0,
            ..CandidateFeatures::default()
        };
        let score = score_pick_with_model(&model, features);
        assert!((score.legacy_pick_score - 5.0).abs() < 1e-9);
        assert!((score.combined_expected_score - 5.0).abs() < 1e-9);
        assert_eq!(score.pickup_prob, 1.0);
    }

    #[test]
    fn v2_combined_score_is_deterministic_and_clamped() {
        let mut model = ModeModel {
            runtime_feature_columns: vec![
                "dist_to_nearest_active_item".to_owned(),
                "order_urgency".to_owned(),
            ],
            ..ModeModel::default()
        };
        model.normalization.mean = vec![0.0, 0.0];
        model.normalization.std = vec![1.0, 1.0];
        model.heads.insert(
            "pickup".to_owned(),
            HeadModel {
                kind: "ridge_logit".to_owned(),
                bias: 2.0,
                weights: vec![0.0, 0.0],
                clip: 8.0,
                temperature: 1.0,
            },
        );
        model.heads.insert(
            "dropoff".to_owned(),
            HeadModel {
                kind: "ridge_logit".to_owned(),
                bias: 1.0,
                weights: vec![0.0, 0.0],
                clip: 8.0,
                temperature: 1.0,
            },
        );
        model.heads.insert(
            "ordering".to_owned(),
            HeadModel {
                kind: "ridge".to_owned(),
                bias: 3.0,
                weights: vec![0.0, 10_000.0],
                clip: 12.0,
                temperature: 1.0,
            },
        );
        let features = CandidateFeatures {
            dist_to_nearest_active_item: 0.0,
            order_urgency: 9.0,
            ..CandidateFeatures::default()
        };
        let a = score_pick_with_model(&model, features);
        let b = score_pick_with_model(&model, features);
        assert!((a.combined_expected_score - b.combined_expected_score).abs() < 1e-12);
        assert!(a.combined_expected_score.is_finite());
        assert!(a.ordering_score <= 12.0);
        assert!((0.0..=1.0).contains(&a.pickup_prob));
    }

    #[test]
    fn v2_uses_runtime_feature_columns_when_present() {
        let mut model = ModeModel {
            feature_columns: vec!["dist_to_nearest_active_item".to_owned()],
            runtime_feature_columns: vec![
                "dist_to_nearest_active_item".to_owned(),
                "dist_to_dropoff".to_owned(),
            ],
            ..ModeModel::default()
        };
        model.normalization.mean = vec![0.0, 0.0];
        model.normalization.std = vec![1.0, 1.0];
        model.heads.insert(
            "pickup".to_owned(),
            HeadModel {
                kind: "ridge_logit".to_owned(),
                bias: 1.0,
                weights: vec![0.0, 0.0],
                clip: 8.0,
                temperature: 1.0,
            },
        );
        model.heads.insert(
            "dropoff".to_owned(),
            HeadModel {
                kind: "ridge_logit".to_owned(),
                bias: 1.0,
                weights: vec![0.0, 0.0],
                clip: 8.0,
                temperature: 1.0,
            },
        );
        model.heads.insert(
            "ordering".to_owned(),
            HeadModel {
                kind: "ridge".to_owned(),
                bias: 2.0,
                weights: vec![0.0, 0.0],
                clip: 12.0,
                temperature: 1.0,
            },
        );
        let score = score_pick_with_model(
            &model,
            CandidateFeatures {
                dist_to_nearest_active_item: 2.0,
                dist_to_dropoff: 2.0,
                ..CandidateFeatures::default()
            },
        );
        assert!(score.combined_expected_score > 0.0);
    }

    #[test]
    fn combined_score_uses_roundtrip_eta_proxy() {
        let mut model = ModeModel {
            runtime_feature_columns: vec![
                "dist_to_nearest_active_item".to_owned(),
                "dist_to_dropoff".to_owned(),
            ],
            ..ModeModel::default()
        };
        model.normalization.mean = vec![0.0, 0.0];
        model.normalization.std = vec![1.0, 1.0];
        model.heads.insert(
            "pickup".to_owned(),
            HeadModel {
                kind: "ridge_logit".to_owned(),
                bias: 8.0,
                weights: vec![0.0, 0.0],
                clip: 8.0,
                temperature: 1.0,
            },
        );
        model.heads.insert(
            "dropoff".to_owned(),
            HeadModel {
                kind: "ridge_logit".to_owned(),
                bias: 8.0,
                weights: vec![0.0, 0.0],
                clip: 8.0,
                temperature: 1.0,
            },
        );
        model.heads.insert(
            "ordering".to_owned(),
            HeadModel {
                kind: "ridge".to_owned(),
                bias: 10.0,
                weights: vec![0.0, 0.0],
                clip: 12.0,
                temperature: 1.0,
            },
        );
        let score = score_pick_with_model(
            &model,
            CandidateFeatures {
                dist_to_nearest_active_item: 0.0,
                dist_to_dropoff: 20.0,
                ..CandidateFeatures::default()
            },
        );
        assert!(score.combined_expected_score < 1.0);
    }

    #[test]
    fn negative_ordering_score_clamped_from_value_proxy() {
        let mut model = ModeModel {
            runtime_feature_columns: vec!["order_urgency".to_owned()],
            ..ModeModel::default()
        };
        model.normalization.mean = vec![0.0];
        model.normalization.std = vec![1.0];
        model.heads.insert(
            "pickup".to_owned(),
            HeadModel {
                kind: "ridge_logit".to_owned(),
                bias: 4.0,
                weights: vec![0.0],
                clip: 8.0,
                temperature: 1.0,
            },
        );
        model.heads.insert(
            "dropoff".to_owned(),
            HeadModel {
                kind: "ridge_logit".to_owned(),
                bias: 4.0,
                weights: vec![0.0],
                clip: 8.0,
                temperature: 1.0,
            },
        );
        model.heads.insert(
            "ordering".to_owned(),
            HeadModel {
                kind: "ridge".to_owned(),
                bias: -5.0,
                weights: vec![0.0],
                clip: 12.0,
                temperature: 1.0,
            },
        );
        let score = score_pick_with_model(
            &model,
            CandidateFeatures {
                order_urgency: 1.0,
                ..CandidateFeatures::default()
            },
        );
        assert!(score.ordering_score < 0.0);
        assert_eq!(score.combined_expected_score, 0.0);
    }
}
