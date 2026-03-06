use serde::{Deserialize, Serialize};

use crate::model::GameState;

use super::state::StepDelta;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SimMetrics {
    pub ticks: u64,
    pub score: i64,
    pub items_delivered: u64,
    pub orders_completed: u64,
    pub invalid_actions: u64,
    pub blocked_moves: u64,
    pub delivered_per_300_rounds: f64,
    pub completion_bonus_total: i64,
    pub avg_order_completion_latency: f64,
    #[serde(skip)]
    order_completion_ticks: Vec<u64>,
}

impl SimMetrics {
    pub fn observe(&mut self, state: &GameState, step: &StepDelta) {
        self.ticks = state.tick;
        self.score = state.score;
        self.items_delivered = self
            .items_delivered
            .saturating_add(step.items_delivered_delta);
        self.orders_completed = self
            .orders_completed
            .saturating_add(step.orders_completed_delta);
        self.invalid_actions = self.invalid_actions.saturating_add(step.invalid_actions);
        self.blocked_moves = self.blocked_moves.saturating_add(step.blocked_moves);
        self.completion_bonus_total = self.orders_completed as i64 * 5;

        self.delivered_per_300_rounds = if self.ticks == 0 {
            0.0
        } else {
            self.items_delivered as f64 * (300.0 / self.ticks as f64)
        };

        if step.orders_completed_delta > 0 {
            for _ in 0..step.orders_completed_delta {
                self.order_completion_ticks.push(state.tick.max(1));
            }
        }
        self.avg_order_completion_latency = if self.order_completion_ticks.is_empty() {
            0.0
        } else {
            self.order_completion_ticks.iter().sum::<u64>() as f64
                / self.order_completion_ticks.len() as f64
        };
    }
}
