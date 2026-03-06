use std::collections::HashSet;

use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::model::{GameState, OrderStatus};

use super::{metrics::SimMetrics, orders::OrderGenerator};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepDelta {
    pub tick: u64,
    pub delta_score: i64,
    pub items_delivered_delta: u64,
    pub orders_completed_delta: u64,
    pub invalid_actions: u64,
    pub blocked_moves: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimSnapshot {
    pub state: GameState,
    pub metrics: SimMetrics,
    pub last_delta: Option<StepDelta>,
}

pub struct SimState {
    pub game_state: GameState,
    pub rng: StdRng,
    pub generator: OrderGenerator,
    pub metrics: SimMetrics,
    pub last_delta: Option<StepDelta>,
}

impl SimState {
    pub fn new(initial: GameState, seed: u64, generator: OrderGenerator) -> Self {
        let mut game_state = initial;
        if game_state.tick == 0 && game_state.active_order_index < 0 {
            game_state.active_order_index = 0;
        }
        Self {
            game_state,
            rng: StdRng::seed_from_u64(seed),
            generator,
            metrics: SimMetrics::default(),
            last_delta: None,
        }
    }

    pub fn snapshot(&self) -> SimSnapshot {
        SimSnapshot {
            state: self.game_state.clone(),
            metrics: self.metrics.clone(),
            last_delta: self.last_delta.clone(),
        }
    }

    pub fn ensure_active_order_queue(&mut self) {
        let active_exists = self
            .game_state
            .orders
            .iter()
            .any(|order| matches!(order.status, OrderStatus::InProgress));
        if active_exists {
            return;
        }

        let mut pending_groups = self.pending_groups();
        pending_groups.sort();
        if let Some(next_group) = pending_groups.first() {
            for order in &mut self.game_state.orders {
                if order_group_index(&order.id) == Some(*next_group)
                    && matches!(order.status, OrderStatus::Pending)
                {
                    order.status = OrderStatus::InProgress;
                }
            }
            self.game_state.active_order_index = i64::from(*next_group);
            return;
        }

        let next_index = self.game_state.active_order_index.max(0) as u64 + 1;
        let new_orders = self.generator.generate_order_entries(
            next_index,
            &self.game_state.items,
            &mut self.rng,
        );
        if new_orders.is_empty() {
            return;
        }
        let mut first = true;
        for mut order in new_orders {
            order.status = if first {
                first = false;
                OrderStatus::InProgress
            } else {
                OrderStatus::Pending
            };
            self.game_state.orders.push(order);
        }
        self.game_state.active_order_index = next_index as i64;
    }

    fn pending_groups(&self) -> Vec<u32> {
        self.game_state
            .orders
            .iter()
            .filter(|order| matches!(order.status, OrderStatus::Pending))
            .filter_map(|order| order_group_index(&order.id))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
    }
}

pub fn order_group_index(order_id: &str) -> Option<u32> {
    order_id
        .split(':')
        .next()
        .and_then(|prefix| prefix.rsplit('_').next())
        .and_then(|tail| tail.parse::<u32>().ok())
}
