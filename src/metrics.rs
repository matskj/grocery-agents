use crate::model::{Action, GameState, OrderStatus};

#[derive(Debug, Clone, Default)]
pub struct OrderMetrics {
    pub order_index: i64,
    pub rounds_to_complete: u64,
    pub items_delivered: u64,
    pub items_pre_staged: u64,
}

#[derive(Debug, Clone, Default)]
pub struct GameMetricsSummary {
    pub total_score: i64,
    pub orders_completed: u64,
    pub items_delivered: u64,
    pub avg_path_length_per_item: f64,
    pub avg_planning_time_ms: f64,
    pub p95_planning_time_ms: u64,
    pub corrected_invalid_actions: u64,
    pub wait_actions: u64,
    pub collisions_prevented: u64,
    pub bots_idle: u64,
    pub order_metrics: Vec<OrderMetrics>,
}

#[derive(Debug, Clone, Default)]
struct ActiveOrderTracker {
    order_index: i64,
    start_tick: u64,
    items_delivered: u64,
    items_pre_staged: u64,
}

#[derive(Debug, Clone, Default)]
pub struct MetricsTracker {
    planning_time_ms: Vec<u64>,
    total_score: i64,
    total_moves: u64,
    wait_actions: u64,
    corrected_invalid_actions: u64,
    collisions_prevented: u64,
    bots_idle: u64,
    items_delivered: u64,
    orders_completed: u64,
    active_order: Option<ActiveOrderTracker>,
    order_metrics: Vec<OrderMetrics>,
}

impl MetricsTracker {
    pub fn on_tick(
        &mut self,
        state: &GameState,
        actions: &[Action],
        team_summary: &serde_json::Value,
        tick_outcome: &serde_json::Value,
        planning_time_ms: u64,
    ) {
        self.total_score = state.score;
        self.planning_time_ms.push(planning_time_ms);

        let mut wait_actions = 0u64;
        let mut move_actions = 0u64;
        for action in actions {
            match action {
                Action::Wait { .. } => wait_actions = wait_actions.saturating_add(1),
                Action::Move { .. } => move_actions = move_actions.saturating_add(1),
                Action::PickUp { .. } | Action::DropOff { .. } => {}
            }
        }
        self.wait_actions = self.wait_actions.saturating_add(wait_actions);
        self.bots_idle = self.bots_idle.saturating_add(wait_actions);
        self.total_moves = self.total_moves.saturating_add(move_actions);

        let invalids = tick_outcome
            .get("invalid_action_count")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        self.corrected_invalid_actions = self.corrected_invalid_actions.saturating_add(invalids);

        let local_conflict_sum = team_summary
            .get("local_conflict_count_by_bot")
            .and_then(serde_json::Value::as_object)
            .map(|m| {
                m.values()
                    .filter_map(serde_json::Value::as_u64)
                    .fold(0u64, |acc, v| acc.saturating_add(v))
            })
            .unwrap_or(0);
        self.collisions_prevented = self.collisions_prevented.saturating_add(local_conflict_sum);

        let items_delta = tick_outcome
            .get("items_delivered_delta")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(0)
            .max(0) as u64;
        let order_delta = tick_outcome
            .get("order_completed_delta")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(0)
            .max(0) as u64;

        self.items_delivered = self.items_delivered.saturating_add(items_delta);

        if self.active_order.is_none() {
            self.active_order = Some(ActiveOrderTracker {
                order_index: state.active_order_index,
                start_tick: state.tick,
                items_delivered: 0,
                items_pre_staged: preview_items_staged_at_dropoff(state),
            });
        }

        if let Some(active) = self.active_order.as_mut() {
            active.items_delivered = active.items_delivered.saturating_add(items_delta);
        }

        if order_delta > 0 {
            for _ in 0..order_delta {
                if let Some(active) = self.active_order.take() {
                    let rounds = state
                        .tick
                        .saturating_sub(active.start_tick)
                        .saturating_add(1)
                        .max(1);
                    self.order_metrics.push(OrderMetrics {
                        order_index: active.order_index,
                        rounds_to_complete: rounds,
                        items_delivered: active.items_delivered,
                        items_pre_staged: active.items_pre_staged,
                    });
                    self.orders_completed = self.orders_completed.saturating_add(1);
                }
                self.active_order = Some(ActiveOrderTracker {
                    order_index: state.active_order_index,
                    start_tick: state.tick,
                    items_delivered: 0,
                    items_pre_staged: preview_items_staged_at_dropoff(state),
                });
            }
        }
    }

    pub fn summary(&self) -> GameMetricsSummary {
        let avg_path = if self.items_delivered > 0 {
            self.total_moves as f64 / self.items_delivered as f64
        } else {
            0.0
        };
        let avg_plan = if self.planning_time_ms.is_empty() {
            0.0
        } else {
            self.planning_time_ms.iter().sum::<u64>() as f64 / self.planning_time_ms.len() as f64
        };
        GameMetricsSummary {
            total_score: self.total_score,
            orders_completed: self.orders_completed,
            items_delivered: self.items_delivered,
            avg_path_length_per_item: avg_path,
            avg_planning_time_ms: avg_plan,
            p95_planning_time_ms: percentile_u64(&self.planning_time_ms, 0.95),
            corrected_invalid_actions: self.corrected_invalid_actions,
            wait_actions: self.wait_actions,
            collisions_prevented: self.collisions_prevented,
            bots_idle: self.bots_idle,
            order_metrics: self.order_metrics.clone(),
        }
    }
}

fn preview_items_staged_at_dropoff(state: &GameState) -> u64 {
    let mut preview_kinds = std::collections::HashSet::<&str>::new();
    for order in &state.orders {
        if matches!(order.status, OrderStatus::Pending) {
            preview_kinds.insert(order.item_id.as_str());
        }
    }
    if preview_kinds.is_empty() {
        return 0;
    }
    state
        .bots
        .iter()
        .filter(|bot| {
            state
                .grid
                .drop_off_tiles
                .iter()
                .any(|tile| tile[0] == bot.x && tile[1] == bot.y)
        })
        .map(|bot| {
            bot.carrying
                .iter()
                .filter(|kind| preview_kinds.contains(kind.as_str()))
                .count() as u64
        })
        .sum()
}

fn percentile_u64(values: &[u64], q: f64) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let idx = ((sorted.len() - 1) as f64 * q.clamp(0.0, 1.0)).round() as usize;
    sorted[idx]
}
