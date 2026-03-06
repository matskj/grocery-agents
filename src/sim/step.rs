use crate::model::Action;

use super::{
    rules::apply_actions,
    state::{SimState, StepDelta},
};

pub fn step_once(sim: &mut SimState, actions: &[Action]) -> StepDelta {
    let pre_score = sim.game_state.score;
    let outcome = apply_actions(&mut sim.game_state, actions);

    // Promote/generate next orders when active order is done.
    if outcome.orders_completed_delta > 0 {
        sim.ensure_active_order_queue();
    }

    let delta_score =
        outcome.items_delivered_delta as i64 + (outcome.orders_completed_delta as i64 * 5);
    sim.game_state.score = sim.game_state.score.saturating_add(delta_score);
    sim.game_state.tick = sim.game_state.tick.saturating_add(1);

    let step = StepDelta {
        tick: sim.game_state.tick,
        delta_score: sim.game_state.score - pre_score,
        items_delivered_delta: outcome.items_delivered_delta,
        orders_completed_delta: outcome.orders_completed_delta,
        invalid_actions: outcome.invalid_actions,
        blocked_moves: outcome.blocked_moves,
    };
    sim.metrics.observe(&sim.game_state, &step);
    sim.last_delta = Some(step.clone());
    step
}

pub fn step_many(
    sim: &mut SimState,
    mut actions_provider: impl FnMut(&SimState) -> Vec<Action>,
    steps: u32,
) -> Vec<StepDelta> {
    let mut out = Vec::with_capacity(steps as usize);
    for _ in 0..steps {
        let actions = actions_provider(sim);
        out.push(step_once(sim, &actions));
    }
    out
}

#[cfg(test)]
mod tests {
    use crate::{
        model::{BotState, GameState, Grid, Item},
        sim::{orders::OrderGenerator, state::SimState},
    };

    use super::step_many;

    #[test]
    fn repeated_steps_are_deterministic_given_same_seed_and_actions() {
        let base = GameState {
            grid: Grid {
                width: 6,
                height: 6,
                drop_off_tiles: vec![[0, 0]],
                ..Grid::default()
            },
            bots: vec![BotState {
                id: "0".to_owned(),
                x: 1,
                y: 1,
                carrying: vec![],
                capacity: 3,
            }],
            items: vec![Item {
                id: "i1".to_owned(),
                kind: "milk".to_owned(),
                x: 2,
                y: 1,
            }],
            ..GameState::default()
        };

        let mut a = SimState::new(base.clone(), 99, OrderGenerator::default());
        let mut b = SimState::new(base, 99, OrderGenerator::default());
        let deltas_a = step_many(&mut a, |_| Vec::new(), 25);
        let deltas_b = step_many(&mut b, |_| Vec::new(), 25);

        assert_eq!(a.game_state.tick, b.game_state.tick);
        assert_eq!(a.game_state.score, b.game_state.score);
        assert_eq!(
            deltas_a.iter().map(|d| d.delta_score).collect::<Vec<_>>(),
            deltas_b.iter().map(|d| d.delta_score).collect::<Vec<_>>()
        );
    }
}
