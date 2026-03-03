use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::time::{Duration, Instant};

use crate::{
    dist::DistanceMap,
    model::{Action, GameState},
    team_context::{BlockedMove, BotRole, MovementReservation},
    world::MapCache,
};

#[derive(Debug, Clone, Default)]
pub struct MotionPlanner {
    horizon: u8,
}

#[derive(Debug, Clone)]
pub struct PlannedAction {
    pub action: Action,
    pub wait_reason: &'static str,
    pub fallback_stage: &'static str,
    pub ordering_stage: &'static str,
    pub path_preview: Vec<u16>,
}

#[derive(Debug, Clone, Default)]
pub struct MotionPlanDiagnostics {
    pub local_conflict_count_by_bot: HashMap<String, u16>,
    pub cbs_timeout: bool,
    pub cbs_expanded_nodes: u16,
    pub budget_cutoff_waits: u16,
    pub reserved_cells_by_t: HashMap<u8, Vec<u16>>,
}

#[derive(Debug, Clone, Default)]
pub struct MotionPlanResult {
    pub actions: HashMap<String, PlannedAction>,
    pub diagnostics: MotionPlanDiagnostics,
}

#[derive(Debug, Clone, Default)]
struct ReservationTable {
    vertex_res: HashSet<(u8, u16)>,
    edge_res: HashSet<(u8, u16, u16)>,
    dropoff_res: HashMap<(u8, u16), u8>,
    dropoff_capacity: u8,
}

impl ReservationTable {
    fn new(dropoff_capacity: u8) -> Self {
        Self {
            vertex_res: HashSet::new(),
            edge_res: HashSet::new(),
            dropoff_res: HashMap::new(),
            dropoff_capacity: dropoff_capacity.max(1),
        }
    }

    fn reserve_wait(&mut self, t: u8, cell: u16) {
        self.vertex_res.insert((t, cell));
    }

    fn reserve_dropoff_slot(&mut self, t: u8, drop_cell: u16) {
        let entry = self.dropoff_res.entry((t, drop_cell)).or_insert(0);
        *entry = entry.saturating_add(1);
    }

    fn reserve_path(
        &mut self,
        path: &[u16],
        reserve_horizon: u8,
        max_horizon: u8,
        dropoff_cells: &[u16],
    ) {
        let mut prev = path.first().copied().unwrap_or(0);
        for t in 1..=reserve_horizon.min(max_horizon) {
            let idx = usize::from(t).min(path.len().saturating_sub(1));
            let next = path.get(idx).copied().unwrap_or(prev);
            self.vertex_res.insert((t, next));
            self.edge_res.insert((t, prev, next));
            self.edge_res.insert((t, next, prev));
            if dropoff_cells.contains(&next) {
                self.reserve_dropoff_slot(t, next);
            }
            prev = next;
        }
    }
}

#[derive(Debug, Clone)]
struct Node {
    f: u16,
    g: u16,
    cell: u16,
    t: u8,
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        other.f.cmp(&self.f).then_with(|| other.g.cmp(&self.g))
    }
}
impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.f == other.f && self.g == other.g && self.cell == other.cell && self.t == other.t
    }
}
impl Eq for Node {}

impl MotionPlanner {
    pub fn new(horizon: u8) -> Self {
        Self {
            horizon: horizon.clamp(12, 20),
        }
    }

    pub fn plan(
        &self,
        state: &GameState,
        map: &MapCache,
        dist: &DistanceMap,
        goals: &HashMap<String, u16>,
        reservation: &MovementReservation,
        explicit_order: &[String],
        soft_deadline: Option<Instant>,
        dropoff_capacity: u8,
    ) -> MotionPlanResult {
        let mut out = HashMap::with_capacity(state.bots.len());
        let mut res_table = ReservationTable::new(dropoff_capacity);
        let mut diagnostics = MotionPlanDiagnostics::default();
        let cbs_start = Instant::now();
        let cbs_budget = Duration::from_millis(3);
        let cbs_node_cap: u16 = 80;
        let _ = explicit_order;
        let mut bots = state.bots.iter().collect::<Vec<_>>();
        bots.sort_by(|a, b| a.id.cmp(&b.id));

        for (ix, bot) in bots.iter().enumerate() {
            if soft_deadline
                .map(|deadline| Instant::now() + Duration::from_millis(2) >= deadline)
                .unwrap_or(false)
            {
                for pending in bots.iter().skip(ix) {
                    if out.contains_key(&pending.id) {
                        continue;
                    }
                    if let Some(start) = map.idx(pending.x, pending.y) {
                        res_table.reserve_wait(1, start);
                    }
                    out.insert(
                        pending.id.clone(),
                        PlannedAction {
                            action: Action::wait(pending.id.clone()),
                            wait_reason: "timeout_fallback",
                            fallback_stage: "budget_guard_wait",
                            ordering_stage: "pmat_budget_guard",
                            path_preview: Vec::new(),
                        },
                    );
                    diagnostics.budget_cutoff_waits =
                        diagnostics.budget_cutoff_waits.saturating_add(1);
                }
                break;
            }
            let start = match map.idx(bot.x, bot.y) {
                Some(v) => v,
                None => {
                    out.insert(
                        bot.id.clone(),
                        PlannedAction {
                            action: Action::wait(bot.id.clone()),
                            wait_reason: "no_path_with_constraints",
                            fallback_stage: "invalid_start",
                            ordering_stage: "pmat_ordered",
                            path_preview: Vec::new(),
                        },
                    );
                    continue;
                }
            };
            let goal = goals.get(&bot.id).copied().unwrap_or(start);
            let blocked = reservation
                .prohibited_moves
                .get(&bot.id)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let forbidden = reservation
                .forbidden_cells
                .get(&bot.id)
                .cloned()
                .unwrap_or_default();
            let role = reservation
                .role_by_bot
                .get(&bot.id)
                .copied()
                .unwrap_or(BotRole::Idle);
            let outcome = self.pick_step_with_fallback(
                start,
                goal,
                map,
                dist,
                &res_table.vertex_res,
                &res_table.edge_res,
                &res_table.dropoff_res,
                res_table.dropoff_capacity,
                blocked,
                &forbidden,
                role,
            );
            let action = if outcome.step == start {
                Action::wait(bot.id.clone())
            } else {
                let (x0, y0) = map.xy(start);
                let (x1, y1) = map.xy(outcome.step);
                Action::Move {
                    bot_id: bot.id.clone(),
                    dx: x1 - x0,
                    dy: y1 - y0,
                }
            };

            if outcome.path.len() > 1 && outcome.step != start {
                let reserve_horizon = self
                    .horizon
                    .min(outcome.path.len().saturating_sub(1) as u8)
                    .max(1);
                res_table.reserve_path(
                    &outcome.path,
                    reserve_horizon,
                    self.horizon,
                    &map.dropoff_cells,
                );
            } else {
                res_table.reserve_wait(1, start);
            }
            out.insert(
                bot.id.clone(),
                PlannedAction {
                    action,
                    wait_reason: outcome.wait_reason,
                    fallback_stage: outcome.fallback_stage,
                    ordering_stage: "pmat_ordered",
                    path_preview: outcome.path.iter().skip(1).take(3).copied().collect(),
                },
            );
        }

        if diagnostics.budget_cutoff_waits == 0 {
            let cbs_result = self.resolve_dropoff_zone_conflicts(
                state,
                map,
                dist,
                goals,
                reservation,
                &mut out,
                cbs_start,
                cbs_budget,
                cbs_node_cap,
            );
            if cbs_result.timeout {
                for bot in &state.bots {
                    let Some(start) = map.idx(bot.x, bot.y) else {
                        continue;
                    };
                    let goal = goals.get(&bot.id).copied().unwrap_or(start);
                    if !(reservation.dropoff_control_zone.contains(&start)
                        || reservation.dropoff_control_zone.contains(&goal))
                    {
                        continue;
                    }
                    if let Some(entry) = out.get_mut(&bot.id) {
                        if entry.fallback_stage == "primary" {
                            entry.fallback_stage = "cbs_fallback_prioritized";
                        }
                        entry.ordering_stage = "pmat_cbs_fallback";
                    }
                }
            }
            diagnostics.local_conflict_count_by_bot = cbs_result.local_conflict_count_by_bot;
            diagnostics.cbs_timeout = cbs_result.timeout;
            diagnostics.cbs_expanded_nodes = cbs_result.expanded_nodes;
        }

        let mut reserved_cells_by_t = HashMap::<u8, Vec<u16>>::new();
        for &(t, cell) in &res_table.vertex_res {
            reserved_cells_by_t.entry(t).or_default().push(cell);
        }
        for cells in reserved_cells_by_t.values_mut() {
            cells.sort_unstable();
            cells.dedup();
        }
        diagnostics.reserved_cells_by_t = reserved_cells_by_t;

        MotionPlanResult {
            actions: out,
            diagnostics,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn pick_step_with_fallback(
        &self,
        start: u16,
        goal: u16,
        map: &MapCache,
        dist: &DistanceMap,
        v_res: &HashSet<(u8, u16)>,
        e_res: &HashSet<(u8, u16, u16)>,
        dropoff_res: &HashMap<(u8, u16), u8>,
        dropoff_capacity: u8,
        blocked_moves: &[BlockedMove],
        forbidden_cells: &HashSet<u16>,
        role: BotRole,
    ) -> StepOutcome {
        let primary = self.next_path(
            start,
            goal,
            map,
            dist,
            v_res,
            e_res,
            dropoff_res,
            dropoff_capacity,
            blocked_moves,
            forbidden_cells,
            self.horizon,
        );
        let step = primary.get(1).copied().unwrap_or(start);
        if step != start {
            return StepOutcome::move_step(step, primary, "primary");
        }
        if matches!(role, BotRole::Collector | BotRole::Yield) {
            if let Some(egress) =
                best_dropoff_egress_step(start, map, dist, forbidden_cells, blocked_moves)
            {
                return StepOutcome::move_step(egress, vec![start, egress], "dropoff_egress");
            }
        }

        let base_reason =
            diagnose_wait_reason(start, map, v_res, e_res, blocked_moves, forbidden_cells);
        let mut reduced_v = HashSet::new();
        let mut reduced_e = HashSet::new();
        for &(t, cell) in v_res {
            if t <= 1 {
                reduced_v.insert((t, cell));
            }
        }
        for &(t, a, b) in e_res {
            if t <= 1 {
                reduced_e.insert((t, a, b));
            }
        }

        let reduced = self.next_path(
            start,
            goal,
            map,
            dist,
            &reduced_v,
            &reduced_e,
            dropoff_res,
            dropoff_capacity,
            blocked_moves,
            forbidden_cells,
            10,
        );
        let step_reduced = reduced.get(1).copied().unwrap_or(start);
        if step_reduced != start {
            return StepOutcome::move_step(step_reduced, reduced, "reduce_horizon");
        }

        if !matches!(role, BotRole::LeadCourier) {
            let relaxed = self.next_path(
                start,
                goal,
                map,
                dist,
                v_res,
                e_res,
                dropoff_res,
                dropoff_capacity,
                blocked_moves,
                &HashSet::new(),
                self.horizon,
            );
            let step_relaxed = relaxed.get(1).copied().unwrap_or(start);
            if step_relaxed != start {
                return StepOutcome::move_step(step_relaxed, relaxed, "relax_forbidden");
            }
        }

        if let Some(side) = best_local_sidestep(start, map, dist, forbidden_cells, blocked_moves) {
            return StepOutcome::move_step(side, vec![start, side], "local_sidestep");
        }

        StepOutcome {
            step: start,
            path: vec![start],
            wait_reason: base_reason,
            fallback_stage: "none",
        }
    }

    fn next_path(
        &self,
        start: u16,
        goal: u16,
        map: &MapCache,
        dist: &DistanceMap,
        v_res: &HashSet<(u8, u16)>,
        e_res: &HashSet<(u8, u16, u16)>,
        dropoff_res: &HashMap<(u8, u16), u8>,
        dropoff_capacity: u8,
        blocked_moves: &[BlockedMove],
        forbidden_cells: &HashSet<u16>,
        max_horizon: u8,
    ) -> Vec<u16> {
        if start == goal {
            return vec![start];
        }
        let mut open = BinaryHeap::new();
        let mut best_g: HashMap<(u16, u8), u16> = HashMap::new();
        let mut parent: HashMap<(u16, u8), (u16, u8)> = HashMap::new();

        open.push(Node {
            f: dist.dist(start, goal),
            g: 0,
            cell: start,
            t: 0,
        });
        best_g.insert((start, 0), 0);

        let mut best_terminal = (start, 0u8, u16::MAX);

        while let Some(Node { g, cell, t, .. }) = open.pop() {
            let h = dist.dist(cell, goal);
            if h < best_terminal.2 {
                best_terminal = (cell, t, h);
            }
            if cell == goal {
                best_terminal = (cell, t, 0);
                break;
            }
            if t >= max_horizon.min(self.horizon) {
                continue;
            }

            let nt = t + 1;
            let mut succ = map.neighbors[cell as usize].clone();
            succ.push(cell); // wait move
            for next in succ {
                if forbidden_cells.contains(&next) && next != goal {
                    continue;
                }
                if v_res.contains(&(nt, next)) || e_res.contains(&(nt, cell, next)) {
                    continue;
                }
                if map.dropoff_cells.contains(&next)
                    && dropoff_res.get(&(nt, next)).copied().unwrap_or(0) >= dropoff_capacity
                {
                    continue;
                }
                if t == 0
                    && is_hard_prohibited_step(
                        start,
                        next,
                        map,
                        blocked_moves,
                        forbidden_cells,
                        v_res,
                        e_res,
                    )
                {
                    continue;
                }
                let mut penalty = 0u16;
                if t == 0 {
                    penalty = 18u16.saturating_mul(u16::from(blocked_move_count(
                        start,
                        next,
                        map,
                        blocked_moves,
                    )));
                }
                let ng = g.saturating_add(1).saturating_add(penalty);
                let key = (next, nt);
                if best_g.get(&key).map(|v| ng >= *v).unwrap_or(false) {
                    continue;
                }
                best_g.insert(key, ng);
                parent.insert(key, (cell, t));
                let f = (ng as u16).saturating_add(dist.dist(next, goal));
                open.push(Node {
                    f,
                    g: ng,
                    cell: next,
                    t: nt,
                });
            }
        }

        let mut path_rev = vec![best_terminal.0];
        let mut cur = (best_terminal.0, best_terminal.1);
        while let Some(prev) = parent.get(&cur).copied() {
            path_rev.push(prev.0);
            cur = prev;
            if prev.1 == 0 {
                break;
            }
        }
        path_rev.reverse();
        if path_rev.first().copied() != Some(start) {
            path_rev.insert(0, start);
        }
        path_rev
    }

    #[allow(clippy::too_many_arguments)]
    fn resolve_dropoff_zone_conflicts(
        &self,
        state: &GameState,
        map: &MapCache,
        dist: &DistanceMap,
        goals: &HashMap<String, u16>,
        reservation: &MovementReservation,
        planned: &mut HashMap<String, PlannedAction>,
        start_time: Instant,
        budget: Duration,
        node_cap: u16,
    ) -> CbsResolution {
        let mut result = CbsResolution::default();
        if reservation.dropoff_control_zone.is_empty() || state.bots.len() < 2 {
            return result;
        }

        let mut start_cells = HashMap::new();
        for bot in &state.bots {
            if let Some(start) = map.idx(bot.x, bot.y) {
                start_cells.insert(bot.id.clone(), start);
            }
        }
        let zone_bots = state
            .bots
            .iter()
            .filter(|bot| {
                let Some(start) = start_cells.get(&bot.id).copied() else {
                    return false;
                };
                let goal = goals.get(&bot.id).copied().unwrap_or(start);
                reservation.dropoff_control_zone.contains(&start)
                    || reservation.dropoff_control_zone.contains(&goal)
            })
            .map(|bot| bot.id.clone())
            .collect::<HashSet<_>>();
        if zone_bots.len() < 2 {
            return result;
        }

        loop {
            if start_time.elapsed() > budget {
                result.timeout = true;
                break;
            }
            if result.expanded_nodes >= node_cap {
                result.timeout = true;
                break;
            }
            let conflicts = detect_zone_conflicts(&zone_bots, &start_cells, planned, state, map);
            if conflicts.is_empty() {
                break;
            }
            for conflict in &conflicts {
                *result
                    .local_conflict_count_by_bot
                    .entry(conflict.a.clone())
                    .or_insert(0) += 1;
                *result
                    .local_conflict_count_by_bot
                    .entry(conflict.b.clone())
                    .or_insert(0) += 1;
            }

            let conflict = conflicts[0].clone();
            let a_prio = reservation
                .priorities
                .get(&conflict.a)
                .copied()
                .unwrap_or(u8::MAX);
            let b_prio = reservation
                .priorities
                .get(&conflict.b)
                .copied()
                .unwrap_or(u8::MAX);
            let try_order = if a_prio > b_prio {
                vec![conflict.a.as_str(), conflict.b.as_str()]
            } else {
                vec![conflict.b.as_str(), conflict.a.as_str()]
            };

            let mut resolved = false;
            for bot_id in try_order {
                if self.replan_conflicted_bot(
                    bot_id,
                    &conflict,
                    state,
                    map,
                    dist,
                    goals,
                    reservation,
                    &start_cells,
                    planned,
                ) {
                    result.expanded_nodes = result.expanded_nodes.saturating_add(1);
                    resolved = true;
                    break;
                }
            }
            if !resolved {
                break;
            }
        }

        result
    }

    #[allow(clippy::too_many_arguments)]
    fn replan_conflicted_bot(
        &self,
        bot_id: &str,
        conflict: &ZoneConflict,
        state: &GameState,
        map: &MapCache,
        dist: &DistanceMap,
        goals: &HashMap<String, u16>,
        reservation: &MovementReservation,
        start_cells: &HashMap<String, u16>,
        planned: &mut HashMap<String, PlannedAction>,
    ) -> bool {
        let Some(bot) = state.bots.iter().find(|b| b.id == bot_id) else {
            return false;
        };
        let Some(start) = start_cells.get(bot_id).copied() else {
            return false;
        };
        let goal = goals.get(bot_id).copied().unwrap_or(start);
        let role = reservation
            .role_by_bot
            .get(bot_id)
            .copied()
            .unwrap_or(BotRole::Idle);
        let blocked = reservation
            .prohibited_moves
            .get(bot_id)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let forbidden = reservation
            .forbidden_cells
            .get(bot_id)
            .cloned()
            .unwrap_or_default();

        let mut v_res: HashSet<(u8, u16)> = HashSet::new();
        let mut e_res: HashSet<(u8, u16, u16)> = HashSet::new();
        for other in &state.bots {
            if other.id == bot_id {
                continue;
            }
            let Some(other_start) = start_cells.get(&other.id).copied() else {
                continue;
            };
            let other_next = next_cell_for_bot(&other.id, other_start, planned, map);
            v_res.insert((1, other_next));
            e_res.insert((1, other_start, other_next));
            e_res.insert((1, other_next, other_start));
        }
        if conflict.kind == "vertex" {
            v_res.insert((1, conflict.cell));
        } else {
            let target = if conflict.a == bot.id {
                conflict.a_to
            } else {
                conflict.b_to
            };
            e_res.insert((1, start, target));
            e_res.insert((1, target, start));
        }

        let outcome = self.pick_step_with_fallback(
            start,
            goal,
            map,
            dist,
            &v_res,
            &e_res,
            &HashMap::new(),
            1,
            blocked,
            &forbidden,
            role,
        );
        let next = outcome.step;
        if next == start {
            return false;
        }
        if would_conflict(bot_id, next, start, start_cells, planned, map) {
            return false;
        }
        let (x0, y0) = map.xy(start);
        let (x1, y1) = map.xy(next);
        planned.insert(
            bot.id.clone(),
            PlannedAction {
                action: Action::Move {
                    bot_id: bot.id.clone(),
                    dx: x1 - x0,
                    dy: y1 - y0,
                },
                wait_reason: outcome.wait_reason,
                fallback_stage: "cbs_primary",
                ordering_stage: "pmat_cbs_replan",
                path_preview: outcome.path.iter().skip(1).take(3).copied().collect(),
            },
        );
        true
    }
}

#[derive(Debug, Clone, Default)]
struct CbsResolution {
    local_conflict_count_by_bot: HashMap<String, u16>,
    timeout: bool,
    expanded_nodes: u16,
}

#[derive(Debug, Clone)]
struct ZoneConflict {
    kind: &'static str,
    a: String,
    b: String,
    cell: u16,
    a_to: u16,
    b_to: u16,
}

fn detect_zone_conflicts(
    zone_bots: &HashSet<String>,
    start_cells: &HashMap<String, u16>,
    planned: &HashMap<String, PlannedAction>,
    state: &GameState,
    map: &MapCache,
) -> Vec<ZoneConflict> {
    let mut conflicts = Vec::new();
    let mut next_by_bot = HashMap::new();
    for bot in &state.bots {
        if !zone_bots.contains(&bot.id) {
            continue;
        }
        let Some(start) = start_cells.get(&bot.id).copied() else {
            continue;
        };
        let next = next_cell_for_bot(&bot.id, start, planned, map);
        next_by_bot.insert(bot.id.clone(), (start, next));
    }
    let bot_ids = next_by_bot.keys().cloned().collect::<Vec<_>>();
    for i in 0..bot_ids.len() {
        for j in (i + 1)..bot_ids.len() {
            let a = &bot_ids[i];
            let b = &bot_ids[j];
            let Some((a_start, a_next)) = next_by_bot.get(a).copied() else {
                continue;
            };
            let Some((b_start, b_next)) = next_by_bot.get(b).copied() else {
                continue;
            };
            if a_next == b_next {
                conflicts.push(ZoneConflict {
                    kind: "vertex",
                    a: a.clone(),
                    b: b.clone(),
                    cell: a_next,
                    a_to: a_next,
                    b_to: b_next,
                });
                continue;
            }
            if a_next == b_start && b_next == a_start {
                conflicts.push(ZoneConflict {
                    kind: "edge",
                    a: a.clone(),
                    b: b.clone(),
                    cell: a_next,
                    a_to: a_next,
                    b_to: b_next,
                });
            }
        }
    }
    conflicts
}

fn next_cell_for_bot(
    bot_id: &str,
    start: u16,
    planned: &HashMap<String, PlannedAction>,
    map: &MapCache,
) -> u16 {
    match planned.get(bot_id).map(|p| &p.action) {
        Some(Action::Move { dx, dy, .. }) => {
            let (x0, y0) = map.xy(start);
            map.idx(x0 + *dx, y0 + *dy).unwrap_or(start)
        }
        _ => start,
    }
}

fn would_conflict(
    bot_id: &str,
    next: u16,
    start: u16,
    start_cells: &HashMap<String, u16>,
    planned: &HashMap<String, PlannedAction>,
    map: &MapCache,
) -> bool {
    for (other_id, other_start) in start_cells {
        if other_id == bot_id {
            continue;
        }
        let other_next = next_cell_for_bot(other_id, *other_start, planned, map);
        if other_next == next {
            return true;
        }
        if other_next == start && *other_start == next {
            return true;
        }
    }
    false
}

fn blocked_move_count(start: u16, next: u16, map: &MapCache, blocked: &[BlockedMove]) -> u8 {
    let (x0, y0) = map.xy(start);
    let (x1, y1) = map.xy(next);
    let dx = x1 - x0;
    let dy = y1 - y0;
    blocked
        .iter()
        .find(|entry| entry.from == start && entry.dx == dx && entry.dy == dy)
        .map(|entry| entry.count.min(4))
        .unwrap_or(0)
}

fn has_alternate_exit(start: u16, chosen: u16, map: &MapCache, blocked: &[BlockedMove]) -> bool {
    map.neighbors[start as usize]
        .iter()
        .copied()
        .any(|candidate| {
            candidate != chosen && blocked_move_count(start, candidate, map, blocked) < 2
        })
}

fn is_hard_prohibited_step(
    start: u16,
    next: u16,
    map: &MapCache,
    blocked: &[BlockedMove],
    forbidden_cells: &HashSet<u16>,
    v_res: &HashSet<(u8, u16)>,
    e_res: &HashSet<(u8, u16, u16)>,
) -> bool {
    let count = blocked_move_count(start, next, map, blocked);
    if count < 2 {
        return false;
    }
    let mut has_legal_alternative = false;
    for &candidate in &map.neighbors[start as usize] {
        if candidate == next || forbidden_cells.contains(&candidate) {
            continue;
        }
        if v_res.contains(&(1, candidate)) || e_res.contains(&(1, start, candidate)) {
            continue;
        }
        has_legal_alternative = true;
        if blocked_move_count(start, candidate, map, blocked) < 2 {
            return true;
        }
    }
    if !has_legal_alternative {
        return false;
    }
    false
}

fn is_prohibited_step(start: u16, next: u16, map: &MapCache, blocked: &[BlockedMove]) -> bool {
    let count = blocked_move_count(start, next, map, blocked);
    count >= 2 && has_alternate_exit(start, next, map, blocked)
}

#[derive(Debug)]
struct StepOutcome {
    step: u16,
    path: Vec<u16>,
    wait_reason: &'static str,
    fallback_stage: &'static str,
}

impl StepOutcome {
    fn move_step(step: u16, path: Vec<u16>, fallback_stage: &'static str) -> Self {
        Self {
            step,
            path,
            wait_reason: "intent_wait",
            fallback_stage,
        }
    }
}

fn diagnose_wait_reason(
    start: u16,
    map: &MapCache,
    v_res: &HashSet<(u8, u16)>,
    e_res: &HashSet<(u8, u16, u16)>,
    blocked_moves: &[BlockedMove],
    forbidden_cells: &HashSet<u16>,
) -> &'static str {
    let mut saw_forbidden = false;
    let mut saw_prohibited = false;
    let mut saw_vertex = false;
    let mut saw_edge = false;
    for &next in &map.neighbors[start as usize] {
        if forbidden_cells.contains(&next) {
            saw_forbidden = true;
            continue;
        }
        if is_prohibited_step(start, next, map, blocked_moves) {
            saw_prohibited = true;
            continue;
        }
        if v_res.contains(&(1, next)) {
            saw_vertex = true;
            continue;
        }
        if e_res.contains(&(1, start, next)) {
            saw_edge = true;
            continue;
        }
        return "no_path_with_constraints";
    }
    if saw_forbidden {
        return "forbidden_queue_zone";
    }
    if saw_prohibited {
        return "prohibited_repeat_move";
    }
    if saw_vertex {
        return "blocked_by_vertex_reservation";
    }
    if saw_edge {
        return "blocked_by_edge_reservation";
    }
    "no_path_with_constraints"
}

fn best_local_sidestep(
    start: u16,
    map: &MapCache,
    dist: &DistanceMap,
    forbidden_cells: &HashSet<u16>,
    blocked_moves: &[BlockedMove],
) -> Option<u16> {
    map.neighbors[start as usize]
        .iter()
        .copied()
        .filter(|next| !forbidden_cells.contains(next))
        .filter(|next| !is_prohibited_step(start, *next, map, blocked_moves))
        .max_by_key(|next| {
            let d = map
                .dropoff_cells
                .iter()
                .map(|&drop| dist.dist(*next, drop))
                .min()
                .unwrap_or(u16::MAX);
            let free_degree = map.neighbors[*next as usize].len() as i32;
            i32::from(d.min(128)) + free_degree
        })
}

fn best_dropoff_egress_step(
    start: u16,
    map: &MapCache,
    dist: &DistanceMap,
    forbidden_cells: &HashSet<u16>,
    blocked_moves: &[BlockedMove],
) -> Option<u16> {
    let start_drop_dist = map
        .dropoff_cells
        .iter()
        .map(|&drop| dist.dist(start, drop))
        .min()
        .unwrap_or(u16::MAX);
    if start_drop_dist > 2 {
        return None;
    }
    map.neighbors[start as usize]
        .iter()
        .copied()
        .filter(|next| !forbidden_cells.contains(next))
        .filter(|next| !is_prohibited_step(start, *next, map, blocked_moves))
        .filter(|next| {
            let next_drop_dist = map
                .dropoff_cells
                .iter()
                .map(|&drop| dist.dist(*next, drop))
                .min()
                .unwrap_or(u16::MAX);
            next_drop_dist > start_drop_dist
        })
        .max_by_key(|next| {
            let d = map
                .dropoff_cells
                .iter()
                .map(|&drop| dist.dist(*next, drop))
                .min()
                .unwrap_or(u16::MAX);
            let free_degree = map.neighbors[*next as usize].len() as i32;
            i32::from(d.min(128)) + free_degree
        })
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use crate::{
        dist::DistanceMap,
        model::{Action, BotState, GameState, Grid},
        team_context::MovementReservation,
        world::World,
    };

    use super::{is_hard_prohibited_step, MotionPlanner};

    #[test]
    fn repeat_move_hard_forbid_does_not_block_only_exit() {
        let state = GameState {
            grid: Grid {
                width: 2,
                height: 1,
                ..Grid::default()
            },
            ..GameState::default()
        };
        let world = World::new(state);
        let map = world.map();
        let start = map.idx(0, 0).expect("start");
        let next = map.idx(1, 0).expect("next");
        let blocked = vec![crate::team_context::BlockedMove {
            from: start,
            dx: 1,
            dy: 0,
            count: 3,
        }];
        let prohibited = is_hard_prohibited_step(
            start,
            next,
            map,
            &blocked,
            &HashSet::new(),
            &HashSet::new(),
            &HashSet::new(),
        );
        assert!(
            !prohibited,
            "single legal exit must never be hard-prohibited"
        );
    }

    #[test]
    fn dropoff_zone_conflict_avoids_edge_swap() {
        let state = GameState {
            tick: 11,
            grid: Grid {
                width: 5,
                height: 3,
                drop_off_tiles: vec![[2, 1]],
                ..Grid::default()
            },
            bots: vec![
                BotState {
                    id: "a".to_owned(),
                    x: 1,
                    y: 1,
                    carrying: vec![],
                    capacity: 3,
                },
                BotState {
                    id: "b".to_owned(),
                    x: 2,
                    y: 1,
                    carrying: vec![],
                    capacity: 3,
                },
            ],
            ..GameState::default()
        };
        let world = World::new(state.clone());
        let map = world.map();
        let dist = DistanceMap::build(map);
        let planner = MotionPlanner::new(16);
        let mut goals = HashMap::new();
        goals.insert("a".to_owned(), map.idx(2, 1).expect("goal a"));
        goals.insert("b".to_owned(), map.idx(1, 1).expect("goal b"));

        let mut reservation = MovementReservation::default();
        reservation.priorities.insert("a".to_owned(), 0);
        reservation.priorities.insert("b".to_owned(), 1);
        reservation.reserve_horizon.insert("a".to_owned(), 2);
        reservation.reserve_horizon.insert("b".to_owned(), 2);
        reservation
            .dropoff_control_zone
            .extend([map.idx(1, 1).expect("z1"), map.idx(2, 1).expect("z2")]);

        let result = planner.plan(&state, map, &dist, &goals, &reservation, &[], None, 1);
        let a_action = result.actions.get("a").expect("a action");
        let b_action = result.actions.get("b").expect("b action");
        let is_swap = matches!(a_action.action, Action::Move { dx: 1, dy: 0, .. })
            && matches!(b_action.action, Action::Move { dx: -1, dy: 0, .. });
        assert!(
            !is_swap,
            "planner must avoid direct edge swap in control zone"
        );
    }

    #[test]
    fn explicit_order_is_respected_deterministically() {
        let state = GameState {
            tick: 7,
            grid: Grid {
                width: 4,
                height: 3,
                ..Grid::default()
            },
            bots: vec![
                BotState {
                    id: "a".to_owned(),
                    x: 1,
                    y: 1,
                    carrying: vec![],
                    capacity: 3,
                },
                BotState {
                    id: "b".to_owned(),
                    x: 1,
                    y: 2,
                    carrying: vec![],
                    capacity: 3,
                },
            ],
            ..GameState::default()
        };
        let world = World::new(state.clone());
        let map = world.map();
        let dist = DistanceMap::build(map);
        let planner = MotionPlanner::new(16);
        let mut goals = HashMap::new();
        let shared = map.idx(2, 1).expect("goal");
        goals.insert("a".to_owned(), shared);
        goals.insert("b".to_owned(), shared);
        let mut reservation = MovementReservation::default();
        reservation.priorities.insert("a".to_owned(), 0);
        reservation.priorities.insert("b".to_owned(), 0);
        reservation.reserve_horizon.insert("a".to_owned(), 1);
        reservation.reserve_horizon.insert("b".to_owned(), 1);
        let explicit_order = vec!["b".to_owned(), "a".to_owned()];

        let first = planner.plan(
            &state,
            map,
            &dist,
            &goals,
            &reservation,
            &explicit_order,
            None,
            1,
        );
        let second = planner.plan(
            &state,
            map,
            &dist,
            &goals,
            &reservation,
            &explicit_order,
            None,
            1,
        );
        let a1 = first.actions.get("a").expect("a action first");
        let b1 = first.actions.get("b").expect("b action first");
        let a2 = second.actions.get("a").expect("a action second");
        let b2 = second.actions.get("b").expect("b action second");
        assert_eq!(
            format!("{:?}", a1.action),
            format!("{:?}", a2.action),
            "explicit ordering must produce deterministic action for bot a"
        );
        assert_eq!(
            format!("{:?}", b1.action),
            format!("{:?}", b2.action),
            "explicit ordering must produce deterministic action for bot b"
        );
    }
}
