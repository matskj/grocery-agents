use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::{
    dist::DistanceMap,
    model::{Action, GameState},
    world::MapCache,
};

#[derive(Debug, Clone, Default)]
pub struct MotionPlanner {
    horizon: u8,
}

#[derive(Debug, Clone)]
struct Node {
    f: u16,
    g: u8,
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
    ) -> HashMap<String, Action> {
        let mut out = HashMap::with_capacity(state.bots.len());
        let mut v_res: HashSet<(u8, u16)> = HashSet::new();
        let mut e_res: HashSet<(u8, u16, u16)> = HashSet::new();

        let mut bots = state.bots.iter().collect::<Vec<_>>();
        bots.sort_by(|a, b| a.id.cmp(&b.id));

        for bot in bots {
            let start = match map.idx(bot.x, bot.y) {
                Some(v) => v,
                None => continue,
            };
            let goal = goals.get(&bot.id).copied().unwrap_or(start);
            let step = self.next_step(start, goal, map, dist, &v_res, &e_res);
            let action = if step == start {
                Action::wait(bot.id.clone())
            } else {
                let (x0, y0) = map.xy(start);
                let (x1, y1) = map.xy(step);
                Action::Move {
                    bot_id: bot.id.clone(),
                    dx: x1 - x0,
                    dy: y1 - y0,
                }
            };

            v_res.insert((1, step));
            e_res.insert((1, start, step));
            e_res.insert((1, step, start));
            out.insert(bot.id.clone(), action);
        }

        out
    }

    fn next_step(
        &self,
        start: u16,
        goal: u16,
        map: &MapCache,
        dist: &DistanceMap,
        v_res: &HashSet<(u8, u16)>,
        e_res: &HashSet<(u8, u16, u16)>,
    ) -> u16 {
        if start == goal {
            return start;
        }
        let mut open = BinaryHeap::new();
        let mut best_g: HashMap<(u16, u8), u8> = HashMap::new();
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
            if t >= self.horizon {
                continue;
            }

            let nt = t + 1;
            let mut succ = map.neighbors[cell as usize].clone();
            succ.push(cell); // wait move
            for next in succ {
                if v_res.contains(&(nt, next)) || e_res.contains(&(nt, cell, next)) {
                    continue;
                }
                let ng = g + 1;
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

        let mut cur = (best_terminal.0, best_terminal.1);
        while let Some(prev) = parent.get(&cur).copied() {
            if prev.1 == 0 {
                return cur.0;
            }
            cur = prev;
        }
        start
    }
}
