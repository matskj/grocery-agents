use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex, OnceLock},
};

use crate::world::MapCache;

const INF: u16 = u16::MAX;

#[derive(Debug, Clone)]
pub struct DistanceMap {
    n: usize,
    all_pairs: Vec<u16>,
    dist_to_dropoff: Vec<u16>,
}

impl DistanceMap {
    pub fn build(map: &MapCache) -> Self {
        let n = (map.width * map.height).max(0) as usize;
        let mut all_pairs = vec![INF; n * n];
        let mut queue = VecDeque::with_capacity(n);
        let mut seen = vec![false; n];

        for src in 0..n {
            if map.wall_mask[src] {
                continue;
            }
            let row = src * n;
            all_pairs[row + src] = 0;
            queue.clear();
            seen.fill(false);
            queue.push_back(src as u16);
            seen[src] = true;

            while let Some(cell) = queue.pop_front() {
                let cell_usize = cell as usize;
                let base_d = all_pairs[row + cell_usize];
                for &nb in &map.neighbors[cell_usize] {
                    let nb_usize = nb as usize;
                    if !seen[nb_usize] {
                        seen[nb_usize] = true;
                        all_pairs[row + nb_usize] = base_d.saturating_add(1);
                        queue.push_back(nb);
                    }
                }
            }
        }

        let mut dist_to_dropoff = vec![INF; n];
        for cell in 0..n {
            if map.wall_mask[cell] {
                continue;
            }
            let mut best = INF;
            for &drop in &map.dropoff_cells {
                best = best.min(all_pairs[cell * n + drop as usize]);
            }
            dist_to_dropoff[cell] = best;
        }

        Self {
            n,
            all_pairs,
            dist_to_dropoff,
        }
    }

    pub fn dist(&self, a: u16, b: u16) -> u16 {
        self.all_pairs[a as usize * self.n + b as usize]
    }

    pub fn dist_to_dropoff(&self, a: u16) -> u16 {
        self.dist_to_dropoff[a as usize]
    }

    pub fn shared_for(map: &MapCache) -> Arc<Self> {
        static CACHE: OnceLock<Mutex<HashMap<usize, Arc<DistanceMap>>>> = OnceLock::new();
        let key = map as *const MapCache as usize;
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let mut guard = cache.lock().expect("distance cache poisoned");
        if let Some(found) = guard.get(&key) {
            return Arc::clone(found);
        }
        let built = Arc::new(Self::build(map));
        guard.insert(key, Arc::clone(&built));
        built
    }
}
