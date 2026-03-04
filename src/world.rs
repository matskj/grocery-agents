use std::{
    collections::{HashMap, VecDeque},
    hash::{Hash, Hasher},
    sync::{Arc, Mutex, OnceLock},
};

use smallvec::SmallVec;

use crate::model::GameState;

#[derive(Debug)]
pub struct World {
    map: Arc<MapCache>,
}

#[derive(Debug)]
pub struct MapCache {
    pub width: i32,
    pub height: i32,
    pub wall_mask: Vec<bool>,
    pub neighbors: Vec<SmallVec<[u16; 4]>>,
    pub dropoff_cells: Vec<u16>,
    pub item_shelf_cells: Vec<u16>,
    pub item_stand_cells: Vec<SmallVec<[u16; 4]>>,
    pub item_by_id: HashMap<String, usize>,
    pub dropoff_bfs: Vec<u16>,
    pub choke_points: Vec<bool>,
    pub intersection_points: Vec<bool>,
    pub aisle_id_by_cell: Vec<u16>,
    pub aisle_dropoff_dist: Vec<u16>,
    pub aisle_vertical: Vec<bool>,
}

impl World {
    pub fn new(state: &GameState) -> Self {
        let map = shared_map_cache(&state);
        Self { map }
    }

    pub fn map(&self) -> &MapCache {
        &self.map
    }

    pub fn map_arc(&self) -> Arc<MapCache> {
        Arc::clone(&self.map)
    }
}

impl MapCache {
    pub fn idx(&self, x: i32, y: i32) -> Option<u16> {
        if x < 0 || y < 0 || x >= self.width || y >= self.height {
            None
        } else {
            Some((y * self.width + x) as u16)
        }
    }

    pub fn xy(&self, idx: u16) -> (i32, i32) {
        let idx = idx as i32;
        (idx % self.width, idx / self.width)
    }

    pub fn is_wall(&self, idx: u16) -> bool {
        self.wall_mask[idx as usize]
    }

    pub fn stand_cells_for_item(&self, item_id: &str) -> &[u16] {
        self.item_by_id
            .get(item_id)
            .map(|ix| self.item_stand_cells[*ix].as_slice())
            .unwrap_or(&[])
    }

    pub fn is_choke_point(&self, idx: u16) -> bool {
        self.choke_points
            .get(idx as usize)
            .copied()
            .unwrap_or(false)
    }
}

#[derive(Clone, Debug, Eq)]
struct MapSignature {
    width: i32,
    height: i32,
    walls: Vec<[i32; 2]>,
    drop: Vec<[i32; 2]>,
    shelves: Vec<[i32; 2]>,
}

impl PartialEq for MapSignature {
    fn eq(&self, other: &Self) -> bool {
        self.width == other.width
            && self.height == other.height
            && self.walls == other.walls
            && self.drop == other.drop
            && self.shelves == other.shelves
    }
}

impl Hash for MapSignature {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.width.hash(state);
        self.height.hash(state);
        self.walls.hash(state);
        self.drop.hash(state);
        self.shelves.hash(state);
    }
}

fn shared_map_cache(state: &GameState) -> Arc<MapCache> {
    static CACHE: OnceLock<Mutex<HashMap<MapSignature, Arc<MapCache>>>> = OnceLock::new();

    let mut walls = state.grid.walls.clone();
    walls.sort_unstable();
    let mut drop = state.grid.drop_off_tiles.clone();
    drop.sort_unstable();
    let mut shelves = state.items.iter().map(|i| [i.x, i.y]).collect::<Vec<_>>();
    shelves.sort_unstable();

    let sig = MapSignature {
        width: state.grid.width,
        height: state.grid.height,
        walls,
        drop,
        shelves,
    };

    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache_guard = cache.lock().expect("map cache poisoned");
    if let Some(found) = cache_guard.get(&sig) {
        return Arc::clone(found);
    }

    let built = Arc::new(build_map_cache(state));
    cache_guard.insert(sig, Arc::clone(&built));
    built
}

fn build_map_cache(state: &GameState) -> MapCache {
    let width = state.grid.width.max(0);
    let height = state.grid.height.max(0);
    let n = (width * height) as usize;

    let mut wall_mask = vec![false; n];
    for [x, y] in &state.grid.walls {
        if *x >= 0 && *y >= 0 && *x < width && *y < height {
            wall_mask[(y * width + x) as usize] = true;
        }
    }
    // In the challenge protocol, item tiles are shelf cells and are not walkable.
    for item in &state.items {
        if item.x >= 0 && item.y >= 0 && item.x < width && item.y < height {
            wall_mask[(item.y * width + item.x) as usize] = true;
        }
    }

    let mut neighbors = vec![SmallVec::<[u16; 4]>::new(); n];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            if wall_mask[idx] {
                continue;
            }
            for (nx, ny) in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)] {
                if nx < 0 || ny < 0 || nx >= width || ny >= height {
                    continue;
                }
                let nidx = (ny * width + nx) as usize;
                if !wall_mask[nidx] {
                    neighbors[idx].push(nidx as u16);
                }
            }
        }
    }

    let mut dropoff_cells = Vec::new();
    for [x, y] in &state.grid.drop_off_tiles {
        if *x >= 0 && *y >= 0 && *x < width && *y < height {
            let idx = (y * width + x) as usize;
            if !wall_mask[idx] {
                dropoff_cells.push(idx as u16);
            }
        }
    }

    let mut dropoff_bfs = vec![u16::MAX; n];
    if !dropoff_cells.is_empty() {
        let mut queue = VecDeque::<u16>::new();
        for &drop in &dropoff_cells {
            dropoff_bfs[drop as usize] = 0;
            queue.push_back(drop);
        }
        while let Some(cell) = queue.pop_front() {
            let base = dropoff_bfs[cell as usize];
            for &nb in &neighbors[cell as usize] {
                if dropoff_bfs[nb as usize] != u16::MAX {
                    continue;
                }
                dropoff_bfs[nb as usize] = base.saturating_add(1);
                queue.push_back(nb);
            }
        }
    }

    let mut choke_points = vec![false; n];
    let mut intersection_points = vec![false; n];
    for idx in 0..n {
        if wall_mask[idx] {
            continue;
        }
        let deg = neighbors[idx].len();
        choke_points[idx] = deg <= 2;
        intersection_points[idx] = deg >= 3;
    }

    let mut aisle_id_by_cell = vec![u16::MAX; n];
    let mut aisle_dropoff_dist = Vec::<u16>::new();
    let mut aisle_vertical = Vec::<bool>::new();
    let mut next_aisle_id = 0u16;
    for idx in 0..n {
        if wall_mask[idx]
            || intersection_points[idx]
            || neighbors[idx].len() > 2
            || aisle_id_by_cell[idx] != u16::MAX
        {
            continue;
        }
        let mut stack = vec![idx as u16];
        let mut cells = Vec::<u16>::new();
        aisle_id_by_cell[idx] = next_aisle_id;
        while let Some(cell) = stack.pop() {
            cells.push(cell);
            for &nb in &neighbors[cell as usize] {
                let nb_usize = nb as usize;
                if wall_mask[nb_usize]
                    || intersection_points[nb_usize]
                    || neighbors[nb_usize].len() > 2
                    || aisle_id_by_cell[nb_usize] != u16::MAX
                {
                    continue;
                }
                aisle_id_by_cell[nb_usize] = next_aisle_id;
                stack.push(nb);
            }
        }
        let mut min_drop = u16::MAX;
        let mut min_x = i32::MAX;
        let mut max_x = i32::MIN;
        let mut min_y = i32::MAX;
        let mut max_y = i32::MIN;
        for &cell in &cells {
            min_drop = min_drop.min(dropoff_bfs[cell as usize]);
            let (x, y) = ((cell as i32) % width, (cell as i32) / width);
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
        aisle_dropoff_dist.push(min_drop);
        aisle_vertical.push((max_y - min_y) >= (max_x - min_x));
        next_aisle_id = next_aisle_id.saturating_add(1);
    }

    let mut item_shelf_cells = Vec::with_capacity(state.items.len());
    let mut item_stand_cells = Vec::with_capacity(state.items.len());
    let mut item_by_id = HashMap::with_capacity(state.items.len());

    for (ix, item) in state.items.iter().enumerate() {
        item_by_id.insert(item.id.clone(), ix);
        let shelf = if item.x >= 0 && item.y >= 0 && item.x < width && item.y < height {
            (item.y * width + item.x) as u16
        } else {
            0
        };
        item_shelf_cells.push(shelf);

        let mut stands = SmallVec::<[u16; 4]>::new();
        for (nx, ny) in [
            (item.x - 1, item.y),
            (item.x + 1, item.y),
            (item.x, item.y - 1),
            (item.x, item.y + 1),
        ] {
            if nx < 0 || ny < 0 || nx >= width || ny >= height {
                continue;
            }
            let nidx = (ny * width + nx) as usize;
            if !wall_mask[nidx] {
                stands.push(nidx as u16);
            }
        }
        item_stand_cells.push(stands);
    }

    MapCache {
        width,
        height,
        wall_mask,
        neighbors,
        dropoff_cells,
        item_shelf_cells,
        item_stand_cells,
        item_by_id,
        dropoff_bfs,
        choke_points,
        intersection_points,
        aisle_id_by_cell,
        aisle_dropoff_dist,
        aisle_vertical,
    }
}

#[cfg(test)]
mod tests {
    use crate::model::{GameState, Grid, Item};

    use super::World;

    #[test]
    fn item_shelf_cells_are_blocked_for_pathing() {
        let state = GameState {
            grid: Grid {
                width: 5,
                height: 5,
                ..Grid::default()
            },
            items: vec![Item {
                id: "item_1".to_owned(),
                kind: "milk".to_owned(),
                x: 2,
                y: 2,
            }],
            ..GameState::default()
        };

        let world = World::new(&state);
        let map = world.map();
        let shelf_idx = map.idx(2, 2).expect("shelf idx");
        let left_idx = map.idx(1, 2).expect("left idx");
        assert!(
            map.wall_mask[shelf_idx as usize],
            "shelf tile must be blocked"
        );
        assert!(
            !map.neighbors[left_idx as usize].contains(&shelf_idx),
            "path graph must not include shelf tile as neighbor"
        );
    }
}
