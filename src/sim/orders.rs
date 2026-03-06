use std::collections::HashSet;

use rand::{rngs::StdRng, Rng};

use crate::{
    difficulty::Difficulty,
    model::{Item, Order, OrderStatus},
};

#[derive(Debug, Clone)]
pub struct OrderGenerator {
    pub difficulty: Difficulty,
}

impl Default for OrderGenerator {
    fn default() -> Self {
        Self {
            difficulty: Difficulty::Easy,
        }
    }
}

impl OrderGenerator {
    pub fn for_difficulty(difficulty: Difficulty) -> Self {
        Self { difficulty }
    }

    pub fn generate_order_entries(
        &self,
        order_index: u64,
        items: &[Item],
        rng: &mut StdRng,
    ) -> Vec<Order> {
        let kinds = items
            .iter()
            .map(|item| item.kind.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let mut kinds = kinds;
        kinds.sort_unstable();
        if kinds.is_empty() {
            return Vec::new();
        }
        let (min_items, max_items) = match self.difficulty {
            Difficulty::Easy => (3usize, 4usize),
            Difficulty::Medium => (4usize, 5usize),
            Difficulty::Hard => (4usize, 6usize),
            Difficulty::Expert => (4usize, 6usize),
            Difficulty::Custom => (3usize, 5usize),
        };
        let count = rng.gen_range(min_items..=max_items).min(kinds.len().max(1));

        let mut picked = Vec::<String>::new();
        while picked.len() < count {
            let idx = rng.gen_range(0..kinds.len());
            let kind = kinds[idx].clone();
            if !picked.contains(&kind) {
                picked.push(kind);
            }
        }

        picked
            .into_iter()
            .enumerate()
            .map(|(slot, kind)| Order {
                id: format!("order_{order_index}:{kind}:{slot}"),
                item_id: kind,
                status: if slot == 0 {
                    OrderStatus::InProgress
                } else {
                    OrderStatus::Pending
                },
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::OrderGenerator;
    use crate::{difficulty::Difficulty, model::Item};

    #[test]
    fn seeded_generation_is_deterministic() {
        let items = vec![
            Item {
                id: "i1".to_owned(),
                kind: "milk".to_owned(),
                x: 1,
                y: 1,
            },
            Item {
                id: "i2".to_owned(),
                kind: "bread".to_owned(),
                x: 2,
                y: 1,
            },
            Item {
                id: "i3".to_owned(),
                kind: "eggs".to_owned(),
                x: 3,
                y: 1,
            },
            Item {
                id: "i4".to_owned(),
                kind: "oats".to_owned(),
                x: 4,
                y: 1,
            },
        ];
        let generator = OrderGenerator::for_difficulty(Difficulty::Easy);
        let mut a = rand::rngs::StdRng::seed_from_u64(42);
        let mut b = rand::rngs::StdRng::seed_from_u64(42);

        let one = generator.generate_order_entries(1, &items, &mut a);
        let two = generator.generate_order_entries(1, &items, &mut b);
        assert_eq!(
            one.iter().map(|o| o.item_id.clone()).collect::<Vec<_>>(),
            two.iter().map(|o| o.item_id.clone()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn respects_difficulty_order_size_bounds() {
        let items = vec![
            Item {
                id: "i1".to_owned(),
                kind: "milk".to_owned(),
                x: 1,
                y: 1,
            },
            Item {
                id: "i2".to_owned(),
                kind: "bread".to_owned(),
                x: 2,
                y: 1,
            },
            Item {
                id: "i3".to_owned(),
                kind: "eggs".to_owned(),
                x: 3,
                y: 1,
            },
            Item {
                id: "i4".to_owned(),
                kind: "oats".to_owned(),
                x: 4,
                y: 1,
            },
            Item {
                id: "i5".to_owned(),
                kind: "apple".to_owned(),
                x: 5,
                y: 1,
            },
            Item {
                id: "i6".to_owned(),
                kind: "rice".to_owned(),
                x: 6,
                y: 1,
            },
        ];
        let cases = [
            (Difficulty::Easy, 3usize, 4usize),
            (Difficulty::Medium, 4usize, 5usize),
            (Difficulty::Hard, 4usize, 6usize),
            (Difficulty::Expert, 4usize, 6usize),
        ];

        for (difficulty, min_size, max_size) in cases {
            let generator = OrderGenerator::for_difficulty(difficulty);
            let mut rng = rand::rngs::StdRng::seed_from_u64(11);
            for order_ix in 0..20 {
                let out = generator.generate_order_entries(order_ix, &items, &mut rng);
                assert!(
                    out.len() >= min_size && out.len() <= max_size,
                    "difficulty={:?} produced {}",
                    difficulty,
                    out.len()
                );
            }
        }
    }
}
