use crate::model::GameState;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Difficulty {
    Easy,
    Medium,
    Hard,
    Expert,
    Custom,
}

impl Difficulty {
    pub fn as_label(self) -> &'static str {
        match self {
            Self::Easy => "easy",
            Self::Medium => "medium",
            Self::Hard => "hard",
            Self::Expert => "expert",
            Self::Custom => "custom",
        }
    }
}

pub fn infer_difficulty(state: &GameState) -> Difficulty {
    match (state.bots.len(), state.grid.width, state.grid.height) {
        (1, 12, 10) => Difficulty::Easy,
        (3, 16, 12) => Difficulty::Medium,
        (5, 22, 14) => Difficulty::Hard,
        (10, 28, 18) => Difficulty::Expert,
        _ => Difficulty::Custom,
    }
}

pub fn detect_mode_label(state: &GameState) -> &'static str {
    infer_difficulty(state).as_label()
}

#[cfg(test)]
mod tests {
    use crate::model::{GameState, Grid};

    use super::{infer_difficulty, Difficulty};

    #[test]
    fn infer_known_modes() {
        let mk = |w: i32, h: i32, bots: usize| GameState {
            grid: Grid {
                width: w,
                height: h,
                ..Grid::default()
            },
            bots: (0..bots)
                .map(|i| crate::model::BotState {
                    id: i.to_string(),
                    ..crate::model::BotState::default()
                })
                .collect(),
            ..GameState::default()
        };
        assert_eq!(infer_difficulty(&mk(12, 10, 1)), Difficulty::Easy);
        assert_eq!(infer_difficulty(&mk(16, 12, 3)), Difficulty::Medium);
        assert_eq!(infer_difficulty(&mk(22, 14, 5)), Difficulty::Hard);
        assert_eq!(infer_difficulty(&mk(28, 18, 10)), Difficulty::Expert);
        assert_eq!(infer_difficulty(&mk(30, 20, 6)), Difficulty::Custom);
    }
}
