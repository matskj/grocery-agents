use crate::model::GameState;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Difficulty {
    Easy,
    Medium,
    Hard,
    Expert,
    Custom,
}

#[derive(Debug, Clone, Copy)]
pub struct DifficultyConfig {
    pub label: &'static str,
    pub map_width: i32,
    pub map_height: i32,
    pub bot_count: usize,
    pub reservation_horizon: u8,
    pub deep_plan_bot_cap: usize,
    pub regional_picker_count: usize,
    pub runner_count: usize,
    pub buffer_count: usize,
}

impl Difficulty {
    pub fn from_label(label: &str) -> Self {
        match label {
            "easy" => Self::Easy,
            "medium" => Self::Medium,
            "hard" => Self::Hard,
            "expert" => Self::Expert,
            _ => Self::Custom,
        }
    }

    pub fn as_label(self) -> &'static str {
        match self {
            Self::Easy => "easy",
            Self::Medium => "medium",
            Self::Hard => "hard",
            Self::Expert => "expert",
            Self::Custom => "custom",
        }
    }

    pub fn config(self) -> DifficultyConfig {
        match self {
            Self::Easy => DifficultyConfig {
                label: "easy",
                map_width: 12,
                map_height: 10,
                bot_count: 1,
                reservation_horizon: 2,
                deep_plan_bot_cap: 1,
                regional_picker_count: 1,
                runner_count: 0,
                buffer_count: 0,
            },
            Self::Medium => DifficultyConfig {
                label: "medium",
                map_width: 16,
                map_height: 12,
                bot_count: 3,
                reservation_horizon: 4,
                deep_plan_bot_cap: 3,
                regional_picker_count: 2,
                runner_count: 1,
                buffer_count: 0,
            },
            Self::Hard => DifficultyConfig {
                label: "hard",
                map_width: 22,
                map_height: 14,
                bot_count: 5,
                reservation_horizon: 6,
                deep_plan_bot_cap: 4,
                regional_picker_count: 4,
                runner_count: 1,
                buffer_count: 0,
            },
            Self::Expert => DifficultyConfig {
                label: "expert",
                map_width: 28,
                map_height: 18,
                bot_count: 10,
                reservation_horizon: 7,
                deep_plan_bot_cap: 5,
                regional_picker_count: 5,
                runner_count: 3,
                buffer_count: 2,
            },
            Self::Custom => DifficultyConfig {
                label: "custom",
                map_width: 0,
                map_height: 0,
                bot_count: 0,
                reservation_horizon: 5,
                deep_plan_bot_cap: 4,
                regional_picker_count: 3,
                runner_count: 1,
                buffer_count: 0,
            },
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
