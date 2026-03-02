use crate::model::Action;

#[derive(Debug, Default)]
pub struct Dispatcher;

impl Dispatcher {
    pub fn new() -> Self {
        Self
    }

    pub fn dispatch(&self, action: Action) -> Action {
        action
    }
}
