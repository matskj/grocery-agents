use crate::world::World;

pub fn advance(world: &mut World) {
    world.state_mut().tick += 1;
}
