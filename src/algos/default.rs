use crate::layout_engine::{GraphLayout, LayoutInput};

/// The default Sugiyama layout algorithm.
pub fn default_layout<'a>(input: &LayoutInput<'a>) -> GraphLayout {
    input.compute()
}
