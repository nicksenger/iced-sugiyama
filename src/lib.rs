mod layout_engine;
pub use layout_engine::graphviz_plain_layout;

#[cfg(feature = "animated")]
mod animated;
#[cfg(feature = "animated")]
pub use animated::*;

#[cfg(not(feature = "animated"))]
mod still;
#[cfg(not(feature = "animated"))]
pub use still::*;
