mod algos;
mod layout_engine;

pub use algos::default::default_layout;
pub use iced_sugiyama_core::*;
pub use layout_engine::{
    graphviz_plain_layout, Cluster, EdgeEndpoint, EdgeEndpointKind, LayoutInput,
};

#[cfg(feature = "circo")]
pub use algos::circo::circo_layout;

#[cfg(feature = "microdot")]
pub use algos::microdot::microdot_layout;

mod animated;
pub use animated::*;
