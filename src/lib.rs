#[cfg(feature = "animated")]
mod animated;
#[cfg(feature = "animated")]
pub use animated::*;

#[cfg(not(feature = "animated"))]
mod still;
#[cfg(not(feature = "animated"))]
pub use still::*;
