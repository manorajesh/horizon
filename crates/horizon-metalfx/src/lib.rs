//! MetalFX frame interpolation for the game's frame generation (macOS 26 and later): the
//! Objective-C calls on wgpu's Metal objects that the render crate, which forbids unsafe
//! code, can't make itself. Nothing here exists on other systems.

#[cfg(target_os = "macos")]
mod metal;
#[cfg(target_os = "macos")]
pub use metal::{Frame, Interpolator, supported};
