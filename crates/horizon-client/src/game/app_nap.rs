//! macOS App Nap: a window that is hidden, occluded or behind another app has
//! its timers coalesced and its threads' quality of service clamped, the render
//! thread's `favour_current` included. While terrain loads in the background or
//! the game runs `--offscreen`, frames then waited up to a second for a core
//! with 80 ms of work in them. A game is latency critical for as long as it
//! runs, so it holds one activity for the whole process, as the system asks
//! (`NSProcessInfo.beginActivityWithOptions:reason:`); idle system sleep
//! stays allowed.

/// Opts the process out of App Nap for its lifetime. Whether the system took it.
#[cfg(target_os = "macos")]
pub fn hold_off() -> bool {
    use objc2_foundation::{NSActivityOptions, NSProcessInfo, NSString};
    let options = NSActivityOptions::UserInitiatedAllowingIdleSystemSleep
        | NSActivityOptions::LatencyCritical;
    let reason = NSString::from_str("Rendering and simulating the game");
    let activity = NSProcessInfo::processInfo().beginActivityWithOptions_reason(options, &reason);
    // Ended only by the process exiting: the token must outlive every frame.
    std::mem::forget(activity);
    true
}

#[cfg(not(target_os = "macos"))]
pub fn hold_off() -> bool {
    false
}
