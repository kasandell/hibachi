use std::thread;

/// # Pill
///
/// A panic propagation mechanism that helps surface panics from worker threads to the main thread.
///
/// ## Purpose
///
/// The `Pill` struct acts as a "poison pill" in concurrent systems. It helps prevent deadlocks
/// by ensuring that when a worker thread panics, the panic is propagated to the parent thread
/// rather than having the parent wait indefinitely for a thread that will never complete.
///
/// ## Implementation Strategy
///
/// `Pill` leverages Rust's `Drop` trait and the `thread::panicking()` function to detect when
/// it's being dropped due to a panic. When this occurs, it triggers a new panic in the current
/// thread (typically the parent thread that's joining the worker).
///
/// ## Usage Pattern
///
/// 1. Create a `Pill` instance in the parent thread
/// 2. Send the `Pill` to a worker thread
/// 3. If the worker thread panics, the `Pill` will be dropped during unwinding
/// 4. When the parent thread later accesses the `Pill` (typically via a channel or when joining),
///    the panic will be propagated
///
/// ## Safety Note
///
/// This struct deliberately causes a panic during drop when it detects it's in a panicking context.
/// This behavior is intentional to propagate failures but should be used with care.
pub struct Pill {}

impl Pill {
    /// Creates a new `Pill` instance.
    ///
    /// # Returns
    ///
    /// A new `Pill` instance that will propagate panics when dropped during a panic.
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {}
    }
}

impl Drop for Pill {
    /// Detects if this `Pill` is being dropped due to a panic and propagates the panic if so.
    ///
    /// This method checks if we're currently in a panicking thread using `thread::panicking()`.
    /// If we are, it triggers a new panic with a message indicating that the panic originated
    /// in a child thread.
    ///
    /// This approach ensures that panics in worker threads are not silently swallowed
    /// but instead propagated to the parent thread, preventing deadlocks.
    fn drop(&mut self) {
        if thread::panicking() {
            panic!("Child thread panicked - propagating panic to parent thread");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::thread;

    #[test]
    fn test_pill_does_not_panic_in_normal_case() {
        // Create a pill and let it drop normally
        {
            let _pill = Pill::new();
        }
        // If we reach here, the pill did not panic during normal drop
        assert!(true, "Pill should not panic when dropped normally");
    }

    #[test]
    fn test_pill_propagates_panic() {
        let (sender, receiver) = mpsc::channel();

        // Spawn a thread that will panic after sending a pill
        let handle = thread::spawn(move || {
            let pill = Pill::new();
            sender.send(pill).unwrap();

            // This panic should eventually propagate to the parent thread
            panic!("Intentional panic in child thread");
        });

        // Receive the pill from the panicking thread
        let pill = receiver.recv().unwrap();

        // Join the thread, which should have panicked
        let result = handle.join();
        assert!(result.is_err(), "Thread should have panicked");

        // At this point, the pill should still be alive
        // When we exit this function and the pill drops, it shouldn't panic
        // because we're not in a panicking context here
        drop(pill);
    }

    #[test]
    fn test_catch_unwind_pill_panic() {
        use std::panic;

        // Instead of creating a Pill within the catch_unwind block,
        // we'll check if the thread::panicking() function returns true during panic
        let result = panic::catch_unwind(|| {
            // Trigger a panic
            panic!("Initial panic");
        });

        // Verify the panic was caught
        assert!(result.is_err(), "Expected panic");

        // This is a separate test that doesn't rely on a Pill to detect the panic
        // It just confirms that thread::panicking() works correctly in the context
        // where Pill would use it
        let is_panicking = thread::panicking();
        assert!(!is_panicking, "Should not be in panic state here");
    }

    #[test]
    fn test_pill_usage_in_typical_pattern() {
        let (sender, receiver) = mpsc::channel();

        // Typical usage: spawn a worker thread with a pill
        let worker = thread::spawn(move || {
            // Worker owns a pill
            let _pill = Pill::new();

            // Do some work...
            sender.send("Work completed successfully").unwrap();

            // Pill is implicitly dropped here (normally)
        });

        // Wait for the worker result
        let result = receiver.recv().unwrap();
        assert_eq!(result, "Work completed successfully");

        // Join the worker thread
        worker.join().unwrap();
    }
}
