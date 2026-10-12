# Changelog

## [2026-10-12 00:21:55] Switch sensors to continuous ranging
- **Prompt:** "let's do that, but 40ms still feels slow"
- **Summary:** Use 20 ms continuous ranging and consume completed measurements without blocking in [main.hpp](../src/impl/main.hpp).

## [2026-10-12 00:16:38] Add velocity threshold and active LED
- **Prompt:** "add a small print for when the signal is no longer active, also make the signal only trigger if the velocity is over #sym:thresholdVelocityMMperS, also turn on the built in LED while the signal is active"
- **Summary:** Gate activation on the configured velocity threshold, mirror the active signal on the built-in LED, and report deactivation in [main.hpp](../src/impl/main.hpp).

## [2026-10-12 00:11:50] Refresh active signal for new objects
- **Prompt:** "instead of blocking itself because of the reset time, it should reset itself if a new object passing by was detected (and then wait another 2000ms before deactivating)"
- **Summary:** Continue pairing new sensor edges while active and restart the reset timer for each valid pair in [main.hpp](../src/impl/main.hpp).

## [2026-10-12 00:05:50] Simplify sensor trigger logic
- **Prompt:** "that code looks VERY ugly, keep the same structure i had before and get rid of that, the idea is there but it looks really bad and unnecessarily complicated"
- **Summary:** Restore the original loop structure and simplify trigger timing and pairing in [main.hpp](../src/impl/main.hpp).

## [2026-10-12 00:03:14] Latch sensor trigger signal
- **Prompt:** "make the least amount of changes needed to make this code react to sudden changes and not just after every 2000ms, i need it to trigger in the window, after both sensors were triggered inside the window, the signal should stay active until the reset time"
- **Summary:** Detect sensor entry transitions, pair them within the trigger window, and hold `active` for the reset duration in [main.hpp](../src/impl/main.hpp).

## [2026-10-11 22:23:53] Measure velocity between time-of-flight sensors
- **Prompt:** "modify as little as possible in the code to detect object velocity based on the time it takes to cross from one sensor to the other and the distance set as sensorDistanceMM, always log the distance, but when an object is detected log it to serial"
- **Summary:** Keep distance output continuous; detect objects within 30 mm, calculate crossing velocity, and fix sensor 2's I2C address in [tof.hpp](../src/impl/tof.hpp) and [tof_servo.hpp](../src/impl/tof_servo.hpp).

## [2026-10-11 19:21:55] Reject invalid time-of-flight readings
- **Prompt:** "this code is always outputting 10912.00mm, 10912.00mm; make the most minimal, necessary changes to make this code work and output the real distance"
- **Summary:** Wait for both VL53L0X sensors to boot; initialize measurement data and reject failed or invalid results instead of printing uninitialized values in [tof.hpp](../src/impl/tof.hpp).

## [2026-10-08 13:16:00] Fix speed calculation and timing logic
- **Prompt:** "fix the code, it just says 999"
- **Summary:** Converted velocity calculations to microsecond precision, added edge-triggered state machine for photogate inputs, fixed distance/time unit conversions, and added timeout handling in [main.hpp](file:///Users/ramon/Documents/Code/IoT-Reto/mcu/src/impl/main.hpp).
