#include "esp32-hal-gpio.h"
#include <Arduino.h>

// Pins for infrared / optical barrier sensors
constexpr uint8_t sensor1 = 18;
constexpr uint8_t sensor2 = 19;

// Physical distance between sensor 1 and sensor 2 in centimeters
constexpr float distanceBetweenSensorsCm = 1.0f;

// Timeout in microseconds before resetting measurement state (e.g., 3 seconds)
constexpr unsigned long timeoutMicros = 3000000UL;

enum class GateState {
    WAITING_FOR_SENSOR1,
    WAITING_FOR_SENSOR2
};

GateState currentState = GateState::WAITING_FOR_SENSOR1;
unsigned long sensor1TimestampUs = 0;
float objectVelocityCmPerSecond = 0.0f;

void setup() {
    Serial.begin(115200);

    // Active-LOW input sensors with internal pull-up enabled
    pinMode(sensor1, INPUT_PULLUP);
    pinMode(sensor2, INPUT_PULLUP);

    Serial.println("Ready: Waiting for object at sensor 1...");
}

void loop() {
    const unsigned long nowUs = micros();

    switch (currentState) {
        case GateState::WAITING_FOR_SENSOR1:
            // Trigger when sensor 1 detects object (active LOW)
            if (digitalRead(sensor1) == LOW) {
                sensor1TimestampUs = nowUs;
                currentState = GateState::WAITING_FOR_SENSOR2;
            }
            break;

        case GateState::WAITING_FOR_SENSOR2:
            // Check for timeout if object never reached sensor 2
            if (nowUs - sensor1TimestampUs > timeoutMicros) {
                Serial.println("Measurement timed out: Resetting.");
                currentState = GateState::WAITING_FOR_SENSOR1;
            } else if (digitalRead(sensor2) == LOW) {
                // Trigger when sensor 2 detects object
                const unsigned long sensor2TimestampUs = nowUs;
                const unsigned long elapsedUs = sensor2TimestampUs - sensor1TimestampUs;

                if (elapsedUs > 0) {
                    // velocity (cm/s) = distance (cm) / (elapsedUs / 1,000,000)
                    //                 = distance (cm) * 1,000,000 / elapsedUs
                    objectVelocityCmPerSecond = (distanceBetweenSensorsCm * 1000000.0f) / static_cast<float>(elapsedUs);

                    Serial.printf("Time: %lu us | Speed: %.2f cm/s\n", elapsedUs, objectVelocityCmPerSecond);
                } else {
                    Serial.println("Warning: Elapsed time too short to measure reliably.");
                }

                // Debounce / wait until object clears sensor 2 before next cycle
                delay(200);
                currentState = GateState::WAITING_FOR_SENSOR1;
            }
            break;
    }
}
