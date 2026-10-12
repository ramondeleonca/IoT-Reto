#include <Arduino.h>
#include <WiFi.h>
#include <esp_wifi.h>
#include <esp_now.h>
#include <Wire.h>
#include <Adafruit_VL53L0X.h>

// #define SENSOR_1_ONLY

constexpr uint8_t lox1xshut = 18;
constexpr uint8_t lox1addr = 0x30;
unsigned long sensor1Time = 0;
Adafruit_VL53L0X lox1 = Adafruit_VL53L0X();
VL53L0X_RangingMeasurementData_t lox1measure;
float lox1Dist = -1;

constexpr uint8_t lox2xshut = 19;
constexpr uint8_t lox2addr = 0x31;
#ifndef SENSOR_1_ONLY
unsigned long sensor2Time = 0;
Adafruit_VL53L0X lox2 = Adafruit_VL53L0X();
VL53L0X_RangingMeasurementData_t lox2measure;
float lox2Dist = -1;
#endif

constexpr uint16_t rangePeriodMs = 20;
constexpr uint32_t rangeBudgetUs = 20000;
constexpr float sensorDistanceMM = 1.0 /* inch */ * 2.54 /* to cm */ * 10 /* to mm */;
constexpr float thresholdMM = 90.0f;
constexpr float thresholdVelocityMMperS = 380.0f; // ! THIS IS THE IMPORTANT PARAMETER, BASICALLY THE SENSITIVITY
constexpr long resetTime = 2000;
constexpr long triggerWindow = 500;

volatile float objectVelocityMMperS;
bool active = false;
unsigned long activeSince = 0;
bool sensor1WasTriggered = false;
bool sensor2WasTriggered = false;

bool updateDistanceMM(Adafruit_VL53L0X &lox, float &distanceMM) {
    if (!lox.isRangeComplete()) return false;
    if (lox.Status != VL53L0X_ERROR_NONE) {
        distanceMM = -1;
        return true;
    }

    const uint16_t range = lox.readRangeResult();
    distanceMM = lox.Status == VL53L0X_ERROR_NONE &&
                 lox.readRangeStatus() == 0 && range != 0xffff
        ? range
        : -1;
    return true;
}

void setup() {
    Serial.begin(115200);
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, LOW);

    pinMode(lox1xshut, OUTPUT);
    digitalWrite(lox1xshut, LOW);
    
    pinMode(lox2xshut, OUTPUT);
    digitalWrite(lox2xshut, LOW);

    Wire.begin();

    digitalWrite(lox1xshut, HIGH);
    delay(10);
    if (!lox1.begin()) Serial.println("Error initializing sensor 1.");
    else if (!lox1.setAddress(lox1addr)) Serial.println("Error setting sensor 1 I2C address.");
    else if (!lox1.configSensor(Adafruit_VL53L0X::VL53L0X_SENSE_HIGH_SPEED)) Serial.println("Error configuring sensor 1.");
    else if (!lox1.setMeasurementTimingBudgetMicroSeconds(rangeBudgetUs)) Serial.println("Error setting sensor 1 timing budget.");
    else if (!lox1.startRangeContinuous(rangePeriodMs)) Serial.println("Error starting sensor 1 continuous ranging.");
    else Serial.println("Sensor 1 initialized correctly.");

    #ifndef SENSOR_1_ONLY
    digitalWrite(lox2xshut, HIGH);
    delay(10);
    if (!lox2.begin()) Serial.println("Error initializing sensor 2.");
    else if (!lox2.setAddress(lox2addr)) Serial.println("Error setting sensor 2 I2C address.");
    else if (!lox2.configSensor(Adafruit_VL53L0X::VL53L0X_SENSE_HIGH_SPEED)) Serial.println("Error configuring sensor 2.");
    else if (!lox2.setMeasurementTimingBudgetMicroSeconds(rangeBudgetUs)) Serial.println("Error setting sensor 2 timing budget.");
    else if (!lox2.startRangeContinuous(rangePeriodMs)) Serial.println("Error starting sensor 2 continuous ranging.");
    else Serial.println("Sensor 2 initialized correctly.");
    #endif
}

void loop() {
    const bool lox1Updated = updateDistanceMM(lox1, lox1Dist);
    #ifndef SENSOR_1_ONLY
    const bool lox2Updated = updateDistanceMM(lox2, lox2Dist);
    #endif

    if (lox1Updated
        #ifndef SENSOR_1_ONLY
        || lox2Updated
        #endif
    ) {
        Serial.print(lox1Dist);
        Serial.print("mm");

        Serial.print(", ");

        #ifndef SENSOR_1_ONLY
        Serial.print(lox2Dist);
        Serial.print("mm");
        #endif
        Serial.println();
    }
    
    if (active && millis() - activeSince >= resetTime) {
        active = false;
        digitalWrite(LED_BUILTIN, LOW);
        sensor1Time = 0;
        #ifndef SENSOR_1_ONLY
        sensor2Time = 0;
        #endif
        Serial.println("Signal no longer active");
    }

    if (lox1Updated) {
        const bool sensor1Triggered = lox1Dist > 0 && lox1Dist <= thresholdMM;
        if (sensor1Triggered && !sensor1WasTriggered) {
            sensor1Time = micros();
            Serial.println("Sensor 1 triggered");
        }
        sensor1WasTriggered = sensor1Triggered;
    }

    #ifndef SENSOR_1_ONLY
    if (lox2Updated) {
        const bool sensor2Triggered = lox2Dist > 0 && lox2Dist <= thresholdMM;
        if (sensor2Triggered && !sensor2WasTriggered) {
            sensor2Time = micros();
            Serial.println("Sensor 2 triggered");
        }
        sensor2WasTriggered = sensor2Triggered;
    }

    // Logic modified by gpt-6-luna
    const unsigned long triggerWindowUs = triggerWindow * 1000UL;
    if (sensor1Time != 0 && sensor2Time != 0) {
        const long deltaTimeUs = static_cast<long>(sensor2Time - sensor1Time);
        const unsigned long elapsedUs = deltaTimeUs < 0 ? -deltaTimeUs : deltaTimeUs;

        if (elapsedUs > 0 && elapsedUs <= triggerWindowUs) {
            objectVelocityMMperS = sensorDistanceMM * 1000000.0f / elapsedUs;
            if (objectVelocityMMperS > thresholdVelocityMMperS) {
                active = true;
                activeSince = millis();
                digitalWrite(LED_BUILTIN, HIGH);
                Serial.printf("Both sensors triggered in window: %.2f mm/s; signal active\n", objectVelocityMMperS);
            }
            sensor1Time = 0;
            sensor2Time = 0;
        } else if (deltaTimeUs > 0) {
            sensor1Time = 0;
        } else {
            sensor2Time = 0;
        }
    }

    if (sensor1Time != 0 && micros() - sensor1Time > triggerWindowUs) {
        sensor1Time = 0;
    }
    if (sensor2Time != 0 && micros() - sensor2Time > triggerWindowUs) {
        sensor2Time = 0;
    }
    #endif
    
    delay(1);
}