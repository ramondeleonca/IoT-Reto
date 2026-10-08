#include <Arduino.h>

void setup() {
  // Iniciamos la Comunicacion Serial a 9600 baudios
  Serial.begin(9600);
}

void loop() {
  int sensorValue = analogRead(2);              // Lectura del ADC
  float voltage = sensorValue * (3.3 / 1023.0); // Escalamos a voltaje

  // Enviamos por el puerto serie el valor obtenido en ADC y el valor obtenido
  // de voltaje
  Serial.print("ADC= ");
  Serial.print(sensorValue);
  Serial.print("  Voltaje= ");
  Serial.println(voltage);

  delay(100);
}
