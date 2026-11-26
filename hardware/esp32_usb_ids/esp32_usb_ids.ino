/*
 * Offline AdaptiveML hardware run for ESP32-S3.
 *
 * - Runs the exported lightweight IDS tree on the ESP32.
 * - Receives the model's exact 18 preprocessed inputs over USB serial.
 * - Reads INA219, DHT22, MPU6500, and SW-420 sensors; updates the OLED and buzzer.
 * - Uses no Wi-Fi, MQTT, fabricated telemetry, or sensor values as IDS features.
 *
 * The laptop can send saved held-out rows for replay or authorized live
 * packet-level metadata over USB after applying the saved training transform.
 * This is packet-level inference, not reconstructed flow analysis.
 */

#include <Arduino.h>
#include <Wire.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <Adafruit_INA219.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <DHT.h>
#include <MPU6500_WE.h>
#include "esp32_model.h"

static constexpr uint32_t SERIAL_BAUD = 115200;
static constexpr size_t MAX_LINE = 512;
static constexpr uint8_t SDA_PIN = 4;
static constexpr uint8_t SCL_PIN = 5;
static constexpr uint8_t DHT_PIN = 6;
static constexpr uint8_t VIBRATION_PIN = 7;
static constexpr uint8_t BUZZER_PIN = 15;
static constexpr uint8_t OLED_ADDRESS = 0x3C;
static constexpr float TEMPERATURE_ALERT_C = 40.0f;
static constexpr float MOVEMENT_THRESHOLD_G = 0.08f;
static constexpr uint32_t VIBRATION_HOLD_MS = 600;
static constexpr uint32_t MOVEMENT_HOLD_MS = 1000;
static constexpr uint32_t SENSOR_SAMPLE_MS = 100;
static constexpr uint32_t DHT_SAMPLE_MS = 2200;
static constexpr uint32_t DISPLAY_SAMPLE_MS = 500;
static constexpr uint32_t USB_SENSOR_REPORT_MS = 250;

static char input_line[MAX_LINE];
static size_t input_length = 0;
static Adafruit_INA219 ina219;
static Adafruit_SSD1306 display(128, 64, &Wire, -1);
static DHT dht(DHT_PIN, DHT22);
static MPU6500_WE mpu6500(0x68);
static bool ina_ready = false;
static bool oled_ready = false;
static bool mpu_ready = false;
static int vibration_idle_level = LOW;
static float temperature_c = NAN, humidity_pct = NAN;
static float accel_x_g = NAN, accel_y_g = NAN, accel_z_g = NAN;
static float voltage_v = NAN, current_a = NAN, power_w = NAN;
static float cumulative_energy_j = 0.0f;
static uint32_t last_sensor_ms = 0, last_dht_ms = 0, last_display_ms = 0;
static uint32_t last_usb_sensor_report_ms = 0;
static uint32_t last_power_ms = 0, vibration_last_seen_ms = 0, movement_last_seen_ms = 0;
static float previous_x = NAN, previous_y = NAN, previous_z = NAN, previous_power_w = NAN;
static bool vibration_now = false, movement_now = false;
static bool last_alert = false;
static char last_prediction[32] = "Not run";

static void emitError(const char *reason) {
  Serial.print("ERROR,");
  Serial.println(reason);
}

static void printFloatOrNA(float value, uint8_t decimals) {
  if (isfinite(value)) Serial.print(value, decimals);
  else Serial.print("NA");
}

static void displayFloatOrNA(float value, uint8_t decimals) {
  if (isfinite(value)) display.print(value, decimals);
  else display.print("NA");
}

static void setBuzzer(bool enabled) {
  if (enabled) ledcWriteTone(BUZZER_PIN, 1600);
  else ledcWriteTone(BUZZER_PIN, 0);
}

static void calibrateVibrationSensor() {
  Serial.println("CALIBRATING_SW420; keep it still");
  delay(500);
  uint16_t low_count = 0, high_count = 0;
  for (uint16_t i = 0; i < 100; ++i) {
    if (digitalRead(VIBRATION_PIN) == LOW) ++low_count;
    else ++high_count;
    delay(10);
  }
  vibration_idle_level = high_count >= low_count ? HIGH : LOW;
  Serial.print("SW420_IDLE_LEVEL,");
  Serial.println(vibration_idle_level == HIGH ? "HIGH" : "LOW");
}

static void readSensors(uint32_t now) {
  if (now - last_sensor_ms >= SENSOR_SAMPLE_MS || last_sensor_ms == 0) {
    last_sensor_ms = now;
    const int vibration_raw = digitalRead(VIBRATION_PIN);
    if (vibration_raw != vibration_idle_level) vibration_last_seen_ms = now;
    vibration_now = vibration_last_seen_ms != 0 && now - vibration_last_seen_ms <= VIBRATION_HOLD_MS;

    if (mpu_ready) {
      const xyzFloat acc = mpu6500.getGValues();
      accel_x_g = acc.x;
      accel_y_g = acc.y;
      accel_z_g = acc.z;
      if (isfinite(previous_x)) {
        const float dx = accel_x_g - previous_x;
        const float dy = accel_y_g - previous_y;
        const float dz = accel_z_g - previous_z;
        if (sqrtf(dx * dx + dy * dy + dz * dz) > MOVEMENT_THRESHOLD_G) movement_last_seen_ms = now;
      }
      previous_x = accel_x_g;
      previous_y = accel_y_g;
      previous_z = accel_z_g;
      movement_now = movement_last_seen_ms != 0 && now - movement_last_seen_ms <= MOVEMENT_HOLD_MS;
    }

    if (ina_ready) {
      voltage_v = ina219.getBusVoltage_V();
      current_a = ina219.getCurrent_mA() / 1000.0f;
      power_w = voltage_v * current_a;
      if (isfinite(power_w) && last_power_ms != 0 && now > last_power_ms) {
        const float elapsed_s = (now - last_power_ms) / 1000.0f;
        cumulative_energy_j += (fmaxf(0.0f, power_w) + fmaxf(0.0f, previous_power_w)) * 0.5f * elapsed_s;
      }
      previous_power_w = power_w;
      last_power_ms = now;
    }
  }

  if (now - last_dht_ms >= DHT_SAMPLE_MS || last_dht_ms == 0) {
    last_dht_ms = now;
    const float t = dht.readTemperature();
    const float h = dht.readHumidity();
    temperature_c = isfinite(t) ? t : NAN;
    humidity_pct = isfinite(h) ? h : NAN;
  }

  last_alert = vibration_now || (isfinite(temperature_c) && temperature_c >= TEMPERATURE_ALERT_C);
  setBuzzer(last_alert);
}

static void updateDisplay() {
  if (!oled_ready) return;
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println("AdaptiveML USB IDS");
  display.print("T:"); displayFloatOrNA(temperature_c, 1); display.print("C H:"); displayFloatOrNA(humidity_pct, 0); display.println("%");
  display.print("Vib:"); display.print(vibration_now ? "YES" : "NO");
  display.print(" Move:"); display.println(movement_now ? "YES" : "NO");
  display.print("Power:"); displayFloatOrNA(power_w, 3); display.println("W");
  display.print("Energy:"); displayFloatOrNA(cumulative_energy_j, 2); display.println("J");
  display.print("Accel:"); displayFloatOrNA(accel_x_g, 1); display.print(","); displayFloatOrNA(accel_y_g, 1); display.print(","); displayFloatOrNA(accel_z_g, 1); display.println("g");
  display.print("IDS:"); display.println(last_prediction);
  display.print("Alert:"); display.print(last_alert ? "ACTIVE" : "Normal");
  display.print(" WiFi:OFF");
  display.display();
}

static void emitSensorFields() {
  printFloatOrNA(voltage_v, 4); Serial.print(',');
  printFloatOrNA(current_a, 6); Serial.print(',');
  printFloatOrNA(power_w, 5); Serial.print(',');
  printFloatOrNA(cumulative_energy_j, 5); Serial.print(',');
  printFloatOrNA(temperature_c, 2); Serial.print(',');
  printFloatOrNA(humidity_pct, 2); Serial.print(',');
  Serial.print(vibration_now ? 1 : 0); Serial.print(',');
  Serial.print(movement_now ? 1 : 0); Serial.print(',');
  printFloatOrNA(accel_x_g, 4); Serial.print(',');
  printFloatOrNA(accel_y_g, 4); Serial.print(',');
  printFloatOrNA(accel_z_g, 4); Serial.print(',');
  Serial.print(last_alert ? 1 : 0);
}

static void emitSensorSnapshot() {
  Serial.print("SENSOR,"); Serial.print(millis()); Serial.print(',');
  emitSensorFields();
  Serial.println();
}

static void processLine(char *line) {
  // Protocol: INFER,<request_id>,<18 already-preprocessed float values>
  char *save = nullptr;
  char *token = strtok_r(line, ",", &save);
  if (!token || strcmp(token, "INFER") != 0) {
    emitError("BAD_COMMAND");
    return;
  }

  token = strtok_r(nullptr, ",", &save);
  if (!token) { emitError("MISSING_ID"); return; }
  char *id_end = nullptr;
  const unsigned long request_id = strtoul(token, &id_end, 10);
  if (id_end == token || *id_end != '\0') { emitError("BAD_ID"); return; }

  float features[TINYML_NUM_FEATURES];
  for (size_t i = 0; i < TINYML_NUM_FEATURES; ++i) {
    token = strtok_r(nullptr, ",", &save);
    if (!token) { emitError("FEATURE_COUNT"); return; }
    char *value_end = nullptr;
    const float value = strtof(token, &value_end);
    if (value_end == token || *value_end != '\0' || !isfinite(value)) { emitError("BAD_FEATURE"); return; }
    features[i] = value;
  }
  if (strtok_r(nullptr, ",", &save) != nullptr) { emitError("FEATURE_COUNT"); return; }

  readSensors(millis());
  const uint32_t start_us = micros();
  const int class_index = predict_tinyml_fixed_point(features);
  const uint32_t elapsed_us = micros() - start_us;
  if (class_index < 0 || class_index >= TINYML_NUM_CLASSES) { emitError("CLASS_INDEX"); return; }
  strncpy(last_prediction, TINYML_CLASS_NAMES[class_index], sizeof(last_prediction) - 1);
  last_prediction[sizeof(last_prediction) - 1] = '\0';

  Serial.print("RESULT,"); Serial.print(request_id); Serial.print(',');
  Serial.print(class_index); Serial.print(','); Serial.print(last_prediction); Serial.print(',');
  Serial.print(elapsed_us); Serial.print(','); Serial.print(millis()); Serial.print(',');
  emitSensorFields();
  Serial.println();
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  const uint32_t serial_wait_start = millis();
  while (!Serial && millis() - serial_wait_start < 1500) delay(10);

  pinMode(VIBRATION_PIN, INPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  setBuzzer(false);
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);

  oled_ready = display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDRESS);
  if (oled_ready) {
    display.clearDisplay(); display.setTextSize(1); display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0); display.println("AdaptiveML USB IDS"); display.println("Starting sensors..."); display.display();
  }
  dht.begin();
  mpu_ready = mpu6500.init();
  if (mpu_ready) {
    mpu6500.setAccRange(MPU6500_ACC_RANGE_8G);
    mpu6500.setGyrRange(MPU6500_GYRO_RANGE_500);
    mpu6500.enableAccDLPF(true);
    mpu6500.setAccDLPF(MPU6500_DLPF_6);
  }
  ina_ready = ina219.begin();
  if (ina_ready) ina219.setCalibration_32V_1A();
  calibrateVibrationSensor();
  readSensors(millis());
  updateDisplay();

  Serial.print("READY,SARA-ESP32-USB,features="); Serial.print(TINYML_NUM_FEATURES);
  Serial.print(",classes="); Serial.print(TINYML_NUM_CLASSES);
  Serial.print(",ina219="); Serial.print(ina_ready ? 1 : 0);
  Serial.print(",dht22="); Serial.print(isfinite(temperature_c) && isfinite(humidity_pct) ? 1 : 0);
  Serial.print(",mpu6500="); Serial.print(mpu_ready ? 1 : 0);
  Serial.print(",oled="); Serial.println(oled_ready ? 1 : 0);
}

void loop() {
  readSensors(millis());
  const uint32_t now = millis();
  if (now - last_usb_sensor_report_ms >= USB_SENSOR_REPORT_MS) {
    last_usb_sensor_report_ms = now;
    emitSensorSnapshot();
  }
  if (now - last_display_ms >= DISPLAY_SAMPLE_MS || last_display_ms == 0) {
    last_display_ms = now;
    updateDisplay();
  }

  while (Serial.available() > 0) {
    const char ch = static_cast<char>(Serial.read());
    if (ch == '\n') {
      input_line[input_length] = '\0';
      if (input_length > 0 && input_line[input_length - 1] == '\r') input_line[input_length - 1] = '\0';
      if (input_length > 0) processLine(input_line);
      input_length = 0;
    } else if (input_length + 1 < sizeof(input_line)) {
      input_line[input_length++] = ch;
    } else {
      input_length = 0;
      emitError("LINE_TOO_LONG");
    }
  }
  delay(5);
}
