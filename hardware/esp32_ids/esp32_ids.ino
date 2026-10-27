/*
 * Adaptive TinyML IIoT Intrusion Detection - ESP32 Firmware
 * ----------------------------------------------------------
 * Reads battery energy via INA219 (I2C), runs the exported TinyML
 * quantized decision-tree model, and publishes telemetry + inference
 * results to the Python dashboard over MQTT (WiFi).
 *
 * Hardware wiring (ESP32 DevKit -> INA219):
 *   3.3V   -> VCC
 *   GND    -> GND
 *   GPIO22 -> SCL
 *   GPIO21 -> SDA
 *
 * Optional sensors:
 *   GPIO4  -> DHT22 data (temperature / humidity)
 *   GPIO5  -> PIR motion
 *   GPIO34 -> MQ-2 analog (gas/smoke)
 *
 * Libraries (Arduino Library Manager):
 *   - Adafruit INA219 (by Adafruit)
 *   - PubSubClient (by Nick O'Leary)   -> MQTT
 *   - ArduinoJson (by Benoit Blanchon) -> JSON payloads
 *   - DHT sensor library (by Adafruit) -> optional DHT22
 */

#include <WiFi.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <Adafruit_INA219.h>
#include <ArduinoJson.h>

// ---- Include the exported TinyML model header --------------------------------
#include "esp32_model.h"

// -------------------- WiFi / MQTT settings -----------------------------------
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* MQTT_BROKER   = "192.168.1.100";   // PC running the Streamlit dashboard
const uint16_t MQTT_PORT  = 1883;
const char* DEVICE_ID     = "esp32-ids-01";

const char* TOPIC_TELEMETRY = "esp32/esp32-ids-01/telemetry";
const char* TOPIC_INFERENCE = "esp32/esp32-ids-01/inference";
const char* TOPIC_COMMAND   = "esp32/esp32-ids-01/command";

// -------------------- Hardware instances -------------------------------------
Adafruit_INA219 ina219;
WiFiClient espClient;
PubSubClient mqtt(espClient);

// -------------------- Config -------------------------------------------------
const float V_MIN = 3.0;   // empty battery voltage (V)
const float V_MAX = 4.2;   // full battery voltage (V)
const uint32_t SAMPLE_INTERVAL_MS = 1000; // inference cadence

// Feature order MUST match the training preprocessing (see config.yaml / pipeline).
// Update these indexes if you change the feature set.
enum FeatureIdx {
  F_FLOW_DURATION = 0,
  F_PACKET_RATE   = 1,
  F_BYTE_RATE     = 2,
  F_PAYLOAD_LEN   = 3,
  F_TCP_FIN       = 4,
  F_TCP_SYN       = 5,
  F_TCP_ACK       = 6,
  F_TCP_PSH       = 7,
  F_MODBUS_FUNC   = 8,
  F_MODBUS_REG    = 9,
  F_HTTP_RES      = 10,
  F_MQTT_TYPE     = 11,
  F_FAILED_CONN   = 12,
  F_BATTERY_PCT   = 13,
  F_BUS_VOLTAGE   = 14,
  F_SHUNT_MA      = 15,
  F_TEMP_C        = 16,
  F_HUM_PCT       = 17,
  NUM_FEATURES    = 18   // must match TINYML_NUM_FEATURES
};

// Class labels (must match label_encoder.classes_ order from training)
const char* CLASS_NAMES[TINYML_NUM_CLASSES] = {
  "Normal", "Scanning", "PortScan", "SQL_Injection", "Backdoor", "DDoS"
};

// -------------------- State --------------------------------------------------
unsigned long last_sample_ms = 0;
float battery_pct = 100.0;
String current_model_tier = "LIGHTWEIGHT"; // LIGHTWEIGHT | MEDIUM | HEAVY

// -------------------- Helpers ------------------------------------------------
float read_battery_pct() {
  float vbus = ina219.getBusVoltage_V(); // INA219 returns V
  float pct = (vbus - V_MIN) / (V_MAX - V_MIN) * 100.0;
  return constrain(pct, 0.0, 100.0);
}

void build_features(float* features) {
  // NOTE: In a real deployment you would read the actual IIoT network features
  // from your packet capture / Modbus / MQTT / TCP streamer. Here we read the
  // energy/environment sensors and placeholder traffic features for illustration.
  features[F_FLOW_DURATION] = 0.0f;   // populate from packet parser
  features[F_PACKET_RATE]   = 0.0f;
  features[F_BYTE_RATE]     = 0.0f;
  features[F_PAYLOAD_LEN]   = 0.0f;
  features[F_TCP_FIN]       = 0.0f;
  features[F_TCP_SYN]       = 0.0f;
  features[F_TCP_ACK]       = 0.0f;
  features[F_TCP_PSH]       = 0.0f;
  features[F_MODBUS_FUNC]   = 0.0f;
  features[F_MODBUS_REG]    = 0.0f;
  features[F_HTTP_RES]      = 0.0f;
  features[F_MQTT_TYPE]     = 0.0f;
  features[F_FAILED_CONN]   = 0.0f;

  features[F_BATTERY_PCT]   = battery_pct;
  features[F_BUS_VOLTAGE]   = ina219.getBusVoltage_V();
  features[F_SHUNT_MA]      = ina219.getCurrent_mA();

  // Optional: DHT22 / other sensors -> fill F_TEMP_C, F_HUM_PCT
  features[F_TEMP_C]        = 25.0f;
  features[F_HUM_PCT]       = 50.0f;

  // IMPORTANT: features must be scaled with the SAME StandardScaler used in
  // training before calling predict_tinyml_int8. Persist the scaler mean/scale
  // (e.g. in a second header) and apply: x_scaled = (x_raw - mean) / scale.
}

void mqtt_callback(char* topic, byte* payload, unsigned int length) {
  StaticJsonDocument<256> doc;
  DeserializationError err = deserializeJson(doc, payload, length);
  if (err) return;
  if (String(topic) == TOPIC_COMMAND) {
    if (doc.containsKey("model_tier")) {
      current_model_tier = doc["model_tier"].as<String>();
      Serial.printf("Command: switch model tier to %s\n", current_model_tier.c_str());
    }
    if (doc.containsKey("sample_rate")) {
      // adjust SAMPLE_INTERVAL_MS if needed
    }
  }
}

void publish_telemetry() {
  StaticJsonDocument<256> doc;
  doc["device_id"] = DEVICE_ID;
  doc["battery_pct"] = battery_pct;
  doc["bus_voltage_v"] = ina219.getBusVoltage_V();
  doc["shunt_current_ma"] = ina219.getCurrent_mA();
  doc["model_tier"] = current_model_tier;
  char buf[256];
  serializeJson(doc, buf);
  mqtt.publish(TOPIC_TELEMETRY, buf);
}

void publish_inference(const char* prediction, float threat_score,
                       const char* severity, unsigned long latency_ms,
                       float energy_uj) {
  StaticJsonDocument<256> doc;
  doc["device_id"] = DEVICE_ID;
  doc["prediction"] = prediction;
  doc["threat_score"] = threat_score;
  doc["severity"] = severity;
  doc["latency_ms"] = (double)latency_ms;
  doc["energy_uj"] = energy_uj;
  doc["model"] = current_model_tier;
  char buf[256];
  serializeJson(doc, buf);
  mqtt.publish(TOPIC_INFERENCE, buf);
}

// -------------------- Setup --------------------------------------------------
void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22); // SDA, SCL

  if (!ina219.begin()) {
    Serial.println("INA219 not found! Check wiring.");
    while (1) delay(1000);
  }
  ina219.setCalibration_32V_1A();

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");

  mqtt.setServer(MQTT_BROKER, MQTT_PORT);
  mqtt.setCallback(mqtt_callback);
  Serial.println("ESP32 TinyML IDS ready");
}

// -------------------- Loop ---------------------------------------------------
void loop() {
  mqtt.loop();
  if (!mqtt.connected()) {
    // simple reconnect
    if (mqtt.connect(DEVICE_ID)) {
      mqtt.subscribe(TOPIC_COMMAND);
    }
  }

  unsigned long now = millis();
  if (now - last_sample_ms >= SAMPLE_INTERVAL_MS) {
    last_sample_ms = now;

    battery_pct = read_battery_pct();

    float features[NUM_FEATURES] = {0};
    build_features(features);

    // --- TinyML inference (quantized decision tree) ------------------------
    unsigned long inf_start = micros();
    int leaf = predict_tinyml_int8(features);
    unsigned long latency_us = micros() - inf_start;
    float latency_ms = latency_us / 1000.0f;

    // Map leaf index to class (training-specific; adjust to your encoder order)
    int pred_class_idx = leaf % TINYML_NUM_CLASSES;
    const char* prediction = CLASS_NAMES[pred_class_idx];

    // Simple threat score (mirror Python ThreatScorer for dashboard parity)
    float threat_score = 0.0f;
    if (strcmp(prediction, "Normal") != 0) threat_score = 60.0f;
    if (battery_pct < 15.0f) threat_score += 20.0f;
    threat_score = constrain(threat_score, 0.0f, 100.0f);

    const char* severity =
      threat_score >= 80 ? "CRITICAL" :
      threat_score >= 60 ? "HIGH" :
      threat_score >= 30 ? "MEDIUM" : "LOW";

    float energy_uj = 8.5f; // lightweight estimate (uJ); adjust per tier

    Serial.printf("[%s] batt=%.1f%% pred=%s threat=%.0f sev=%s latency=%.2fms\n",
                  DEVICE_ID, battery_pct, prediction, threat_score, severity, latency_ms);

    publish_telemetry();
    publish_inference(prediction, threat_score, severity, (unsigned long)latency_ms, energy_uj);
  }
}
