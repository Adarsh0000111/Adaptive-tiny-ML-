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
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <DHT.h>
#include <time.h>

// ---- Include the exported TinyML model header --------------------------------
#include "esp32_model.h"
#include "secrets.h"

// -------------------- WiFi / MQTT settings -----------------------------------
const char *WIFI_SSID = WIFI_SSID_VALUE;
const char *WIFI_PASSWORD = WIFI_PASSWORD_VALUE;
const char *MQTT_BROKER = MQTT_BROKER_VALUE; // PC LAN IP running the MQTT broker
const uint16_t MQTT_PORT = 1883;
const char *DEVICE_ID = "esp32-ids-01";

const char *TOPIC_TELEMETRY = "esp32/esp32-ids-01/telemetry";
const char *TOPIC_INFERENCE = "esp32/esp32-ids-01/inference";
const char *TOPIC_COMMAND = "esp32/esp32-ids-01/command";

// -------------------- Hardware instances -------------------------------------
Adafruit_INA219 ina219;
Adafruit_MPU6050 mpu;
Adafruit_SSD1306 display(128, 64, &Wire, -1);
DHT dht(6, DHT22);
WiFiClient espClient;
PubSubClient mqtt(espClient);

const uint8_t SW420_PIN = 7;
const uint8_t BUZZER_PIN = 15;
float temperature_c = 25.0f;
float humidity_pct = 50.0f;
bool vibration_detected = false;
float accel_x = 0.0f, accel_y = 0.0f, accel_z = 0.0f;
bool display_ready = false;

// -------------------- Config -------------------------------------------------
const float V_MIN = 3.0;                  // empty battery voltage (V)
const float V_MAX = 4.2;                  // full battery voltage (V)
const uint32_t SAMPLE_INTERVAL_MS = 1000; // inference cadence

// Feature order MUST match the training preprocessing (see config.yaml / pipeline).
// Update these indexes if you change the feature set.
enum FeatureIdx
{
  F_FLOW_DURATION = 0,
  F_PACKET_RATE = 1,
  F_BYTE_RATE = 2,
  F_PAYLOAD_LEN = 3,
  F_TCP_FIN = 4,
  F_TCP_SYN = 5,
  F_TCP_ACK = 6,
  F_TCP_PSH = 7,
  F_MODBUS_FUNC = 8,
  F_MODBUS_REG = 9,
  F_HTTP_RES = 10,
  F_MQTT_TYPE = 11,
  F_FAILED_CONN = 12,
  F_BATTERY_PCT = 13,
  F_BUS_VOLTAGE = 14,
  F_SHUNT_MA = 15,
  F_TEMP_C = 16,
  F_HUM_PCT = 17,
  NUM_FEATURES = 18 // must match TINYML_NUM_FEATURES
};

// The trained model feature order is documented by edge-iiotset-18-v1.
// Environmental sensor readings are resource context, not classifier inputs.
enum FeatureIdx
{
  F_FLOW_DURATION = 0, F_PACKET_RATE = 1, F_BYTE_RATE = 2, F_PAYLOAD_LEN = 3,
  F_TCP_FIN = 4, F_TCP_SYN = 5, F_TCP_ACK = 6, F_TCP_PSH = 7,
  F_MODBUS_FUNC = 8, F_MODBUS_REG = 9, F_HTTP_RES = 10, F_MQTT_TYPE = 11,
  F_FAILED_CONN = 12, F_PROTOCOL_HTTP = 13, F_PROTOCOL_MODBUS = 14,
  F_PROTOCOL_MQTT = 15, F_PROTOCOL_TCP = 16, F_PROTOCOL_UDP = 17,
  NUM_FEATURES = 18
};

// -------------------- State --------------------------------------------------
unsigned long last_sample_ms = 0;
float battery_pct = 100.0;
String current_model_tier = "LIGHTWEIGHT"; // LIGHTWEIGHT | MEDIUM | HEAVY
double cumulative_energy_j = 0.0;
float previous_power_w = 0.0f;
unsigned long previous_energy_sample_ms = 0;
bool temperature_valid = false;
bool humidity_valid = false;

// -------------------- Helpers ------------------------------------------------
float read_battery_pct()
{
  float vbus = ina219.getBusVoltage_V(); // INA219 returns V
  float pct = (vbus - V_MIN) / (V_MAX - V_MIN) * 100.0;
  return constrain(pct, 0.0, 100.0);
}

void update_environmental_sensors()
{
  float t = dht.readTemperature();
  float h = dht.readHumidity();
  if (!isnan(t)) temperature_c = t;
  if (!isnan(h)) humidity_pct = h;
  temperature_valid = !isnan(t);
  humidity_valid = !isnan(h);
  vibration_detected = digitalRead(SW420_PIN) == HIGH;

  sensors_event_t accel, gyro, mpu_temp;
  mpu.getEvent(&accel, &gyro, &mpu_temp);
  accel_x = accel.acceleration.x;
  accel_y = accel.acceleration.y;
  accel_z = accel.acceleration.z;

}

void mqtt_callback(char *topic, byte *payload, unsigned int length)
{
  StaticJsonDocument<512> doc;
  DeserializationError err = deserializeJson(doc, payload, length);
  if (err)
    return;
  if (String(topic) == TOPIC_COMMAND)
  {
    if (doc.containsKey("model_tier"))
    {
      current_model_tier = doc["model_tier"].as<String>();
      Serial.printf("Command: switch model tier to %s\n", current_model_tier.c_str());
    }
    if (doc.containsKey("sample_rate"))
    {
      // adjust SAMPLE_INTERVAL_MS if needed
    }
  }
}

void publish_telemetry()
{
  StaticJsonDocument<512> doc;
  doc["device_id"] = DEVICE_ID;
  doc["timestamp_ms"] = millis();
  time_t epoch = time(nullptr);
  if (epoch > 1700000000)
  {
    struct tm utc_time;
    gmtime_r(&epoch, &utc_time);
    char iso_time[25];
    strftime(iso_time, sizeof(iso_time), "%Y-%m-%dT%H:%M:%SZ", &utc_time);
    doc["timestamp"] = iso_time;
  }
  doc["source"] = "physical";
  doc["hardware_status"] = "online";
  doc["quality"] = (temperature_valid && humidity_valid) ? "good" : "partial";
  doc["battery_pct"] = battery_pct;
  float voltage_v = ina219.getBusVoltage_V();
  float current_a = ina219.getCurrent_mA() / 1000.0f;
  doc["voltage_v"] = voltage_v;
  doc["current_a"] = current_a;
  doc["power_w"] = voltage_v * current_a;
  doc["energy_j"] = cumulative_energy_j;
  doc["bus_voltage_v"] = ina219.getBusVoltage_V(); // Backward-compatible UI fields.
  doc["shunt_current_ma"] = ina219.getCurrent_mA();
  if (temperature_valid) doc["temperature_c"] = temperature_c;
  else doc["temperature_c"] = nullptr;
  if (humidity_valid) doc["humidity_pct"] = humidity_pct;
  else doc["humidity_pct"] = nullptr;
  doc["temperature_kind"] = "ambient";
  doc["vibration"] = vibration_detected;
  doc["accel_x"] = accel_x;
  doc["accel_y"] = accel_y;
  doc["accel_z"] = accel_z;
  doc["model_tier"] = current_model_tier;
  char buf[512];
  serializeJson(doc, buf);
  mqtt.publish(TOPIC_TELEMETRY, buf);
}

void publish_inference(const char *prediction, float threat_score,
                       const char *severity, unsigned long latency_ms,
                       float energy_uj)
{
  StaticJsonDocument<256> doc;
  doc["device_id"] = DEVICE_ID;
  doc["prediction"] = prediction;
  doc["inference_valid"] = false;
  doc["reason"] = "Live traffic features are not currently extracted with the trained edge-iiotset-18-v1 definitions; sensor telemetry is not used as a substitute.";
  doc["threat_score"] = threat_score;
  doc["severity"] = severity;
  doc["latency_ms"] = (double)latency_ms;
  doc["energy_uj"] = nullptr;
  doc["model"] = current_model_tier;
  char buf[256];
  serializeJson(doc, buf);
  mqtt.publish(TOPIC_INFERENCE, buf);
}

// -------------------- Setup --------------------------------------------------
void setup()
{
  Serial.begin(115200);
  Wire.begin(4, 5); // SDA, SCL

  if (!ina219.begin())
  {
    Serial.println("INA219 not found! Check wiring.");
    while (1)
      delay(1000);
  }
  ina219.setCalibration_32V_1A();

  dht.begin();
  pinMode(SW420_PIN, INPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  if (!mpu.begin(0x70))
  {
    Serial.println("MPU6050 not found! Check wiring.");
    while (1) delay(1000);
  }

  display_ready = display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  if (display_ready)
  {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.println("Adaptive IDS");
    display.println("Starting...");
    display.display();
  }

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
  configTime(0, 0, "pool.ntp.org", "time.nist.gov");

  mqtt.setServer(MQTT_BROKER, MQTT_PORT);
  mqtt.setCallback(mqtt_callback);
  Serial.println("ESP32 TinyML IDS ready");
}

// -------------------- Loop ---------------------------------------------------
void loop()
{
  mqtt.loop();
  if (!mqtt.connected())
  {
    // simple reconnect
    if (mqtt.connect(DEVICE_ID))
    {
      mqtt.subscribe(TOPIC_COMMAND);
    }
  }

  unsigned long now = millis();
  if (now - last_sample_ms >= SAMPLE_INTERVAL_MS)
  {
    last_sample_ms = now;

    battery_pct = read_battery_pct();

    update_environmental_sensors();
    float voltage = ina219.getBusVoltage_V();
    float current_a = ina219.getCurrent_mA() / 1000.0f;
    float power_w = voltage * current_a;
    if (previous_energy_sample_ms != 0 && now > previous_energy_sample_ms)
    {
      float elapsed_s = (now - previous_energy_sample_ms) / 1000.0f;
      cumulative_energy_j += (max(0.0f, power_w) + max(0.0f, previous_power_w)) * 0.5f * elapsed_s;
    }
    previous_energy_sample_ms = now;
    previous_power_w = power_w;
    Serial.printf("[%s] batt=%.1f%% V=%.3f I=%.4fA P=%.3fW E=%.3fJ; live model inference unavailable until a compatible network feature extractor is installed.\n",
                  DEVICE_ID, battery_pct, voltage, current_a, power_w, cumulative_energy_j);
    publish_telemetry();
    publish_inference("UNAVAILABLE", 0.0f, "UNKNOWN", 0, 0.0f);

    if (display_ready)
    {
      display.clearDisplay();
      display.setCursor(0, 0);
      display.printf("Batt: %.1f%%\nTemp: %.1f C\nHum: %.1f%%\nVib: %s\nInference: n/a\n",
                     battery_pct, temperature_c, humidity_pct,
                     vibration_detected ? "YES" : "NO");
      display.display();
    }
    digitalWrite(BUZZER_PIN, vibration_detected ? HIGH : LOW);
  }
}
