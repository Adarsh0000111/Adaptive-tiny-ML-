#include <WiFi.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <DHT.h>
#include <MPU6500_WE.h>
#include <ArduinoJson.h>
#include <math.h>

const char *WIFI_SSID = "moto";
const char *WIFI_PASSWORD = "Hanu@123";
const char *MQTT_BROKER = "10.238.105.90";
const uint16_t MQTT_PORT = 1883;
const char *DEVICE_ID = "esp32-ids-01";
const char *TOPIC_TELEMETRY = "esp32/esp32-ids-01/telemetry";
const char *TOPIC_INFERENCE = "esp32/esp32-ids-01/inference";

#define SDA_PIN 4
#define SCL_PIN 5
#define DHT_PIN 6
#define VIBRATION_PIN 7
#define BUZZER_PIN 15
#define DHT_TYPE DHT22
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_ADDRESS 0x3C

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);
DHT dht(DHT_PIN, DHT_TYPE);
MPU6500_WE mpu6500 = MPU6500_WE(0x68);
WiFiClient espClient;
PubSubClient mqtt(espClient);

const float TEMP_LIMIT_C = 30.0;
const float MOVEMENT_THRESHOLD_G = 0.15;
const unsigned long MOVEMENT_HOLD_MS = 1000;
const unsigned long VIBRATION_HOLD_MS = 1200;
const unsigned long SAMPLE_INTERVAL_MS = 1000;

bool oledReady = false;
bool mpuReady = false;
int vibrationIdleLevel = HIGH;
bool vibrationNow = false;
unsigned long vibrationLastSeen = 0;
bool havePreviousAcceleration = false;
float previousX = 0, previousY = 0, previousZ = 0;
bool movementNow = false;
unsigned long movementLastSeen = 0;
float temperature = NAN, humidity = NAN;
float ax = 0, ay = 0, az = 0;
unsigned long lastDHTRead = 0, lastSample = 0;

void setBuzzer(bool on) { digitalWrite(BUZZER_PIN, on ? HIGH : LOW); }

void calibrateVibrationSensor() {
  Serial.println("Keep SW-420 still...");
  delay(500);
  int lowCount = 0, highCount = 0;
  for (int i = 0; i < 100; i++) {
    if (digitalRead(VIBRATION_PIN) == LOW) lowCount++;
    else highCount++;
    delay(10);
  }
  vibrationIdleLevel = (highCount >= lowCount) ? HIGH : LOW;
  Serial.printf("SW-420 idle level: %s\n", vibrationIdleLevel == HIGH ? "HIGH" : "LOW");
}

void connectWiFi() {
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.printf("\nWiFi connected: %s\n", WiFi.localIP().toString().c_str());
}

void connectMQTT() {
  while (!mqtt.connected()) {
    Serial.print("Connecting MQTT...");
    if (mqtt.connect(DEVICE_ID)) Serial.println(" connected");
    else { Serial.printf(" failed, rc=%d\n", mqtt.state()); delay(2000); }
  }
}

void publishTelemetry(bool alert, bool temperatureAlert) {
  StaticJsonDocument<768> doc;
  doc["device_id"] = DEVICE_ID;
  if (isnan(temperature)) doc["temperature_c"] = nullptr; else doc["temperature_c"] = temperature;
  if (isnan(humidity)) doc["humidity_pct"] = nullptr; else doc["humidity_pct"] = humidity;
  doc["vibration"] = vibrationNow;
  doc["movement"] = movementNow;
  doc["accel_x_g"] = ax;
  doc["accel_y_g"] = ay;
  doc["accel_z_g"] = az;
  doc["temperature_alert"] = temperatureAlert;
  doc["alert"] = alert;
  char payload[768];
  serializeJson(doc, payload, sizeof(payload));
  mqtt.publish(TOPIC_TELEMETRY, payload);
}

void publishInference(bool alert, bool temperatureAlert) {
  const char *prediction = temperatureAlert ? "TemperatureAlert" : vibrationNow ? "Vibration" : movementNow ? "Movement" : "Normal";
  StaticJsonDocument<384> doc;
  doc["device_id"] = DEVICE_ID;
  doc["prediction"] = prediction;
  doc["threat_score"] = alert ? 60 : 0;
  doc["severity"] = alert ? "HIGH" : "LOW";
  doc["model"] = "SENSOR_RULES";
  char payload[384];
  serializeJson(doc, payload, sizeof(payload));
  mqtt.publish(TOPIC_INFERENCE, payload);
}

void updateDisplay(bool alert) {
  if (!oledReady) return;
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println("INDUSTRIAL MONITOR");
  display.println("------------------");
  display.print("Temp: "); if (isnan(temperature)) display.println("-- C"); else { display.print(temperature, 1); display.println(" C"); }
  display.print("Humidity: "); if (isnan(humidity)) display.println("-- %"); else { display.print(humidity, 1); display.println(" %"); }
  display.print("Vibration: "); display.println(vibrationNow ? "DETECTED" : "Normal");
  display.print("Movement: "); display.println(movementNow ? "DETECTED" : "Stable");
  display.print("Alert: "); display.println(alert ? "ACTIVE!" : "Normal");
  display.display();
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  pinMode(VIBRATION_PIN, INPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  setBuzzer(false);
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);

  oledReady = display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDRESS);
  if (oledReady) {
    display.clearDisplay(); display.setTextSize(1); display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0); display.println("Industrial Monitor"); display.println("Starting..."); display.display();
  }
  dht.begin();
  mpuReady = mpu6500.init();
  if (mpuReady) {
    mpu6500.setAccRange(MPU6500_ACC_RANGE_8G);
    mpu6500.setGyrRange(MPU6500_GYRO_RANGE_500);
    mpu6500.enableAccDLPF(true);
    mpu6500.setAccDLPF(MPU6500_DLPF_6);
    Serial.println("MPU-6500 ready");
  } else Serial.println("MPU-6500 initialization failed");
  calibrateVibrationSensor();
  connectWiFi();
  mqtt.setServer(MQTT_BROKER, MQTT_PORT);
  Serial.println("All sensors initialized.");
}

void loop() {
  if (!mqtt.connected()) connectMQTT();
  mqtt.loop();
  unsigned long now = millis();

  if (now - lastDHTRead >= 2000 || lastDHTRead == 0) {
    lastDHTRead = now;
    temperature = dht.readTemperature();
    humidity = dht.readHumidity();
  }

  int vibrationRaw = digitalRead(VIBRATION_PIN);
  if (vibrationRaw != vibrationIdleLevel) vibrationLastSeen = now;
  vibrationNow = vibrationLastSeen != 0 && now - vibrationLastSeen <= VIBRATION_HOLD_MS;

  if (mpuReady) {
    xyzFloat acc = mpu6500.getGValues();
    ax = acc.x; ay = acc.y; az = acc.z;
    if (havePreviousAcceleration) {
      float dx = ax - previousX, dy = ay - previousY, dz = az - previousZ;
      if (sqrt(dx * dx + dy * dy + dz * dz) > MOVEMENT_THRESHOLD_G) movementLastSeen = now;
    }
    previousX = ax; previousY = ay; previousZ = az;
    havePreviousAcceleration = true;
  }
  movementNow = mpuReady && movementLastSeen != 0 && now - movementLastSeen <= MOVEMENT_HOLD_MS;

  bool temperatureAlert = !isnan(temperature) && temperature > TEMP_LIMIT_C;
  bool alert = vibrationNow || temperatureAlert;
  setBuzzer(alert);

  if (now - lastSample >= SAMPLE_INTERVAL_MS) {
    lastSample = now;
    publishTelemetry(alert, temperatureAlert);
    publishInference(alert, temperatureAlert);
    updateDisplay(alert);
    Serial.printf("Temp=%s Hum=%s Vib=%s Move=%s Alert=%s\n",
      isnan(temperature) ? "ERR" : String(temperature, 1).c_str(),
      isnan(humidity) ? "ERR" : String(humidity, 1).c_str(),
      vibrationNow ? "YES" : "NO", movementNow ? "YES" : "NO", alert ? "YES" : "NO");
  }
  delay(100);
}
