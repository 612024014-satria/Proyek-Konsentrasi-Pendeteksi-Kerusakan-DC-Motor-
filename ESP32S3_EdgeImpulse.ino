#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <a612024014-satria-project-1_inferencing.h>

// ---------- Pin / hardware settings ----------
static const int MPU_SDA_PIN = 8;
static const int MPU_SCL_PIN = 9;
static const int HALL_PIN = 4;
static const int OLED_SDA_PIN = 17;   
static const int OLED_SCL_PIN = 18;
static const uint8_t MPU_ADDR = 0x68;
static const uint8_t OLED_ADDR = 0x3C;
static const uint8_t PULSES_PER_REV = 1; 


static const int OLED_WIDTH = 128;
static const int OLED_HEIGHT = 64;
Adafruit_SSD1306 display(OLED_WIDTH, OLED_HEIGHT, &Wire1, -1);

// ---------- Sampling / inference ----------
static const uint32_t SAMPLE_INTERVAL_US = 10000; // 100 Hz
static const uint32_t RPM_INTERVAL_MS = 250;
static const size_t WINDOW_SAMPLES = EI_CLASSIFIER_RAW_SAMPLE_COUNT; // 100
static float sampleBuffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE]; // 100 x 4

volatile uint32_t hallPulses = 0;
uint32_t lastRpmUpdateMs = 0;
uint32_t lastSampleUs = 0;
uint32_t lastOledMs = 0;
float currentRPM = 0.0f;
float axG = 0, ayG = 0, azG = 0;
size_t sampleIndex = 0;

String predictedLabel = "MENUNGGU";
float predictedConfidence = 0.0f;
String inferenceStatus = "Mengumpulkan data";

void IRAM_ATTR onHallPulse() {
  hallPulses++;
}

// I2C register helpers for MPU6050; no MPU library required.
bool writeMPURegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

bool readMPUBytes(uint8_t startReg, uint8_t *data, size_t length) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(startReg);
  if (Wire.endTransmission(false) != 0) return false;
  size_t received = Wire.requestFrom((int)MPU_ADDR, (int)length, (int)true);
  if (received != length) return false;
  for (size_t i = 0; i < length; i++) data[i] = Wire.read();
  return true;
}

bool initMPU6050() {
  // Wake sensor, select ±2 g range, and disable its internal sleep mode.
  if (!writeMPURegister(0x6B, 0x00)) return false;
  delay(100);
  if (!writeMPURegister(0x1C, 0x00)) return false; // ±2 g, 16384 LSB/g
  if (!writeMPURegister(0x1A, 0x03)) return false; // DLPF setting
  return true;
}

bool readAcceleration(float &x, float &y, float &z) {
  uint8_t b[6];
  if (!readMPUBytes(0x3B, b, sizeof(b))) return false;
  int16_t rawX = (int16_t)((b[0] << 8) | b[1]);
  int16_t rawY = (int16_t)((b[2] << 8) | b[3]);
  int16_t rawZ = (int16_t)((b[4] << 8) | b[5]);

  // Convert to g. MUST match the units used in the training CSV.
  x = (float)rawX / 16384.0f;
  y = (float)rawY / 16384.0f;
  z = (float)rawZ / 16384.0f;
  return true;
}

void updateRPM() {
  uint32_t now = millis();
  uint32_t elapsed = now - lastRpmUpdateMs;
  if (elapsed < RPM_INTERVAL_MS) return;

  noInterrupts();
  uint32_t pulses = hallPulses;
  hallPulses = 0;
  interrupts();

  currentRPM = (elapsed > 0)
      ? ((float)pulses * 60000.0f / ((float)elapsed * (float)PULSES_PER_REV))
      : 0.0f;
  lastRpmUpdateMs = now;
}

int getTopClass(const ei_impulse_result_t &result, float &confidence) {
  int top = 0;
  confidence = 0.0f;
  for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
    float value = result.classification[i].value;
    if (value > confidence || i == 0) {
      confidence = value;
      top = (int)i;
    }
  }
  return top;
}

int eiGetSignalData(size_t offset, size_t length, float *out_ptr) {
  memcpy(out_ptr, sampleBuffer + offset, length * sizeof(float));
  return 0;
}

void runInference() {
  signal_t signal;
  signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
  signal.get_data = &eiGetSignalData;

  ei_impulse_result_t result = { 0 };
  EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false);
  if (err != EI_IMPULSE_OK) {
    inferenceStatus = "Inferensi gagal";
    Serial.print("run_classifier error: ");
    Serial.println((int)err);
    return;
  }

  int best = getTopClass(result, predictedConfidence);
  predictedLabel = result.classification[best].label;
  inferenceStatus = "Inferensi OK";

  Serial.printf("Prediksi: %s (%.1f%%) | RPM: %.0f | Ax: %.3f Ay: %.3f Az: %.3f\n",
                predictedLabel.c_str(), predictedConfidence * 100.0f,
                currentRPM, axG, ayG, azG);
#if EI_CLASSIFIER_HAS_ANOMALY == 1
  Serial.printf("Anomaly: %.3f\n", result.anomaly);
#endif
}

void drawOLED() {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println(F("MONITOR MOTOR DC"));
  display.drawLine(0, 10, 127, 10, SSD1306_WHITE);

  display.setCursor(0, 14);
  display.print(F("Status: "));
  display.println(predictedLabel);

  display.setCursor(0, 26);
  display.print(F("Conf : "));
  display.print(predictedConfidence * 100.0f, 1);
  display.println(F("%"));

  display.setCursor(0, 38);
  display.print(F("RPM  : "));
  display.println(currentRPM, 0);

  display.setCursor(0, 50);
  display.print(F("Ax:")); display.print(axG, 2);
  display.print(F(" Ay:")); display.print(ayG, 2);
  display.print(F(" Az:")); display.print(azG, 2);
  display.display();
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println();
  Serial.println(F("ESP32-S3 Edge Impulse Motor Monitor"));

  Wire.begin(MPU_SDA_PIN, MPU_SCL_PIN);
  Wire.setClock(400000);
  
  Wire1.begin(OLED_SDA_PIN, OLED_SCL_PIN);
  Wire1.setClock(400000);

  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println(F("OLED tidak terdeteksi. Periksa alamat 0x3C dan wiring."));
  } else {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.println(F("Motor Monitor"));
    display.println(F("Memulai sensor..."));
    display.display();
  }

  if (!initMPU6050()) {
    Serial.println(F("MPU6050 tidak terdeteksi di alamat 0x68."));
    inferenceStatus = "MPU ERROR";
    predictedLabel = "SENSOR ERROR";
    drawOLED();
    while (true) delay(1000);
  }
  Serial.println(F("MPU6050 terdeteksi."));

  pinMode(HALL_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(HALL_PIN), onHallPulse, FALLING);

  lastRpmUpdateMs = millis();
  lastSampleUs = micros();
  Serial.printf("Model input: %d samples, %d fitur/sampel, %d Hz\n",
                EI_CLASSIFIER_RAW_SAMPLE_COUNT,
                EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME,
                EI_CLASSIFIER_FREQUENCY);
  Serial.println(F("Mulai pengumpulan window 1 detik..."));
}

void loop() {
  updateRPM();

  uint32_t nowUs = micros();
  if ((uint32_t)(nowUs - lastSampleUs) < SAMPLE_INTERVAL_US) {
    if (millis() - lastOledMs >= 250) {
      drawOLED();
      lastOledMs = millis();
    }
    return;
  }
  lastSampleUs += SAMPLE_INTERVAL_US;

  if (!readAcceleration(axG, ayG, azG)) {
    Serial.println(F("Gagal membaca MPU6050."));
    return;
  }

  // Exact feature order from the exported model: Ax, Ay, Az, rpm.
  size_t base = sampleIndex * EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME;
  sampleBuffer[base + 0] = axG;
  sampleBuffer[base + 1] = ayG;
  sampleBuffer[base + 2] = azG;
  sampleBuffer[base + 3] = currentRPM;

  sampleIndex++;
  if (sampleIndex >= WINDOW_SAMPLES) {
    inferenceStatus = "Memproses...";
    runInference();
    sampleIndex = 0;
  }

  if (millis() - lastOledMs >= 250) {
    drawOLED();
    lastOledMs = millis();
  }
}
