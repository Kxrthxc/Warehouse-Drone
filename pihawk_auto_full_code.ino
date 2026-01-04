#include "esp_camera.h"
#include <WiFi.h>
#include <WiFiUdp.h>
#include "esp_http_server.h"
#include <HardwareSerial.h>
#include "driver/uart.h"

// ================== WIFI ==================
const char* ssid     = "Redmi Note 13";
const char* password = "12345678";

// ================== UDP ==================
WiFiUDP udp;
const uint16_t UDP_PORT = 14550;
// Packet format: 'R''C' + 4xU16 + seq + csum = 12 bytes

// ================== PINS ==================
#define LED_PIN        33
#define IBUS_RX_PIN    13
#define SBUS_TX_PIN    12

// ================== MODE SWITCH ==================
// If you use CH5 on radio -> index 4. If CH6 -> index 5.
#define MODE_CH_INDEX   4       // <<< SET THIS (4 for CH5, 5 for CH6)
#define MODE_THRESHOLD  1600
#define UDP_MAX_AGE_MS  200

// ================== iBUS ==================
#define IBUS_FRAME_LENGTH 32
#define IBUS_COMMAND40    0x40
#define IBUS_MAX_CHANNELS 14
HardwareSerial ibusRX(1);

// ================== SBUS ==================
HardwareSerial sbusTX(2);
static const uint32_t SBUS_BAUD = 100000;
static const uint8_t  SBUS_FRAME_LEN = 25;

// Pixhawk RCIN typically expects INVERTED SBUS
#define SBUS_INVERTED  1

static const int SBUS_MIN = 172;
static const int SBUS_MAX = 1811;
static const int SBUS_MID = 992;

// ================== CAMERA PINS (AI THINKER) ==================
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// ================== STATE ==================
uint8_t  ibusBuf[IBUS_FRAME_LENGTH];
uint8_t  ibusIdx = 0;

uint16_t rxCh[IBUS_MAX_CHANNELS]; // iBUS channels 1..14 (0..13)
uint16_t udpRoll=1500, udpPitch=1500, udpThrottle=1000, udpYaw=1500;

uint8_t  sbusFrame[SBUS_FRAME_LEN];

unsigned long lastIbusMs = 0;
unsigned long lastUdpMs  = 0;
unsigned long lastSbusMs = 0;

bool ledState = false;
unsigned long lastLedToggle = 0;

// ================== HELPERS ==================
static inline uint16_t clampU16(int v, int lo=1000, int hi=2000) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return (uint16_t)v;
}

static inline bool ibusChecksumOk(const uint8_t* f) {
  uint16_t csum = 0xFFFF;
  for (uint8_t i = 0; i < IBUS_FRAME_LENGTH - 2; i++) csum -= f[i];
  uint16_t rx = f[30] | (f[31] << 8);
  return (csum == rx);
}

static inline void parseIbusChannels(const uint8_t* f) {
  for (uint8_t i = 0; i < IBUS_MAX_CHANNELS; i++) {
    uint8_t idx = 2 + i * 2;
    rxCh[i] = f[idx] | (f[idx + 1] << 8);
  }
  lastIbusMs = millis();
}

static inline bool readIbusFrame() {
  while (ibusRX.available() > 0) {
    uint8_t v = ibusRX.read();

    if (ibusIdx == 0 && v != 0x20) continue;
    if (ibusIdx == 1 && v != IBUS_COMMAND40) { ibusIdx = 0; continue; }

    ibusBuf[ibusIdx++] = v;

    if (ibusIdx == IBUS_FRAME_LENGTH) {
      ibusIdx = 0;
      if (ibusChecksumOk(ibusBuf)) {
        parseIbusChannels(ibusBuf);
        return true;
      }
    }
  }
  return false;
}

// UDP packet: 'R''C' + 4xU16 + seq + checksum
static inline bool parseUdpPacket() {
  uint8_t b[32];
  int len = udp.read(b, sizeof(b));
  if (len != 12) return false;
  if (b[0] != 'R' || b[1] != 'C') return false;

  uint8_t csum = 0;
  for (int i = 0; i < 11; i++) csum += b[i];
  csum &= 0xFF;
  if (csum != b[11]) return false;

  uint16_t r = b[2] | (b[3] << 8);
  uint16_t p = b[4] | (b[5] << 8);
  uint16_t t = b[6] | (b[7] << 8);
  uint16_t y = b[8] | (b[9] << 8);

  udpRoll     = clampU16(r);
  udpPitch    = clampU16(p);
  udpThrottle = clampU16(t);
  udpYaw      = clampU16(y);

  lastUdpMs = millis();
  return true;
}

static inline bool udpFresh(unsigned long now) {
  return (now - lastUdpMs) <= UDP_MAX_AGE_MS;
}

static inline uint16_t ibusToSbus(uint16_t v) {
  v = clampU16(v, 1000, 2000);
  long out = (long)(v - 1000) * (SBUS_MAX - SBUS_MIN) / 1000 + SBUS_MIN;
  return (uint16_t)out;
}

// Pack 16 channels (11-bit) into SBUS frame
static void buildSbusFrame(const uint16_t* ch16, bool failsafe, bool frameLost) {
  sbusFrame[0] = 0x0F;
  for (int i = 1; i <= 22; i++) sbusFrame[i] = 0;

  uint32_t bitIndex = 0;
  for (int ch = 0; ch < 16; ch++) {
    uint16_t v = (uint16_t)(ch16[ch] & 0x07FF);
    for (int b = 0; b < 11; b++) {
      if (v & (1 << b)) {
        uint32_t byteIndex = 1 + (bitIndex >> 3);
        uint8_t  bitInByte = bitIndex & 7;
        sbusFrame[byteIndex] |= (1 << bitInByte);
      }
      bitIndex++;
    }
  }

  uint8_t flags = 0x00;
  if (frameLost) flags |= (1 << 2);
  if (failsafe)  flags |= (1 << 3);
  sbusFrame[23] = flags;
  sbusFrame[24] = 0x00;
}

// ================== CAMERA STREAM ==================
static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t * fb = NULL;
  esp_err_t res = httpd_resp_set_type(req, "multipart/x-mixed-replace; boundary=frame");
  if (res != ESP_OK) return res;

  char part_buf[64];

  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) return ESP_FAIL;

    size_t hlen = snprintf(part_buf, sizeof(part_buf),
      "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n",
      (unsigned)fb->len
    );

    res = httpd_resp_send_chunk(req, part_buf, hlen);
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char*)fb->buf, fb->len);
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, "\r\n", 2);

    esp_camera_fb_return(fb);
    if (res != ESP_OK) break;

    delay(0);
  }
  return res;
}

static void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.backlog_conn = 8;

  httpd_handle_t server = NULL;
  httpd_uri_t stream_uri = {
    .uri      = "/stream",
    .method   = HTTP_GET,
    .handler  = stream_handler,
    .user_ctx = NULL
  };

  if (httpd_start(&server, &config) == ESP_OK) {
    httpd_register_uri_handler(server, &stream_uri);
  }
}

void setup() {
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  Serial.begin(115200);
  delay(200);
  Serial.println("\nESP32-CAM: STREAM + iBUS->SBUS + UDP R/P/Y Override");

  for (int i = 0; i < IBUS_MAX_CHANNELS; i++) rxCh[i] = 1500;

  ibusRX.begin(115200, SERIAL_8N1, IBUS_RX_PIN, -1);

  sbusTX.begin(SBUS_BAUD, SERIAL_8E2, -1, SBUS_TX_PIN);
#if SBUS_INVERTED
  uart_set_line_inverse(UART_NUM_2, UART_SIGNAL_TXD_INV);
#endif

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(ssid, password);

  Serial.print("Connecting WiFi");
  int retry = 0;
  while (WiFi.status() != WL_CONNECTED && retry < 40) {
    delay(300);
    Serial.print(".");
    retry++;
  }
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\nWiFi failed, restarting");
    delay(200);
    ESP.restart();
  }
  Serial.println("\nWiFi OK");
  Serial.print("IP: "); Serial.println(WiFi.localIP());
  Serial.print("STREAM: http://"); Serial.print(WiFi.localIP()); Serial.println("/stream");

  udp.begin(UDP_PORT);
  Serial.print("UDP PORT: "); Serial.println(UDP_PORT);

  camera_config_t cfg;
  cfg.ledc_channel = LEDC_CHANNEL_0;
  cfg.ledc_timer   = LEDC_TIMER_0;
  cfg.pin_d0 = Y2_GPIO_NUM; cfg.pin_d1 = Y3_GPIO_NUM; cfg.pin_d2 = Y4_GPIO_NUM; cfg.pin_d3 = Y5_GPIO_NUM;
  cfg.pin_d4 = Y6_GPIO_NUM; cfg.pin_d5 = Y7_GPIO_NUM; cfg.pin_d6 = Y8_GPIO_NUM; cfg.pin_d7 = Y9_GPIO_NUM;
  cfg.pin_xclk = XCLK_GPIO_NUM; cfg.pin_pclk = PCLK_GPIO_NUM; cfg.pin_vsync = VSYNC_GPIO_NUM; cfg.pin_href = HREF_GPIO_NUM;
  cfg.pin_sscb_sda = SIOD_GPIO_NUM; cfg.pin_sscb_scl = SIOC_GPIO_NUM;
  cfg.pin_pwdn = PWDN_GPIO_NUM; cfg.pin_reset = RESET_GPIO_NUM;
  cfg.xclk_freq_hz = 10000000;
  cfg.pixel_format = PIXFORMAT_JPEG;
  cfg.frame_size   = FRAMESIZE_QQVGA;
  cfg.jpeg_quality = 15;
  cfg.fb_count     = 2;

  esp_err_t err = esp_camera_init(&cfg);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed 0x%x\n", err);
    delay(500);
    ESP.restart();
  }

  startCameraServer();
  Serial.println("Camera server started");
}

void loop() {
  readIbusFrame();

  if (udp.parsePacket() > 0) {
    parseUdpPacket();
  }

  unsigned long now = millis();

  bool ibusGood = (now - lastIbusMs) < 300;
  bool autoMode = ibusGood && (rxCh[MODE_CH_INDEX] > MODE_THRESHOLD) && udpFresh(now);

  uint16_t sbusCh[16];
  for (int i = 0; i < 16; i++) sbusCh[i] = SBUS_MID;

  bool failsafe  = !ibusGood;
  bool frameLost = !ibusGood;

  if (ibusGood) {
    for (int i = 0; i < 14; i++) sbusCh[i] = ibusToSbus(rxCh[i]);

    if (autoMode) {
      // override ONLY roll/pitch/yaw; keep throttle manual (radio)
      sbusCh[0] = ibusToSbus(udpRoll);   // CH1 Roll
      sbusCh[1] = ibusToSbus(udpPitch);  // CH2 Pitch
      sbusCh[3] = ibusToSbus(udpYaw);    // CH4 Yaw
    }
  } else {
    sbusCh[0] = SBUS_MID;
    sbusCh[1] = SBUS_MID;
    sbusCh[2] = SBUS_MIN;
    sbusCh[3] = SBUS_MID;
  }

  if (now - lastSbusMs >= 10) {
    lastSbusMs = now;
    buildSbusFrame(sbusCh, failsafe, frameLost);
    sbusTX.write(sbusFrame, SBUS_FRAME_LEN);
  }

  // LED heartbeat
  if (ibusGood && (now - lastLedToggle >= 150)) {
    lastLedToggle = now;
    ledState = !ledState;
    digitalWrite(LED_PIN, ledState);
  }
  if (!ibusGood) digitalWrite(LED_PIN, LOW);

  delay(0);
}
