// USB (UART0) と STS サーボバス (UART1, GPIO18=RX, GPIO19=TX) を素通しするだけの
// 透過ブリッジ。純正 Waveshare デモの「Start Serial Forwarding」相当を
// Wi-Fi なしで常時有効にしたもの。
//
// 対象基板: Waveshare "Servo Driver with ESP32" (ESP32-D0WD-V3)
// USB 側 921600bps / サーボバス側 1Mbps のレート変換ブリッジ。
// macOS の CP2102 ドライバは 1Mbps を安定に扱えない個体があるため、USB は
// 921600 に固定 (AhaRobot 本家ファームと同じレート)。
// ホスト側ツールは --baud 921600 で開くこと。

static const int S_RXD = 18;
static const int S_TXD = 19;
static const uint32_t USB_BAUD = 921600;
static const uint32_t BUS_BAUD = 1000000;

// 内蔵 LED (基板によっては GPIO2 が LED)。ブリッジ動作の目印用にゆっくり点滅
static const int LED_PIN = 2;

void setup() {
  // setRxBufferSize は begin() より先に呼ばないと反映されない
  Serial.setRxBufferSize(1024);
  Serial1.setRxBufferSize(1024);
  Serial.begin(USB_BAUD);
  Serial1.begin(BUS_BAUD, SERIAL_8N1, S_RXD, S_TXD);
  pinMode(LED_PIN, OUTPUT);
}

void loop() {
  // どちらも 128 バイト内蔵 FIFO があるので、フラッシュせずに 1 バイトずつ渡すだけで十分
  while (Serial.available()) {
    Serial1.write((uint8_t)Serial.read());
  }
  while (Serial1.available()) {
    Serial.write((uint8_t)Serial1.read());
  }
  // 目視用の心拍。1 秒周期
  static uint32_t last = 0;
  uint32_t now = millis();
  if (now - last >= 500) {
    last = now;
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
  }
}
