#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <esp_system.h>

// ==================== Hardware ====================

constexpr uint8_t SDA_PIN = 21;
constexpr uint8_t SCL_PIN = 22;
constexpr uint8_t OLED_ADDRESS = 0x3C;

constexpr int SCREEN_WIDTH = 128;
constexpr int SCREEN_HEIGHT = 64;
constexpr int OLED_RESET = -1;

Adafruit_SSD1306 display(
  SCREEN_WIDTH,
  SCREEN_HEIGHT,
  &Wire,
  OLED_RESET,
  400000UL,  // I2C speed during display transfers
  400000UL   // I2C speed after display transfers
);

bool displayReady = false;

// ==================== Animation settings ====================

// A full SSD1306 frame takes roughly 23+ ms at 400 kHz I2C.
// 30 FPS is a more realistic target than 60 FPS.
constexpr uint32_t FRAME_INTERVAL_MS = 33;

constexpr int EYE_WIDTH = 28;
constexpr int EYE_HEIGHT = 34;
constexpr int EYE_RADIUS = 7;

constexpr int LEFT_CENTER_X = 40;
constexpr int RIGHT_CENTER_X = 88;
constexpr int EYE_CENTER_Y = 32;

enum class Expression : uint8_t {
  IDLE,
  LOOK_LEFT,
  LOOK_RIGHT,
  ANGRY,
  HAPPY,
  SLEEP,
  HEARTS
};

Expression currentExpression = Expression::IDLE;

uint32_t stateStartedAt = 0;
uint32_t stateDuration = 3000;
uint32_t lastFrameAt = 0;

float gazeX = 0.0f;

// ==================== Blink state ====================

bool blinking = false;
uint32_t blinkStartedAt = 0;
uint32_t blinkWaitStartedAt = 0;
uint32_t blinkWaitDuration = 2500;

float blinkClosure = 0.0f;

constexpr uint32_t BLINK_CLOSE_MS = 90;
constexpr uint32_t BLINK_HOLD_MS = 40;
constexpr uint32_t BLINK_OPEN_MS = 120;

constexpr uint32_t BLINK_TOTAL_MS =
  BLINK_CLOSE_MS + BLINK_HOLD_MS + BLINK_OPEN_MS;

bool canBlink() {
  return currentExpression != Expression::SLEEP &&
         currentExpression != Expression::HEARTS;
}

void scheduleBlink(uint32_t now) {
  blinkWaitStartedAt = now;
  blinkWaitDuration = (uint32_t)random(1800, 5001);
}

void updateBlink(uint32_t now) {
  if (!canBlink()) {
    blinking = false;
    blinkClosure = 0.0f;
    return;
  }

  if (!blinking &&
      (uint32_t)(now - blinkWaitStartedAt) >= blinkWaitDuration) {
    blinking = true;
    blinkStartedAt = now;
  }

  if (!blinking) {
    blinkClosure = 0.0f;
    return;
  }

  const uint32_t elapsed = now - blinkStartedAt;

  if (elapsed < BLINK_CLOSE_MS) {
    blinkClosure = (float)elapsed / BLINK_CLOSE_MS;
  } else if (elapsed < BLINK_CLOSE_MS + BLINK_HOLD_MS) {
    blinkClosure = 1.0f;
  } else if (elapsed < BLINK_TOTAL_MS) {
    const uint32_t openingElapsed =
      elapsed - BLINK_CLOSE_MS - BLINK_HOLD_MS;

    blinkClosure =
      1.0f - (float)openingElapsed / BLINK_OPEN_MS;
  } else {
    blinking = false;
    blinkClosure = 0.0f;
    scheduleBlink(now);
  }
}

// ==================== Expression state machine ====================

Expression pickExpression() {
  const int roll = random(0, 100);

  if (roll < 35) return Expression::IDLE;        // 35%
  if (roll < 48) return Expression::LOOK_LEFT;   // 13%
  if (roll < 61) return Expression::LOOK_RIGHT;  // 13%
  if (roll < 72) return Expression::HAPPY;       // 11%
  if (roll < 82) return Expression::ANGRY;       // 10%
  if (roll < 91) return Expression::HEARTS;      //  9%
  return Expression::SLEEP;                     //  9%
}

uint32_t durationFor(Expression expression) {
  switch (expression) {
    case Expression::IDLE:
      return (uint32_t)random(3000, 7001);

    case Expression::LOOK_LEFT:
    case Expression::LOOK_RIGHT:
      return (uint32_t)random(1500, 2801);

    case Expression::ANGRY:
      return (uint32_t)random(1800, 3201);

    case Expression::HAPPY:
    case Expression::HEARTS:
      return (uint32_t)random(2200, 4201);

    case Expression::SLEEP:
      return (uint32_t)random(4000, 7001);
  }

  return 3000;
}

void enterExpression(Expression next, uint32_t now) {
  currentExpression = next;
  stateStartedAt = now;
  stateDuration = durationFor(next);

  blinking = false;
  blinkClosure = 0.0f;
  scheduleBlink(now);
}

void updateExpression(uint32_t now) {
  if ((uint32_t)(now - stateStartedAt) >= stateDuration) {
    // Re-selecting the same expression is allowed, producing
    // occasional longer idle or emotional periods.
    enterExpression(pickExpression(), now);
  }
}

void updateGaze(uint32_t elapsedMs) {
  float targetX = 0.0f;

  if (currentExpression == Expression::LOOK_LEFT) {
    targetX = -12.0f;
  } else if (currentExpression == Expression::LOOK_RIGHT) {
    targetX = 12.0f;
  }

  // Time-dependent easing toward the desired eye position.
  const float blend =
    (float)elapsedMs / (110.0f + (float)elapsedMs);

  gazeX += (targetX - gazeX) * blend;
}

// ==================== Drawing primitives ====================

void drawNormalEye(int centerX) {
  display.fillRoundRect(
    centerX - EYE_WIDTH / 2,
    EYE_CENTER_Y - EYE_HEIGHT / 2,
    EYE_WIDTH,
    EYE_HEIGHT,
    EYE_RADIUS,
    SSD1306_WHITE
  );
}

void drawAngryEye(int centerX, bool leftEye) {
  drawNormalEye(centerX);

  const int x = centerX - EYE_WIDTH / 2;
  const int y = EYE_CENTER_Y - EYE_HEIGHT / 2;

  // Remove a wedge from the upper edge.
  // The inner corners sit lower to form an angry brow.
  if (leftEye) {
    display.fillTriangle(
      x - 1, y - 1,
      x + EYE_WIDTH, y - 1,
      x + EYE_WIDTH, y + 14,
      SSD1306_BLACK
    );
  } else {
    display.fillTriangle(
      x - 1, y - 1,
      x + EYE_WIDTH, y - 1,
      x - 1, y + 14,
      SSD1306_BLACK
    );
  }
}

void drawHappyEye(int centerX) {
  drawNormalEye(centerX);

  // A lower circular cutout leaves an upward arch.
  display.fillCircle(
    centerX,
    EYE_CENTER_Y + 18,
    20,
    SSD1306_BLACK
  );
}

void drawSleepEye(int centerX) {
  const int centerY = EYE_CENTER_Y - 3;

  // Hollow circle with its top half removed: a U-shaped lid.
  display.fillCircle(
    centerX, centerY, 14, SSD1306_WHITE
  );

  display.fillCircle(
    centerX, centerY, 11, SSD1306_BLACK
  );

  display.fillRect(
    centerX - 15,
    centerY - 15,
    31,
    15,
    SSD1306_BLACK
  );
}

void drawHeartEye(int centerX) {
  const int lobeY = EYE_CENTER_Y - 6;

  display.fillCircle(
    centerX - 7, lobeY, 9, SSD1306_WHITE
  );

  display.fillCircle(
    centerX + 7, lobeY, 9, SSD1306_WHITE
  );

  display.fillTriangle(
    centerX - 16, lobeY + 2,
    centerX + 16, lobeY + 2,
    centerX, EYE_CENTER_Y + 17,
    SSD1306_WHITE
  );
}

void drawBlink(int leftX, int rightX) {
  if (blinkClosure <= 0.0f) {
    return;
  }

  // Narrow the visible area around the horizontal eye center.
  const int visibleHeight =
    2 + (int)((EYE_HEIGHT - 2) * (1.0f - blinkClosure));

  const int top = EYE_CENTER_Y - visibleHeight / 2;
  const int bottom = top + visibleHeight;

  display.fillRect(
    0, 0, SCREEN_WIDTH, top, SSD1306_BLACK
  );

  display.fillRect(
    0, bottom,
    SCREEN_WIDTH, SCREEN_HEIGHT - bottom,
    SSD1306_BLACK
  );

  // Ensure fully closed eyes remain visible as thin lids,
  // including when blinking from the happy expression.
  if (blinkClosure > 0.95f) {
    display.fillRect(
      leftX - EYE_WIDTH / 2,
      EYE_CENTER_Y - 1,
      EYE_WIDTH, 2,
      SSD1306_WHITE
    );

    display.fillRect(
      rightX - EYE_WIDTH / 2,
      EYE_CENTER_Y - 1,
      EYE_WIDTH, 2,
      SSD1306_WHITE
    );
  }
}

// ==================== Frame rendering ====================

void renderEyes() {
  display.clearDisplay();

  const int offset =
    (int)(gazeX + (gazeX >= 0.0f ? 0.5f : -0.5f));

  const int leftX = LEFT_CENTER_X + offset;
  const int rightX = RIGHT_CENTER_X + offset;

  switch (currentExpression) {
    case Expression::IDLE:
    case Expression::LOOK_LEFT:
    case Expression::LOOK_RIGHT:
      drawNormalEye(leftX);
      drawNormalEye(rightX);
      break;

    case Expression::ANGRY:
      drawAngryEye(leftX, true);
      drawAngryEye(rightX, false);
      break;

    case Expression::HAPPY:
      drawHappyEye(leftX);
      drawHappyEye(rightX);
      break;

    case Expression::SLEEP:
      drawSleepEye(leftX);
      drawSleepEye(rightX);
      break;

    case Expression::HEARTS:
      drawHeartEye(leftX);
      drawHeartEye(rightX);
      break;
  }

  if (canBlink()) {
    drawBlink(leftX, rightX);
  }

  display.display();
}

// ==================== Arduino entry points ====================

void setup() {
  Serial.begin(115200);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);

  // periphBegin=false preserves the Wire configuration above.
  if (!display.begin(
        SSD1306_SWITCHCAPVCC,
        OLED_ADDRESS,
        true,
        false)) {
    Serial.println("SSD1306 framebuffer allocation failed.");
    return;
  }

  displayReady = true;

  randomSeed(esp_random());

  display.clearDisplay();
  display.setRotation(0);

  const uint32_t now = millis();

  enterExpression(Expression::IDLE, now);
  lastFrameAt = now;

  renderEyes();
}

void loop() {
  if (!displayReady) {
    return;
  }

  const uint32_t now = millis();
  const uint32_t elapsed = now - lastFrameAt;

  if (elapsed < FRAME_INTERVAL_MS) {
    return;
  }

  lastFrameAt = now;

  updateExpression(now);
  updateGaze(elapsed);
  updateBlink(now);
  renderEyes();

  // Future sensor, button, or communication logic can be
  // scheduled here or before the frame-timing check.
}