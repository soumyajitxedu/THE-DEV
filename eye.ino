/*
 * EXPRESSIVE OLED EYES - Arduino Uno (ATmega328P)
 * 0.96" SSD1306 OLED (I2C, Address 0x3C)
 * Developed by Soumyajit | Designed by Soham
 * 
 * Features:
 * - 20+ animation states with smooth transitions
 * - Realistic saccadic eye movements
 * - Random behavior engine (sleep 30s, active 2min cycles)
 * - Animated text quotes with typewriter effects
 * - Non-blocking animation system using millis()
 * - Optimized for ~80% PROGMEM usage on ATmega328P
 */

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define OLED_ADDRESS 0x3C

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// ============== CONFIGURATION ==============
const unsigned long SLEEP_DURATION = 30000;      // 30 seconds sleep
const unsigned long ACTIVE_DURATION = 120000;    // 2 minutes active
const unsigned long FRAME_INTERVAL = 16;         // ~60 FPS target
const uint8_t MAX_QUOTES = 15;

// ============== ANIMATION STATES ==============
enum EyeState {
  STATE_IDLE,           // 0 - Looking around
  STATE_BLINK,          // 1 - Blinking
  STATE_HAPPY,          // 2 - Happy/Cute
  STATE_ANGRY,          // 3 - Angry
  STATE_SLEEP,          // 4 - Sleeping
  STATE_WAKEUP,         // 5 - Waking up
  STATE_SURPRISED,      // 6 - Surprised
  STATE_SAD,            // 7 - Sad
  STATE_CONFUSED,       // 8 - Confused
  STATE_WINK_LEFT,      // 9 - Wink left
  STATE_WINK_RIGHT,     // 10 - Wink right
  STATE_LOOK_SIDE,      // 11 - Looking side to side
  STATE_SHY,            // 12 - Shy/Hiding
  STATE_ENERGETIC,      // 13 - Energetic bounce
  STATE_THINKING,       // 14 - Thinking
  STATE_WRITING,        // 15 - Writing motion
  STATE_SHOOT,          // 16 - Shooting bullets (rare)
  STATE_WOBBLE,         // 17 - Wobble/shake
  STATE_GREETING,       // 18 - Greeting wave
  STATE_LOVE,           // 19 - Love hearts
  STATE_MOTIVATED,      // 20 - Motivated
  STATE_TEXT_DISPLAY,   // 21 - Show quote text
  STATE_TRANSITION      // 22 - Transition between states
};

// ============== EYE PARAMETERS ==============
struct Eye {
  float x, y;           // Current position
  float targetX, targetY; // Target position
  float width, height;  // Current size
  float targetW, targetH; // Target size
  float pupilX, pupilY; // Pupil offset
  float targetPupilX, targetPupilY;
  uint8_t cornerRadius;
  bool isOpen;
};

Eye leftEye = {32, 28, 32, 28, 28, 24, 28, 24, 0, 0, 0, 0, 8, true};
Eye rightEye = {96, 28, 96, 28, 28, 24, 28, 24, 0, 0, 0, 0, 8, true};

// ============== GLOBAL VARIABLES ==============
EyeState currentState = STATE_IDLE;
EyeState nextState = STATE_IDLE;
EyeState previousState = STATE_IDLE;

unsigned long lastFrameTime = 0;
unsigned long stateStartTime = 0;
unsigned long stateDuration = 0;
unsigned long sleepCycleStart = 0;
unsigned long lastSaccadeTime = 0;
unsigned long lastQuoteTime = 0;
unsigned long quoteDisplayStart = 0;

bool isSleeping = false;
bool isActive = true;
bool showingQuote = false;
bool transitionInProgress = false;

uint8_t animationFrame = 0;
uint8_t maxFrames = 0;
uint8_t saccadeCounter = 0;

float globalOffsetX = 0;
float globalOffsetY = 0;
float targetGlobalOffsetX = 0;
float targetGlobalOffsetY = 0;

// ============== QUOTES DATABASE ==============
const char* quotes[MAX_QUOTES] = {
  "eww",
  "hi",
  "yoo, i am watching you",
  "be kind always",
  "you should work hard",
  "do whatever it takes",
  "wanna be yours",
  "can say that day when",
  "you didn't kill yourself?",
  "you always special",
  "at least for me",
  "i am developed by soumyajit",
  "design by soham",
  "love you",
  "stay motivated"
};

uint8_t currentQuoteIndex = 0;
char displayBuffer[40];
uint8_t typewriterPos = 0;

// ============== BULLET SYSTEM ==============
struct Bullet {
  float x, y;
  float vx, vy;
  bool active;
};
Bullet bullets[3];

// ============== HEART PARTICLES ==============
struct Heart {
  float x, y;
  float vy;
  bool active;
};
Heart hearts[5];

// ============== FUNCTION PROTOTYPES ==============
void updatePhysics();
void renderFrame();
void updateStateMachine();
void transitionTo(EyeState newState, unsigned long duration);
void drawEye(Eye &eye, bool isLeft);
void drawPupil(Eye &eye, bool isLeft);
void drawEyelid(Eye &eye, float closure);
void drawExpressionOverlay();
void drawTextQuote();
void updateBullets();
void updateHearts();
void triggerSaccade();
void lerpEye(Eye &eye, float factor);
float lerp(float a, float b, float t);
int randomRange(int min, int max);

// ============== SETUP ==============
void setup() {
  Serial.begin(115200);
  
  if(!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDRESS)) {
    Serial.println(F("SSD1306 allocation failed"));
    for(;;);
  }
  
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.display();
  
  randomSeed(analogRead(A0) + analogRead(A1) + analogRead(A2));
  
  // Initialize bullets
  for(int i = 0; i < 3; i++) bullets[i].active = false;
  // Initialize hearts
  for(int i = 0; i < 5; i++) hearts[i].active = false;
  
  sleepCycleStart = millis();
  lastQuoteTime = millis();
  
  // Startup animation
  transitionTo(STATE_WAKEUP, 2000);
}

// ============== MAIN LOOP ==============
void loop() {
  unsigned long currentTime = millis();
  
  if(currentTime - lastFrameTime < FRAME_INTERVAL) return;
  lastFrameTime = currentTime;
  
  updateStateMachine();
  updatePhysics();
  renderFrame();
}

// ============== STATE MACHINE ==============
void updateStateMachine() {
  unsigned long currentTime = millis();
  unsigned long elapsed = currentTime - stateStartTime;
  
  // Sleep cycle management
  if(isActive && currentTime - sleepCycleStart >= ACTIVE_DURATION) {
    isActive = false;
    isSleeping = true;
    sleepCycleStart = currentTime;
    transitionTo(STATE_SLEEP, 1000);
    return;
  }
  
  if(!isActive && currentTime - sleepCycleStart >= SLEEP_DURATION) {
    isActive = true;
    isSleeping = false;
    sleepCycleStart = currentTime;
    transitionTo(STATE_WAKEUP, 1500);
    return;
  }
  
  // State duration check
  if(!transitionInProgress && stateDuration > 0 && elapsed >= stateDuration) {
    pickNextRandomState();
  }
  
  // Random saccade during idle
  if(currentState == STATE_IDLE && currentTime - lastSaccadeTime > randomRange(800, 2500)) {
    triggerSaccade();
    lastSaccadeTime = currentTime;
  }
  
  // Random quote display
  if(isActive && !showingQuote && currentTime - lastQuoteTime > randomRange(15000, 30000)) {
    showingQuote = true;
    quoteDisplayStart = currentTime;
    currentQuoteIndex = randomRange(0, MAX_QUOTES);
    typewriterPos = 0;
    previousState = currentState;
    transitionTo(STATE_TEXT_DISPLAY, 4000);
  }
}

void pickNextRandomState() {
  if(isSleeping) {
    transitionTo(STATE_SLEEP, 5000);
    return;
  }
  
  // Weighted random selection
  int r = randomRange(0, 100);
  EyeState next;
  
  if(r < 25) next = STATE_IDLE;
  else if(r < 35) next = STATE_BLINK;
  else if(r < 42) next = STATE_HAPPY;
  else if(r < 48) next = STATE_ANGRY;
  else if(r < 54) next = STATE_SURPRISED;
  else if(r < 60) next = STATE_SAD;
  else if(r < 65) next = STATE_CONFUSED;
  else if(r < 70) next = STATE_WINK_LEFT;
  else if(r < 75) next = STATE_WINK_RIGHT;
  else if(r < 80) next = STATE_LOOK_SIDE;
  else if(r < 84) next = STATE_SHY;
  else if(r < 88) next = STATE_ENERGETIC;
  else if(r < 91) next = STATE_THINKING;
  else if(r < 94) next = STATE_WRITING;
  else if(r < 96) next = STATE_SHOOT;      // Rare
  else if(r < 98) next = STATE_WOBBLE;
  else next = STATE_GREETING;
  
  // Love and motivated are special - triggered less frequently
  if(randomRange(0, 20) == 0) next = STATE_LOVE;
  if(randomRange(0, 25) == 0) next = STATE_MOTIVATED;
  
  transitionTo(next, randomRange(2000, 5000));
}

void transitionTo(EyeState newState, unsigned long duration) {
  if(transitionInProgress) return;
  
  previousState = currentState;
  nextState = newState;
  currentState = STATE_TRANSITION;
  stateStartTime = millis();
  stateDuration = 300; // Transition time
  transitionInProgress = true;
  animationFrame = 0;
  
  // Pre-configure next state
  switch(newState) {
    case STATE_IDLE:
      leftEye.targetX = 32; leftEye.targetY = 28;
      rightEye.targetX = 96; rightEye.targetY = 28;
      leftEye.targetW = 28; leftEye.targetH = 24;
      rightEye.targetW = 28; rightEye.targetH = 24;
      break;
      
    case STATE_HAPPY:
      leftEye.targetW = 32; leftEye.targetH = 20;
      rightEye.targetW = 32; rightEye.targetH = 20;
      leftEye.targetY = 30; rightEye.targetY = 30;
      break;
      
    case STATE_ANGRY:
      leftEye.targetW = 30; leftEye.targetH = 22;
      rightEye.targetW = 30; rightEye.targetH = 22;
      leftEye.targetPupilX = 4; leftEye.targetPupilY = -2;
      rightEye.targetPupilX = -4; rightEye.targetPupilY = -2;
      break;
      
    case STATE_SURPRISED:
      leftEye.targetW = 30; leftEye.targetH = 30;
      rightEye.targetW = 30; rightEye.targetH = 30;
      leftEye.targetPupilY = -4; rightEye.targetPupilY = -4;
      break;
      
    case STATE_SAD:
      leftEye.targetW = 26; leftEye.targetH = 22;
      rightEye.targetW = 26; rightEye.targetH = 22;
      leftEye.targetPupilY = 3; rightEye.targetPupilY = 3;
      break;
      
    case STATE_SLEEP:
      leftEye.targetH = 2; rightEye.targetH = 2;
      leftEye.targetW = 24; rightEye.targetW = 24;
      break;
      
    case STATE_WAKEUP:
      leftEye.targetH = 24; rightEye.targetH = 24;
      leftEye.targetW = 28; rightEye.targetW = 28;
      break;
      
    case STATE_CONFUSED:
      leftEye.targetPupilX = -5; leftEye.targetPupilY = -3;
      rightEye.targetPupilX = 5; rightEye.targetPupilY = 2;
      break;
      
    case STATE_WINK_LEFT:
      leftEye.targetH = 2;
      rightEye.targetH = 24;
      break;
      
    case STATE_WINK_RIGHT:
      leftEye.targetH = 24;
      rightEye.targetH = 2;
      break;
      
    case STATE_LOOK_SIDE:
      leftEye.targetPupilX = randomRange(0, 2) == 0 ? -6 : 6;
      rightEye.targetPupilX = leftEye.targetPupilX;
      break;
      
    case STATE_SHY:
      leftEye.targetPupilY = 5; rightEye.targetPupilY = 5;
      leftEye.targetW = 24; rightEye.targetW = 24;
      break;
      
    case STATE_ENERGETIC:
      targetGlobalOffsetY = -3;
      break;
      
    case STATE_THINKING:
      leftEye.targetPupilY = -2; rightEye.targetPupilY = -2;
      break;
      
    case STATE_SHOOT:
      // Initialize bullets
      for(int i = 0; i < 3; i++) {
        bullets[i].x = 64;
        bullets[i].y = 32;
        bullets[i].vx = randomRange(-30, 30) / 10.0;
        bullets[i].vy = randomRange(20, 40) / 10.0;
        bullets[i].active = true;
      }
      break;
      
    case STATE_WOBBLE:
      targetGlobalOffsetX = randomRange(-3, 3);
      break;
      
    case STATE_GREETING:
      targetGlobalOffsetY = -2;
      break;
      
    case STATE_LOVE:
      for(int i = 0; i < 5; i++) {
        hearts[i].x = randomRange(20, 108);
        hearts[i].y = 64;
        hearts[i].vy = randomRange(10, 25) / 10.0;
        hearts[i].active = true;
      }
      break;
      
    case STATE_MOTIVATED:
      leftEye.targetW = 30; leftEye.targetH = 26;
      rightEye.targetW = 30; rightEye.targetH = 26;
      break;
      
    case STATE_TEXT_DISPLAY:
      typewriterPos = 0;
      break;
      
    default:
      break;
  }
  
  // Schedule actual state after transition
  stateDuration = 300;
  maxFrames = duration / FRAME_INTERVAL;
}

void completeTransition() {
  currentState = nextState;
  stateStartTime = millis();
  transitionInProgress = false;
  
  if(currentState != STATE_TEXT_DISPLAY && currentState != STATE_TRANSITION) {
    stateDuration = maxFrames * FRAME_INTERVAL;
  }
  
  // Reset some targets after transition
  if(currentState == STATE_IDLE) {
    leftEye.targetPupilX = 0; leftEye.targetPupilY = 0;
    rightEye.targetPupilX = 0; rightEye.targetPupilY = 0;
  }
}

// ============== PHYSICS UPDATE ==============
void updatePhysics() {
  float lerpFactor = 0.15; // Smooth movement factor
  
  if(currentState == STATE_TRANSITION) {
    lerpFactor = 0.25;
    unsigned long elapsed = millis() - stateStartTime;
    if(elapsed >= stateDuration) {
      completeTransition();
    }
  }
  
  // Lerp eyes
  lerpEye(leftEye, lerpFactor);
  lerpEye(rightEye, lerpFactor);
  
  // Global offset lerp
  globalOffsetX = lerp(globalOffsetX, targetGlobalOffsetX, 0.1);
  globalOffsetY = lerp(globalOffsetY, targetGlobalOffsetY, 0.1);
  
  // Reset global offsets for certain states
  if(currentState != STATE_ENERGETIC && currentState != STATE_WOBBLE && currentState != STATE_GREETING) {
    targetGlobalOffsetX = 0;
    targetGlobalOffsetY = 0;
  }
  
  // Energetic bounce
  if(currentState == STATE_ENERGETIC) {
    float bounce = sin(millis() / 150.0) * 3;
    globalOffsetY = bounce;
  }
  
  // Wobble shake
  if(currentState == STATE_WOBBLE) {
    globalOffsetX = sin(millis() / 50.0) * 2;
    globalOffsetY = cos(millis() / 40.0) * 1.5;
  }
  
  // Greeting wave
  if(currentState == STATE_GREETING) {
    globalOffsetY = sin(millis() / 200.0) * 2;
  }
  
  // Update bullets
  if(currentState == STATE_SHOOT) {
    updateBullets();
  }
  
  // Update hearts
  if(currentState == STATE_LOVE) {
    updateHearts();
  }
  
  // Typewriter effect
  if(currentState == STATE_TEXT_DISPLAY) {
    unsigned long elapsed = millis() - stateStartTime;
    if(elapsed > 500 && typewriterPos < strlen(quotes[currentQuoteIndex])) {
      if((millis() / 80) > typewriterPos) {
        typewriterPos++;
      }
    }
    if(elapsed > 3500) {
      showingQuote = false;
      lastQuoteTime = millis();
      pickNextRandomState();
    }
  }
  
  // Blink during idle
  if(currentState == STATE_IDLE && randomRange(0, 150) == 0) {
    leftEye.targetH = 2;
    rightEye.targetH = 2;
  }
  if(currentState == STATE_IDLE && leftEye.height < 5 && randomRange(0, 10) == 0) {
    leftEye.targetH = 24;
    rightEye.targetH = 24;
  }
  
  animationFrame++;
}

void lerpEye(Eye &eye, float factor) {
  eye.x = lerp(eye.x, eye.targetX, factor);
  eye.y = lerp(eye.y, eye.targetY, factor);
  eye.width = lerp(eye.width, eye.targetW, factor);
  eye.height = lerp(eye.height, eye.targetH, factor);
  eye.pupilX = lerp(eye.pupilX, eye.targetPupilX, factor);
  eye.pupilY = lerp(eye.pupilY, eye.targetPupilY, factor);
}

float lerp(float a, float b, float t) {
  return a + (b - a) * t;
}

void triggerSaccade() {
  int dirX = randomRange(-6, 7);
  int dirY = randomRange(-4, 5);
  leftEye.targetPupilX = dirX;
  leftEye.targetPupilY = dirY;
  rightEye.targetPupilX = dirX;
  rightEye.targetPupilY = dirY;
  saccadeCounter++;
}

// ============== RENDERING ==============
void renderFrame() {
  display.clearDisplay();
  
  // Draw bullets behind eyes
  if(currentState == STATE_SHOOT) {
    for(int i = 0; i < 3; i++) {
      if(bullets[i].active) {
        display.fillCircle(bullets[i].x, bullets[i].y, 2, SSD1306_WHITE);
      }
    }
  }
  
  // Draw hearts behind eyes
  if(currentState == STATE_LOVE) {
    for(int i = 0; i < 5; i++) {
      if(hearts[i].active) {
        drawSmallHeart(hearts[i].x, hearts[i].y);
      }
    }
  }
  
  // Draw eyes
  drawEye(leftEye, true);
  drawEye(rightEye, false);
  
  // Draw expression overlays
  drawExpressionOverlay();
  
  // Draw text quote if active
  if(currentState == STATE_TEXT_DISPLAY) {
    drawTextQuote();
  }
  
  display.display();
}

void drawEye(Eye &eye, bool isLeft) {
  int x = eye.x + globalOffsetX - eye.width / 2;
  int y = eye.y + globalOffsetY - eye.height / 2;
  int w = eye.width;
  int h = eye.height;
  
  // Eye shape (rounded rectangle)
  if(h > 3) {
    display.fillRoundRect(x, y, w, h, eye.cornerRadius, SSD1306_WHITE);
    
    // Draw pupil
    if(currentState != STATE_SLEEP && currentState != STATE_WAKEUP) {
      drawPupil(eye, isLeft);
    }
  } else {
    // Closed eye line
    display.drawLine(x, eye.y + globalOffsetY, x + w, eye.y + globalOffsetY, SSD1306_WHITE);
  }
  
  // Eyelid overlays for expressions
  if(currentState == STATE_HAPPY) {
    // Happy arch
    display.fillTriangle(x, y, x + w/2, y - 6, x + w, y, SSD1306_BLACK);
  }
  else if(currentState == STATE_ANGRY) {
    // Angry slant
    display.fillTriangle(x, y - 2, x + w, y + 4, x + w, y - 2, SSD1306_BLACK);
    if(!isLeft) {
      display.fillTriangle(x, y - 2, x, y + 4, x + w, y - 2, SSD1306_BLACK);
    }
  }
  else if(currentState == STATE_SAD) {
    // Sad droop
    display.fillTriangle(x, y - 2, x + w/2, y + 4, x + w, y - 2, SSD1306_BLACK);
  }
  else if(currentState == STATE_SURPRISED) {
    // Raised eyebrows effect
    display.drawLine(x - 2, y - 4, x + w + 2, y - 4, SSD1306_WHITE);
  }
}

void drawPupil(Eye &eye, bool isLeft) {
  int centerX = eye.x + globalOffsetX + eye.pupilX;
  int centerY = eye.y + globalOffsetY + eye.pupilY;
  int pupilSize = 8;
  
  // Constrain pupil to eye bounds
  int maxOffsetX = (eye.width / 2) - pupilSize - 2;
  int maxOffsetY = (eye.height / 2) - pupilSize - 2;
  
  centerX = constrain(centerX, eye.x + globalOffsetX - maxOffsetX, eye.x + globalOffsetX + maxOffsetX);
  centerY = constrain(centerY, eye.y + globalOffsetY - maxOffsetY, eye.y + globalOffsetY + maxOffsetY);
  
  // Draw pupil (filled circle with highlight)
  display.fillCircle(centerX, centerY, pupilSize / 2, SSD1306_BLACK);
  
  // Pupil highlight
  display.fillCircle(centerX - 1, centerY - 1, 2, SSD1306_WHITE);
}

void drawExpressionOverlay() {
  switch(currentState) {
    case STATE_CONFUSED:
      // Question marks
      display.setCursor(52, 8);
      display.print(F("?"));
      display.setCursor(72, 8);
      display.print(F("?"));
      break;
      
    case STATE_THINKING:
      // Thought bubble dots
      if((millis() / 500) % 3 == 0) {
        display.fillCircle(64, 8, 2, SSD1306_WHITE);
      } else if((millis() / 500) % 3 == 1) {
        display.fillCircle(64, 8, 2, SSD1306_WHITE);
        display.fillCircle(70, 6, 1, SSD1306_WHITE);
      } else {
        display.fillCircle(64, 8, 2, SSD1306_WHITE);
        display.fillCircle(70, 6, 1, SSD1306_WHITE);
        display.fillCircle(75, 4, 1, SSD1306_WHITE);
      }
      break;
      
    case STATE_WRITING:
      // Pencil motion
      int penX = 64 + sin(millis() / 100.0) * 10;
      int penY = 50 + cos(millis() / 80.0) * 3;
      display.drawLine(penX, penY, penX - 3, penY + 6, SSD1306_WHITE);
      display.fillCircle(penX - 3, penY + 6, 1, SSD1306_WHITE);
      break;
      
    case STATE_SHY:
      // Blush marks
      display.drawPixel(18, 38, SSD1306_WHITE);
      display.drawPixel(20, 40, SSD1306_WHITE);
      display.drawPixel(108, 38, SSD1306_WHITE);
      display.drawPixel(110, 40, SSD1306_WHITE);
      break;
      
    case STATE_MOTIVATED:
      // Sparkle effect
      if((millis() / 200) % 2 == 0) {
        display.drawPixel(10, 10, SSD1306_WHITE);
        display.drawPixel(118, 10, SSD1306_WHITE);
      }
      break;
      
    default:
      break;
  }
}

void drawTextQuote() {
  display.fillRect(0, 52, 128, 12, SSD1306_BLACK);
  display.drawRect(0, 52, 128, 12, SSD1306_WHITE);
  
  display.setCursor(4, 55);
  display.setTextSize(1);
  
  // Typewriter effect
  strncpy(displayBuffer, quotes[currentQuoteIndex], typewriterPos);
  displayBuffer[typewriterPos] = '\0';
  display.print(displayBuffer);
  
  // Cursor blink
  if((millis() / 300) % 2 == 0 && typewriterPos < strlen(quotes[currentQuoteIndex])) {
    display.print(F("_"));
  }
}

void drawSmallHeart(int x, int y) {
  // Simple 5x5 heart shape
  display.drawPixel(x, y - 1, SSD1306_WHITE);
  display.drawPixel(x - 1, y - 2, SSD1306_WHITE);
  display.drawPixel(x + 1, y - 2, SSD1306_WHITE);
  display.drawPixel(x - 2, y - 1, SSD1306_WHITE);
  display.drawPixel(x + 2, y - 1, SSD1306_WHITE);
  display.drawPixel(x - 1, y, SSD1306_WHITE);
  display.drawPixel(x + 1, y, SSD1306_WHITE);
  display.drawPixel(x, y + 1, SSD1306_WHITE);
}

// ============== BULLET SYSTEM ==============
void updateBullets() {
  for(int i = 0; i < 3; i++) {
    if(bullets[i].active) {
      bullets[i].x += bullets[i].vx;
      bullets[i].y += bullets[i].vy;
      
      // Deactivate if off screen
      if(bullets[i].y > 64 || bullets[i].x < 0 || bullets[i].x > 128) {
        bullets[i].active = false;
      }
    }
  }
}

// ============== HEART SYSTEM ==============
void updateHearts() {
  for(int i = 0; i < 5; i++) {
    if(hearts[i].active) {
      hearts[i].y -= hearts[i].vy;
      
      // Deactivate if off top
      if(hearts[i].y < 0) {
        hearts[i].active = false;
      }
    }
  }
}

// ============== UTILITY ==============
int randomRange(int min, int max) {
  return random(min, max);
}