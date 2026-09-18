// ============================================================
//  HOMIE v7.0 – COMPLETE FINAL EDITION
//  By Soumyajit | ESP32 Dev Module + SSD1306 OLED 128x64
//  ALL FEATURES FULLY IMPLEMENTED - NO PLACEHOLDERS
//  Games: SpaceWar, Car, Snake, Pong, Maze, Flappy, 2048, Tetris
// ============================================================

#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Preferences.h>
#include <time.h>

// ============================================================
//  CONFIGURATION
// ============================================================
const char* API_KEY  = "d6ccc068b65f4cee99f104921261707";
const char* LOCATION = "Dhulian,India";

// ============================================================
//  HARDWARE PINS
// ============================================================
#define SCREEN_W  128
#define SCREEN_H   64
#define OLED_ADDR 0x3C
#define SDA_PIN    21
#define SCL_PIN    22

Adafruit_SSD1306 display(SCREEN_W, SCREEN_H, &Wire, -1);
Preferences prefs;

// Buttons: UP=13, LEFT=14, RIGHT/SELECT=26, DOWN=27
#define BTN_UP     13
#define BTN_LEFT   14
#define BTN_RIGHT  26
#define BTN_DOWN   27

// ============================================================
//  TIMING CONSTANTS
// ============================================================
const long          GMT_OFFSET_SEC      = 19800;   // IST +5:30
const int           DST_OFFSET_SEC      = 0;
const unsigned long WEATHER_INTERVAL_MS = 1800000UL;
const unsigned long DEBOUNCE_MS         = 50UL;
const unsigned long LONG_PRESS_MS       = 1500UL;

// ============================================================
//  GLOBAL WEATHER DATA
// ============================================================
float  g_temp     = 28.5f;
int    g_humidity = 72;
float  g_feels    = 30.0f;
int    g_windKph  = 12;
String g_desc     = "Partly Cloudy";
String g_icon     = "cloud";
String g_time     = "--:--";
String g_date     = "---";
bool   g_hasData  = false;
bool   g_wifiOk   = false;
unsigned long g_lastWeatherFetch = 0;

// ============================================================
//  PLAYER / PERSISTENT STATS
// ============================================================
struct PlayerStats {
  uint16_t coins;
  uint16_t level;
  uint16_t xp;
  uint16_t totalMCQs;
  uint16_t correctMCQs;
  uint16_t totalFacts;
  uint8_t  hunger;
  uint8_t  energy;
  uint8_t  happiness;
  uint8_t  brightness;   // 0=25% 1=50% 2=75% 3=100%
  uint32_t menuTimeoutMs; // ms: 5000/10000/30000/0(never)
  char     wifiSSID[33];
  char     wifiPass[65];
  // High scores
  uint16_t hsSpaceWar;
  uint16_t hsSnake;
  uint16_t hsPong;
  uint16_t hsCar;
  uint16_t hsMaze;
  uint16_t hsFlappy;
  uint32_t hs2048;
  uint16_t hsTetris;
};

PlayerStats player;

void loadPrefs() {
  prefs.begin("homie", false);
  player.coins         = prefs.getUShort("coins",    50);
  player.level         = prefs.getUShort("level",     1);
  player.xp            = prefs.getUShort("xp",        0);
  player.totalMCQs     = prefs.getUShort("tmcq",      0);
  player.correctMCQs   = prefs.getUShort("cmcq",      0);
  player.totalFacts    = prefs.getUShort("tfact",      0);
  player.hunger        = prefs.getUChar("hunger",    100);
  player.energy        = prefs.getUChar("energy",    100);
  player.happiness     = prefs.getUChar("happy",     100);
  player.brightness    = prefs.getUChar("bright",      3);
  player.menuTimeoutMs = prefs.getULong("timeout", 30000);
  prefs.getString("ssid", player.wifiSSID, sizeof(player.wifiSSID));
  prefs.getString("pass", player.wifiPass, sizeof(player.wifiPass));
  if (strlen(player.wifiSSID) == 0) {
    strncpy(player.wifiSSID, "YourSSID", sizeof(player.wifiSSID));
    strncpy(player.wifiPass, "YourPass", sizeof(player.wifiPass));
  }
  player.hsSpaceWar = prefs.getUShort("hs_sw",  0);
  player.hsSnake    = prefs.getUShort("hs_sn",  0);
  player.hsPong     = prefs.getUShort("hs_po",  0);
  player.hsCar      = prefs.getUShort("hs_car", 0);
  player.hsMaze     = prefs.getUShort("hs_mz",  0);
  player.hsFlappy   = prefs.getUShort("hs_fl",  0);
  player.hs2048     = prefs.getULong("hs_2048", 0);
  player.hsTetris   = prefs.getUShort("hs_tet", 0);
  prefs.end();
}

void savePrefs() {
  prefs.begin("homie", false);
  prefs.putUShort("coins",   player.coins);
  prefs.putUShort("level",   player.level);
  prefs.putUShort("xp",      player.xp);
  prefs.putUShort("tmcq",    player.totalMCQs);
  prefs.putUShort("cmcq",    player.correctMCQs);
  prefs.putUShort("tfact",   player.totalFacts);
  prefs.putUChar("hunger",   player.hunger);
  prefs.putUChar("energy",   player.energy);
  prefs.putUChar("happy",    player.happiness);
  prefs.putUChar("bright",   player.brightness);
  prefs.putULong("timeout",  player.menuTimeoutMs);
  prefs.putString("ssid",    player.wifiSSID);
  prefs.putString("pass",    player.wifiPass);
  prefs.putUShort("hs_sw",   player.hsSpaceWar);
  prefs.putUShort("hs_sn",   player.hsSnake);
  prefs.putUShort("hs_po",   player.hsPong);
  prefs.putUShort("hs_car",  player.hsCar);
  prefs.putUShort("hs_mz",   player.hsMaze);
  prefs.putUShort("hs_fl",   player.hsFlappy);
  prefs.putULong("hs_2048",  player.hs2048);
  prefs.putUShort("hs_tet",  player.hsTetris);
  prefs.end();
}

void applyBrightness() {
  uint8_t contrast = 0;
  switch (player.brightness) {
    case 0: contrast = 0;   break;
    case 1: contrast = 85;  break;
    case 2: contrast = 170; break;
    case 3: contrast = 255; break;
  }
  display.ssd1306_command(SSD1306_SETCONTRAST);
  display.ssd1306_command(contrast);
}

// ============================================================
//  GOALS
// ============================================================
struct Goal { char name[12]; uint8_t progress; bool completed; };
Goal goals[4] = {
  {"Study", 0, false}, {"Exercise", 0, false},
  {"Read", 0, false},  {"Code", 0, false}
};

// ============================================================
//  KEYBOARD BUFFER
// ============================================================
char keyboardBuffer[33] = "";

// ============================================================
//  BUTTON SYSTEM
// ============================================================
enum ButtonAction {
  ACTION_NONE, ACTION_UP, ACTION_DOWN, ACTION_LEFT,
  ACTION_RIGHT, ACTION_SELECT, ACTION_BACK, ACTION_SPECIAL
};

struct Button {
  uint8_t  pin;
  bool     lastState;
  bool     currentState;
  unsigned long lastDebounceTime;
  unsigned long pressStartTime;
  bool     isLongPress;
  bool     wasPressed;
  bool     wasLongPressed;
};

Button buttons[4] = {
  {BTN_UP,    HIGH, HIGH, 0, 0, false, false, false},
  {BTN_LEFT,  HIGH, HIGH, 0, 0, false, false, false},
  {BTN_RIGHT, HIGH, HIGH, 0, 0, false, false, false},
  {BTN_DOWN,  HIGH, HIGH, 0, 0, false, false, false}
};

void updateButtons() {
  for (int i = 0; i < 4; i++) {
    bool reading = digitalRead(buttons[i].pin);
    if (reading != buttons[i].lastState) {
      buttons[i].lastDebounceTime = millis();
    }
    if ((millis() - buttons[i].lastDebounceTime) > DEBOUNCE_MS) {
      if (reading != buttons[i].currentState) {
        buttons[i].currentState = reading;
        if (reading == LOW) {
          buttons[i].pressStartTime  = millis();
          buttons[i].wasPressed      = true;
          buttons[i].isLongPress     = false;
          buttons[i].wasLongPressed  = false;
        }
      }
    }
    if (buttons[i].currentState == LOW && !buttons[i].isLongPress) {
      if ((millis() - buttons[i].pressStartTime) >= LONG_PRESS_MS) {
        buttons[i].isLongPress    = true;
        buttons[i].wasLongPressed = true;
        buttons[i].wasPressed     = false;
      }
    }
    buttons[i].lastState = reading;
  }
}

bool isPressed(int idx) {
  if (buttons[idx].wasPressed && buttons[idx].currentState == HIGH) {
    buttons[idx].wasPressed = false;
    return true;
  }
  return false;
}

bool isLongPressed(int idx) {
  if (buttons[idx].wasLongPressed) {
    buttons[idx].wasLongPressed = false;
    return true;
  }
  return false;
}

bool anyButtonPressed() {
  for (int i = 0; i < 4; i++)
    if (buttons[i].wasPressed || buttons[i].wasLongPressed) return true;
  return false;
}

ButtonAction getButtonAction() {
  updateButtons();
  // Long presses
  if (isLongPressed(0)) return ACTION_SPECIAL;  // UP long
  if (isLongPressed(1)) return ACTION_BACK;      // LEFT long
  if (isLongPressed(2)) return ACTION_SELECT;    // RIGHT long
  if (isLongPressed(3)) return ACTION_NONE;      // DOWN long
  // Short presses
  if (isPressed(0)) return ACTION_UP;
  if (isPressed(1)) return ACTION_LEFT;
  if (isPressed(2)) return ACTION_RIGHT;
  if (isPressed(3)) return ACTION_DOWN;
  return ACTION_NONE;
}

// ============================================================
//  STATE MACHINE
// ============================================================
enum SystemState {
  STATE_BOOT, STATE_WEATHER, STATE_MAIN_MENU, STATE_GAMES_MENU,
  STATE_STUDY_MENU, STATE_APPS_MENU, STATE_SETTINGS_MENU,
  STATE_STUDY_MCQ, STATE_STUDY_FACTS, STATE_STUDY_STATS,
  STATE_SHOP, STATE_TIMER, STATE_GOALS, STATE_COIN_FLIP,
  STATE_FORTUNE, STATE_MOOD, STATE_HEALTH, STATE_SCANNER,
  STATE_KEYBOARD, STATE_SLEEP_MODE,
  STATE_GAME_SPACE_WAR, STATE_GAME_CAR, STATE_GAME_SNAKE,
  STATE_GAME_PONG, STATE_GAME_MAZE, STATE_GAME_FLAPPY,
  STATE_GAME_2048, STATE_GAME_TETRIS,
  STATE_SETTINGS_BRIGHTNESS, STATE_SETTINGS_TIMEOUT,
  STATE_SETTINGS_WIFI, STATE_SETTINGS_RESET, STATE_HIGH_SCORES
};

SystemState g_state     = STATE_BOOT;
int         g_menuSel   = 0;
unsigned long g_menuTime = 0;
unsigned long g_lastActivity = 0;

// ============================================================
//  MENUS
// ============================================================
const char* MAIN_MENU[]     = {"Weather","Games","Study","Shop","Apps","Settings"};
const char* GAMES_MENU[]    = {"Space War","Car Game","Snake","Pong","Maze","Flappy","2048","Tetris","High Scores"};
const char* STUDY_MENU[]    = {"MCQ Quiz","Facts Mode","Study Stats"};
const char* APPS_MENU[]     = {"Timer","Goals","Coin Flip","Fortune","Mood","Health","WiFi Scan","Keyboard","Sleep Mode"};
const char* SETTINGS_MENU[] = {"Brightness","Menu Timeout","WiFi SSID","WiFi Pass","Reset All","About"};

#define MAIN_MENU_COUNT     6
#define GAMES_MENU_COUNT    9
#define STUDY_MENU_COUNT    3
#define APPS_MENU_COUNT     9
#define SETTINGS_MENU_COUNT 6

// ============================================================
//  MCQ SYSTEM – 100 UNIQUE QUESTIONS
// ============================================================
struct MCQ {
  const char* question;
  const char* optA;
  const char* optB;
  const char* optC;
  const char* optD;
  uint8_t     answer;
  const char* explain;
};

// Geography/Soil (Q0–Q11)
const char mq0[]   PROGMEM = "Which soil is ex-situ (transported)?";
const char mo0a[]  PROGMEM = "Alluvial";     const char mo0b[]  PROGMEM = "Black";
const char mo0c[]  PROGMEM = "Red";          const char mo0d[]  PROGMEM = "Laterite";
const char me0[]   PROGMEM = "Rivers transport alluvial from parent rock";

const char mq1[]   PROGMEM = "Khadar vs Bhangar?";
const char mo1a[]  PROGMEM = "Older, coarser";    const char mo1b[]  PROGMEM = "Newer,finer,fertile";
const char mo1c[]  PROGMEM = "Volcanic in-situ";  const char mo1d[]  PROGMEM = "Leached,acidic";
const char me1[]   PROGMEM = "Khadar=new alluvium, fine, flood-replenished";

const char mq2[]   PROGMEM = "What gives Red Soil its color?";
const char mo2a[]  PROGMEM = "Alumina";    const char mo2b[]  PROGMEM = "Magnesium";
const char mo2c[]  PROGMEM = "Iron oxide"; const char mo2d[]  PROGMEM = "Silica";
const char me2[]   PROGMEM = "Fe2O3 diffusion causes red color";

const char mq3[]   PROGMEM = "Which soil is 'self-ploughing'?";
const char mo3a[]  PROGMEM = "Alluvial"; const char mo3b[]  PROGMEM = "Red";
const char mo3c[]  PROGMEM = "Black";    const char mo3d[]  PROGMEM = "Laterite";
const char me3[]   PROGMEM = "Black soil cracks deeply when dry";

const char mq4[]   PROGMEM = "Laterite forms primarily by?";
const char mo4a[]  PROGMEM = "Leaching";    const char mo4b[]  PROGMEM = "River deposit";
const char mo4c[]  PROGMEM = "Volcanic";    const char mo4d[]  PROGMEM = "Wind";
const char me4[]   PROGMEM = "Heavy rain leaches silica/lime";

const char mq5[]   PROGMEM = "LIMCAP minerals in which soil?";
const char mo5a[]  PROGMEM = "Alluvial"; const char mo5b[]  PROGMEM = "Black";
const char mo5c[]  PROGMEM = "Red";      const char mo5d[]  PROGMEM = "Laterite";
const char me5[]   PROGMEM = "Lime,Iron,Mg,Ca,Alumina,Potash=Black";

const char mq6[]   PROGMEM = "Chambal ravines caused by?";
const char mo6a[]  PROGMEM = "Sheet erosion"; const char mo6b[]  PROGMEM = "Rill erosion";
const char mo6c[]  PROGMEM = "Gully erosion"; const char mo6d[]  PROGMEM = "Wind erosion";
const char me6[]   PROGMEM = "Gully erosion cuts deep channels";

const char mq7[]   PROGMEM = "Best conservation for steep hills?";
const char mo7a[]  PROGMEM = "Shelter belts";   const char mo7b[]  PROGMEM = "Terrace farming";
const char mo7c[]  PROGMEM = "Strip cropping";  const char mo7d[]  PROGMEM = "Monoculture";
const char me7[]   PROGMEM = "Terraces slow water runoff on slopes";

const char mq8[]   PROGMEM = "Most widespread soil in India?";
const char mo8a[]  PROGMEM = "Black"; const char mo8b[]  PROGMEM = "Red";
const char mo8c[]  PROGMEM = "Laterite"; const char mo8d[]  PROGMEM = "Alluvial";
const char me8[]   PROGMEM = "Alluvial covers 40% of India";

const char mq9[]   PROGMEM = "Critical organic element for fertility?";
const char mo9a[]  PROGMEM = "Silt";  const char mo9b[]  PROGMEM = "Kankar";
const char mo9c[]  PROGMEM = "Humus"; const char mo9d[]  PROGMEM = "Clay";
const char me9[]   PROGMEM = "Humus from decayed organic matter";

const char mq10[]  PROGMEM = "Wind erosion check in Rajasthan?";
const char mo10a[] PROGMEM = "Shelter belts"; const char mo10b[] PROGMEM = "Gully plugging";
const char mo10c[] PROGMEM = "Terrace farming"; const char mo10d[] PROGMEM = "Contour ploughing";
const char me10[]  PROGMEM = "Tree rows break wind force";

const char mq11[]  PROGMEM = "Red soil has poor water retention because?";
const char mo11a[] PROGMEM = "Highly clayey";  const char mo11b[] PROGMEM = "Rich in humus";
const char mo11c[] PROGMEM = "Porous, coarse"; const char mo11d[] PROGMEM = "Compacted";
const char me11[]  PROGMEM = "Coarse porous texture drains fast";

// History (Q12–Q23)
const char mq12[]  PROGMEM = "First INC President (1885)?";
const char mo12a[] PROGMEM = "W.C.Bonnerjee"; const char mo12b[] PROGMEM = "D.Naoroji";
const char mo12c[] PROGMEM = "A.O.Hume";      const char mo12d[] PROGMEM = "S.Banerjee";
const char me12[]  PROGMEM = "Bonnerjee presided,Bombay 1885";

const char mq13[]  PROGMEM = "Satya Shodhak Samaj founded by?";
const char mo13a[] PROGMEM = "R.R.M.Roy";  const char mo13b[] PROGMEM = "Jyotiba Phule";
const char mo13c[] PROGMEM = "S.Dayanand"; const char mo13d[] PROGMEM = "D.Naoroji";
const char me13[]  PROGMEM = "Phule 1873 against caste oppression";

const char mq14[]  PROGMEM = "Vernacular Press Act 1878 by?";
const char mo14a[] PROGMEM = "Lord Ripon";   const char mo14b[] PROGMEM = "Lord Lytton";
const char mo14c[] PROGMEM = "Lord Curzon";  const char mo14d[] PROGMEM = "Lord Dufferin";
const char me14[]  PROGMEM = "Lytton suppressed anti-British press";

const char mq15[]  PROGMEM = "Ilbert Bill 1883 under which Viceroy?";
const char mo15a[] PROGMEM = "Lord Lytton";    const char mo15b[] PROGMEM = "Lord Ripon";
const char mo15c[] PROGMEM = "Lord Curzon";    const char mo15d[] PROGMEM = "Lord Dalhousie";
const char me15[]  PROGMEM = "Ripon: Indian judges to try Europeans";

const char mq16[]  PROGMEM = "'Poverty & Un-British Rule' author?";
const char mo16a[] PROGMEM = "S.Banerjee";    const char mo16b[] PROGMEM = "D.Naoroji";
const char mo16c[] PROGMEM = "Bonnerjee";     const char mo16d[] PROGMEM = "A.O.Hume";
const char me16[]  PROGMEM = "Naoroji Drain of Wealth theory";

const char mq17[]  PROGMEM = "INC was founded in?";
const char mo17a[] PROGMEM = "1857"; const char mo17b[] PROGMEM = "1876";
const char mo17c[] PROGMEM = "1885"; const char mo17d[] PROGMEM = "1905";
const char me17[]  PROGMEM = "December 1885 by Hume and Indian leaders";

const char mq18[]  PROGMEM = "Brahmo Samaj founder?";
const char mo18a[] PROGMEM = "R.R.M.Roy";     const char mo18b[] PROGMEM = "J.Phule";
const char mo18c[] PROGMEM = "S.Vivekananda"; const char mo18d[] PROGMEM = "D.Tagore";
const char me18[]  PROGMEM = "Raja Ram Mohan Roy 1828";

const char mq19[]  PROGMEM = "Amrit Bazar Patrika escaped 1878 Act by?";
const char mo19a[] PROGMEM = "Closing down";    const char mo19b[] PROGMEM = "Filing petition";
const char mo19c[] PROGMEM = "Switch to English"; const char mo19d[] PROGMEM = "Move to Bombay";
const char me19[]  PROGMEM = "Converted to English overnight";

const char mq20[]  PROGMEM = "East India Assoc 1866 London by?";
const char mo20a[] PROGMEM = "Bonnerjee";  const char mo20b[] PROGMEM = "S.Banerjee";
const char mo20c[] PROGMEM = "D.Naoroji";  const char mo20d[] PROGMEM = "A.O.Hume";
const char me20[]  PROGMEM = "Naoroji raised Indian grievances in UK";

const char mq21[]  PROGMEM = "Phule book exposing caste oppression?";
const char mo21a[] PROGMEM = "Satyarth Prakash";   const char mo21b[] PROGMEM = "Ghulamgiri";
const char mo21c[] PROGMEM = "Discovery of India"; const char mo21d[] PROGMEM = "Poverty & Rule";
const char me21[]  PROGMEM = "Ghulamgiri 1873: caste=US slavery";

const char mq22[]  PROGMEM = "First INC session venue?";
const char mo22a[] PROGMEM = "Calcutta"; const char mo22b[] PROGMEM = "Madras";
const char mo22c[] PROGMEM = "Bombay";   const char mo22d[] PROGMEM = "Delhi";
const char me22[]  PROGMEM = "Gokuldas Tejpal College Bombay 1885";

const char mq23[]  PROGMEM = "Lytton reduced ICS age to prevent?";
const char mo23a[] PROGMEM = "Learning English";      const char mo23b[] PROGMEM = "Competing-Indians";
const char mo23c[] PROGMEM = "Joining army";          const char mo23d[] PROGMEM = "Voting";
const char me23[]  PROGMEM = "21->19: Indians couldn't travel to London";

// Genetics (Q24–Q34)
const char mq24[]  PROGMEM = "Mendel's plant for genetics?";
const char mo24a[] PROGMEM = "Sweet pea";        const char mo24b[] PROGMEM = "Garden pea";
const char mo24c[] PROGMEM = "Wild pea";         const char mo24d[] PROGMEM = "Pigeon pea";
const char me24[]  PROGMEM = "Pisum sativum: clear traits, self-poll";

const char mq25[]  PROGMEM = "F2 monohybrid phenotypic ratio?";
const char mo25a[] PROGMEM = "1:2:1"; const char mo25b[] PROGMEM = "3:1";
const char mo25c[] PROGMEM = "9:3:3:1"; const char mo25d[] PROGMEM = "1:1:1:1";
const char me25[]  PROGMEM = "Tt x Tt = 3 dominant : 1 recessive";

const char mq26[]  PROGMEM = "F2 dihybrid phenotypic ratio?";
const char mo26a[] PROGMEM = "3:1"; const char mo26b[] PROGMEM = "1:2:1";
const char mo26c[] PROGMEM = "9:3:3:1"; const char mo26d[] PROGMEM = "1:1";
const char me26[]  PROGMEM = "RrYy x RrYy = 9:3:3:1";

const char mq27[]  PROGMEM = "Alternative forms of gene called?";
const char mo27a[] PROGMEM = "Chromatid"; const char mo27b[] PROGMEM = "Autosome";
const char mo27c[] PROGMEM = "Allele";    const char mo27d[] PROGMEM = "Phenotype";
const char me27[]  PROGMEM = "Alleles e.g. T and t on same locus";

const char mq28[]  PROGMEM = "Sex of offspring determined by?";
const char mo28a[] PROGMEM = "Mother";  const char mo28b[] PROGMEM = "Father";
const char mo28c[] PROGMEM = "Both";    const char mo28d[] PROGMEM = "Temperature";
const char me28[]  PROGMEM = "Father: X and Y sperm decide sex";

const char mq29[]  PROGMEM = "Haemophilia is inherited as?";
const char mo29a[] PROGMEM = "Autosomal dom";  const char mo29b[] PROGMEM = "Autosomal rec";
const char mo29c[] PROGMEM = "X-linked rec";   const char mo29d[] PROGMEM = "Y-linked dom";
const char me29[]  PROGMEM = "Defective X allele, common in males";

const char mq30[]  PROGMEM = "Carrier female genotype haemophilia?";
const char mo30a[] PROGMEM = "XX";    const char mo30b[] PROGMEM = "XhX";
const char mo30c[] PROGMEM = "XhXh";  const char mo30d[] PROGMEM = "XY";
const char me30[]  PROGMEM = "One normal X + one defective X";

const char mq31[]  PROGMEM = "Y-linked (holandric) trait example?";
const char mo31a[] PROGMEM = "Colour blind"; const char mo31b[] PROGMEM = "Haemophilia";
const char mo31c[] PROGMEM = "Ear hair";     const char mo31d[] PROGMEM = "Albinism";
const char me31[]  PROGMEM = "Hypertrichosis: father to son only";

const char mq32[]  PROGMEM = "Outward appearance of organism?";
const char mo32a[] PROGMEM = "Genotype";     const char mo32b[] PROGMEM = "Phenotype";
const char mo32c[] PROGMEM = "Heterozygous"; const char mo32d[] PROGMEM = "Homozygous";
const char me32[]  PROGMEM = "Phenotype=visible; genotype=genetic";

const char mq33[]  PROGMEM = "Human autosome pairs in somatic cell?";
const char mo33a[] PROGMEM = "23 pairs"; const char mo33b[] PROGMEM = "22 pairs";
const char mo33c[] PROGMEM = "1 pair";   const char mo33d[] PROGMEM = "44 pairs";
const char me33[]  PROGMEM = "23 pairs total: 22 auto+1 sex";

const char mq34[]  PROGMEM = "Permanent inheritable DNA change?";
const char mo34a[] PROGMEM = "Heredity";    const char mo34b[] PROGMEM = "Segregation";
const char mo34c[] PROGMEM = "Mutation";    const char mo34d[] PROGMEM = "Assortment";
const char me34[]  PROGMEM = "Mutation alters DNA sequence";

// Chemistry (Q35–Q46)
const char mq35[]  PROGMEM = "Why is H3PO3 dibasic?";
const char mo35a[] PROGMEM = "One H atom";   const char mo35b[] PROGMEM = "2 H bonded to O";
const char mo35c[] PROGMEM = "Unstable liq"; const char mo35d[] PROGMEM = "Lacks phosphorus";
const char me35[]  PROGMEM = "H bonded to P cannot dissociate";

const char mq36[]  PROGMEM = "CH3COOH is monobasic because?";
const char mo36a[] PROGMEM = "Only COOH-H replaceable"; const char mo36b[] PROGMEM = "Insoluble";
const char mo36c[] PROGMEM = "Only acid salts";         const char mo36d[] PROGMEM = "Strong acid";
const char me36[]  PROGMEM = "Only COOH hydrogen ionizes";

const char mq37[]  PROGMEM = "PbO2 not a true base because?";
const char mo37a[] PROGMEM = "No acid reaction";  const char mo37b[] PROGMEM = "Gives extra Cl2";
const char mo37c[] PROGMEM = "Turns litmus red";  const char mo37d[] PROGMEM = "Soluble";
const char me37[]  PROGMEM = "True base: salt+water ONLY";

const char mq38[]  PROGMEM = "Methyl orange in acid turns?";
const char mo38a[] PROGMEM = "Yellow"; const char mo38b[] PROGMEM = "Orange";
const char mo38c[] PROGMEM = "Red";    const char mo38d[] PROGMEM = "Pink";
const char me38[]  PROGMEM = "Red in acid, yellow in base";

const char mq39[]  PROGMEM = "Anhydrous FeCl3 prepared by?";
const char mo39a[] PROGMEM = "NaCl method";    const char mo39b[] PROGMEM = "FeCl2 method";
const char mo39c[] PROGMEM = "Fe + Cl2 direct"; const char mo39d[] PROGMEM = "CuSO4 method";
const char me39[]  PROGMEM = "Heated Fe + dry Cl2 gas only";

const char mq40[]  PROGMEM = "Na2CO3.10H2O in dry air undergoes?";
const char mo40a[] PROGMEM = "Deliquescence"; const char mo40b[] PROGMEM = "Efflorescence";
const char mo40c[] PROGMEM = "Sublimation";   const char mo40d[] PROGMEM = "Hydration";
const char me40[]  PROGMEM = "Loses water of crystallization";

const char mq41[]  PROGMEM = "NaOH absorbs moisture and dissolves:";
const char mo41a[] PROGMEM = "Hygroscopic";   const char mo41b[] PROGMEM = "Efflorescence";
const char mo41c[] PROGMEM = "Deliquescence"; const char mo41d[] PROGMEM = "Sublimation";
const char me41[]  PROGMEM = "Deliquescent: absorbs AND dissolves";

const char mq42[]  PROGMEM = "Conc H2SO4 is drying agent because?";
const char mo42a[] PROGMEM = "Deliquescent";  const char mo42b[] PROGMEM = "Efflorescent";
const char mo42c[] PROGMEM = "Hygroscopic";   const char mo42d[] PROGMEM = "Reactive";
const char me42[]  PROGMEM = "Hygroscopic: absorbs without dissolving";

const char mq43[]  PROGMEM = "CO2 from salt turns limewater milky?";
const char mo43a[] PROGMEM = "Sulphite";  const char mo43b[] PROGMEM = "Carbonate";
const char mo43c[] PROGMEM = "Chloride";  const char mo43d[] PROGMEM = "Sulphate";
const char me43[]  PROGMEM = "CO2 from carbonates+HCl";

const char mq44[]  PROGMEM = "pH of strongly basic solution?";
const char mo44a[] PROGMEM = "2"; const char mo44b[] PROGMEM = "7";
const char mo44c[] PROGMEM = "5"; const char mo44d[] PROGMEM = "13";
const char me44[]  PROGMEM = "pH 13-14 = strongly basic";

const char mq45[]  PROGMEM = "NO2 is mixed acid anhydride giving?";
const char mo45a[] PROGMEM = "HNO3 only"; const char mo45b[] PROGMEM = "HNO2 only";
const char mo45c[] PROGMEM = "HNO2+HNO3"; const char mo45d[] PROGMEM = "H2O+N2";
const char me45[]  PROGMEM = "One oxide produces two different acids";

const char mq46[]  PROGMEM = "NaCl in lab by?";
const char mo46a[] PROGMEM = "Lead sulfate";  const char mo46b[] PROGMEM = "Titration/neutral";
const char mo46c[] PROGMEM = "FeCl3 method";  const char mo46d[] PROGMEM = "Precipitation";
const char me46[]  PROGMEM = "NaOH + HCl titration";

// Neurology (Q47–Q99)
const char mq47[]  PROGMEM = "Structural/functional unit of NS?";
const char mo47a[] PROGMEM = "Neuron";  const char mo47b[] PROGMEM = "Nephron";
const char mo47c[] PROGMEM = "Cyton";   const char mo47d[] PROGMEM = "Axon";
const char me47[]  PROGMEM = "Neuron: receives and transmits impulses";

const char mq48[]  PROGMEM = "Cytoplasmic matrix of neuron body?";
const char mo48a[] PROGMEM = "Axoplasm";   const char mo48b[] PROGMEM = "Neuroplasm";
const char mo48c[] PROGMEM = "Sarcoplasm"; const char mo48d[] PROGMEM = "Nucleoplasm";
const char me48[]  PROGMEM = "Neuroplasm in cyton of nerve cell";

const char mq49[]  PROGMEM = "Ribosomes in neuron cell body called?";
const char mo49a[] PROGMEM = "Mitochondria"; const char mo49b[] PROGMEM = "Centrosomes";
const char mo49c[] PROGMEM = "Nissl granules"; const char mo49d[] PROGMEM = "Neurofibrils";
const char me49[]  PROGMEM = "Nissl granules for protein synthesis";

const char mq50[]  PROGMEM = "Primary function of dendrites?";
const char mo50a[] PROGMEM = "Carry away";     const char mo50b[] PROGMEM = "Receive toward";
const char mo50c[] PROGMEM = "Insulate axon";  const char mo50d[] PROGMEM = "Release neurotransmitter";
const char me50[]  PROGMEM = "Dendrites receive impulses to cyton";

const char mq51[]  PROGMEM = "Fatty insulating layer around axon?";
const char mo51a[] PROGMEM = "Axolemma";    const char mo51b[] PROGMEM = "Neurolemma";
const char mo51c[] PROGMEM = "Myelin sheath"; const char mo51d[] PROGMEM = "Meninges";
const char me51[]  PROGMEM = "Myelin insulates, speeds impulse";

const char mq52[]  PROGMEM = "Gaps in myelin between Schwann cells?";
const char mo52a[] PROGMEM = "Synaptic cleft"; const char mo52b[] PROGMEM = "Dendritic nodes";
const char mo52c[] PROGMEM = "Nodes of Ranvier"; const char mo52d[] PROGMEM = "Axonal gaps";
const char me52[]  PROGMEM = "Ranvier nodes: saltatory conduction";

const char mq53[]  PROGMEM = "Outer Schwann cell membrane?";
const char mo53a[] PROGMEM = "Axolemma";    const char mo53b[] PROGMEM = "Neurolemma";
const char mo53c[] PROGMEM = "Pia mater";   const char mo53d[] PROGMEM = "Dura mater";
const char me53[]  PROGMEM = "Neurolemma: outermost nucleated sheath";

const char mq54[]  PROGMEM = "Cluster of nerve cell bodies outside CNS?";
const char mo54a[] PROGMEM = "Ganglion"; const char mo54b[] PROGMEM = "Nerve";
const char mo54c[] PROGMEM = "Plexus";   const char mo54d[] PROGMEM = "Synapse";
const char me54[]  PROGMEM = "Ganglion: distinct cluster in PNS";

const char mq55[]  PROGMEM = "Junction between axon and next dendrite?";
const char mo55a[] PROGMEM = "Ganglion";    const char mo55b[] PROGMEM = "Synapse";
const char mo55c[] PROGMEM = "Ranvier node"; const char mo55d[] PROGMEM = "NMJ";
const char me55[]  PROGMEM = "Synapse: gap bridged by neurotransmitters";

const char mq56[]  PROGMEM = "Parasympathetic neurotransmitter?";
const char mo56a[] PROGMEM = "Noradrenaline"; const char mo56b[] PROGMEM = "Adrenaline";
const char mo56c[] PROGMEM = "Acetylcholine"; const char mo56d[] PROGMEM = "Dopamine";
const char me56[]  PROGMEM = "Acetylcholine: primary parasympathetic";

const char mq57[]  PROGMEM = "Automatic involuntary response?";
const char mo57a[] PROGMEM = "Voluntary";   const char mo57b[] PROGMEM = "Reflex action";
const char mo57c[] PROGMEM = "Habit";       const char mo57d[] PROGMEM = "Cerebral";
const char me57[]  PROGMEM = "Reflex: protects without conscious thought";

const char mq58[]  PROGMEM = "Correct reflex arc pathway?";
const char mo58a[] PROGMEM = "Stimulus>Recept>Sens>Cord>Mot>Effec";
const char mo58b[] PROGMEM = "Stimulus>Recept>Mot>Cord>Sens>Effec";
const char mo58c[] PROGMEM = "Stimulus>Effec>Sens>Cord>Mot>Recept";
const char mo58d[] PROGMEM = "Stimulus>Recept>Sens>Brain>Mot>Effec";
const char me58[]  PROGMEM = "Receptor>Sensory>Cord>Motor>Effector";

const char mq59[]  PROGMEM = "3 protective membranes around brain?";
const char mo59a[] PROGMEM = "Pleura";       const char mo59b[] PROGMEM = "Meninges";
const char mo59c[] PROGMEM = "Pericardium";  const char mo59d[] PROGMEM = "Peritoneum";
const char me59[]  PROGMEM = "Dura mater,Arachnoid,Pia mater";

const char mq60[]  PROGMEM = "Meninges order outer to inner?";
const char mo60a[] PROGMEM = "Arachn>Dura>Pia"; const char mo60b[] PROGMEM = "Dura>Pia>Arachn";
const char mo60c[] PROGMEM = "Dura>Arachn>Pia"; const char mo60d[] PROGMEM = "Pia>Arachn>Dura";
const char me60[]  PROGMEM = "DAP: Dura,Arachnoid,Pia mater";

const char mq61[]  PROGMEM = "CSF main function?";
const char mo61a[] PROGMEM = "Shock absorber"; const char mo61b[] PROGMEM = "Conducts impulse";
const char mo61c[] PROGMEM = "Secretes insulin"; const char mo61d[] PROGMEM = "Makes RBCs";
const char me61[]  PROGMEM = "CSF cushions brain, exchanges nutrients";

const char mq62[]  PROGMEM = "Ridges/elevations on cerebral cortex?";
const char mo62a[] PROGMEM = "Sulci";     const char mo62b[] PROGMEM = "Gyri";
const char mo62c[] PROGMEM = "Fissures";  const char mo62d[] PROGMEM = "Ventricles";
const char me62[]  PROGMEM = "Gyri increase surface area for neurons";

const char mq63[]  PROGMEM = "Sheet connecting cerebral hemispheres?";
const char mo63a[] PROGMEM = "Corpus callosum"; const char mo63b[] PROGMEM = "Pons";
const char mo63c[] PROGMEM = "Medulla";          const char mo63d[] PROGMEM = "Thalamus";
const char me63[]  PROGMEM = "Corpus callosum: 200M+ myelinated fibers";

const char mq64[]  PROGMEM = "Seat of intelligence and memory?";
const char mo64a[] PROGMEM = "Cerebellum"; const char mo64b[] PROGMEM = "Medulla";
const char mo64c[] PROGMEM = "Cerebrum";   const char mo64d[] PROGMEM = "Pons";
const char me64[]  PROGMEM = "Cerebrum: largest, center of consciousness";

const char mq65[]  PROGMEM = "Balance/coordination controlled by?";
const char mo65a[] PROGMEM = "Cerebrum";  const char mo65b[] PROGMEM = "Thalamus";
const char mo65c[] PROGMEM = "Cerebellum"; const char mo65d[] PROGMEM = "Medulla";
const char me65[]  PROGMEM = "Cerebellum: coordination and balance";

const char mq66[]  PROGMEM = "Controls heartbeat and breathing?";
const char mo66a[] PROGMEM = "Cerebrum";   const char mo66b[] PROGMEM = "Cerebellum";
const char mo66c[] PROGMEM = "Pons";       const char mo66d[] PROGMEM = "Medulla oblongata";
const char me66[]  PROGMEM = "Medulla: vital cardiac/respiratory center";

const char mq67[]  PROGMEM = "Destroying medulla causes?";
const char mo67a[] PROGMEM = "Immediate death"; const char mo67b[] PROGMEM = "Memory loss";
const char mo67c[] PROGMEM = "Balance loss";    const char mo67d[] PROGMEM = "Blindness";
const char me67[]  PROGMEM = "Medulla houses cardiac/respiratory: death";

const char mq68[]  PROGMEM = "Controls hunger, thirst, temperature?";
const char mo68a[] PROGMEM = "Thalamus";      const char mo68b[] PROGMEM = "Hypothalamus";
const char mo68c[] PROGMEM = "Cerebellum";    const char mo68d[] PROGMEM = "Pons";
const char me68[]  PROGMEM = "Hypothalamus: master of homeostasis";

const char mq69[]  PROGMEM = "Grey matter location in BRAIN?";
const char mo69a[] PROGMEM = "Outside/cortex"; const char mo69b[] PROGMEM = "Inside";
const char mo69c[] PROGMEM = "Mixed";           const char mo69d[] PROGMEM = "Absent";
const char me69[]  PROGMEM = "Brain: grey outside, white inside";

const char mq70[]  PROGMEM = "Grey matter in SPINAL CORD?";
const char mo70a[] PROGMEM = "Outside"; const char mo70b[] PROGMEM = "Inside H-shaped";
const char mo70c[] PROGMEM = "Mixed";   const char mo70d[] PROGMEM = "Only white";
const char me70[]  PROGMEM = "Spinal: inner H-shaped grey, outer white";

const char mq71[]  PROGMEM = "Human PNS cranial+spinal nerve pairs?";
const char mo71a[] PROGMEM = "10 and 30"; const char mo71b[] PROGMEM = "12 and 31";
const char mo71c[] PROGMEM = "31 and 12"; const char mo71d[] PROGMEM = "12 and 12";
const char me71[]  PROGMEM = "12 cranial + 31 spinal pairs";

const char mq72[]  PROGMEM = "Fight-or-flight division of ANS?";
const char mo72a[] PROGMEM = "Sympathetic";     const char mo72b[] PROGMEM = "Parasympathetic";
const char mo72c[] PROGMEM = "Somatic NS";      const char mo72d[] PROGMEM = "Central NS";
const char me72[]  PROGMEM = "Sympathetic: accelerates heart in emergency";

const char mq73[]  PROGMEM = "'Rest and digest' ANS division?";
const char mo73a[] PROGMEM = "Sympathetic";     const char mo73b[] PROGMEM = "Parasympathetic";
const char mo73c[] PROGMEM = "Somatic NS";      const char mo73d[] PROGMEM = "Sensory NS";
const char me73[]  PROGMEM = "Parasympathetic: conserves energy";

const char mq74[]  PROGMEM = "Pavlov's dog salivating at bell?";
const char mo74a[] PROGMEM = "Natural reflex";    const char mo74b[] PROGMEM = "Conditioned reflex";
const char mo74c[] PROGMEM = "Voluntary move";    const char mo74d[] PROGMEM = "Spinal reflex";
const char me74[]  PROGMEM = "Conditioned: learned by association";

const char mq75[]  PROGMEM = "Example of inborn (natural) reflex?";
const char mo75a[] PROGMEM = "Apply car brakes"; const char mo75b[] PROGMEM = "Blink at dust";
const char mo75c[] PROGMEM = "Type keyboard";    const char mo75d[] PROGMEM = "Play instrument";
const char me75[]  PROGMEM = "Blinking: inborn protective reflex";

const char mq76[]  PROGMEM = "Sensory neuron also called?";
const char mo76a[] PROGMEM = "Efferent";    const char mo76b[] PROGMEM = "Afferent";
const char mo76c[] PROGMEM = "Association"; const char mo76d[] PROGMEM = "Motor";
const char me76[]  PROGMEM = "Afferent: carries toward CNS";

const char mq77[]  PROGMEM = "Motor neuron also called?";
const char mo77a[] PROGMEM = "Afferent"; const char mo77b[] PROGMEM = "Efferent";
const char mo77c[] PROGMEM = "Relay";    const char mo77d[] PROGMEM = "Interneuron";
const char me77[]  PROGMEM = "Efferent: carries away from CNS";

const char mq78[]  PROGMEM = "Sympathetic neurotransmitter?";
const char mo78a[] PROGMEM = "Acetylcholine"; const char mo78b[] PROGMEM = "Noradrenaline";
const char mo78c[] PROGMEM = "Serotonin";     const char mo78d[] PROGMEM = "GABA";
const char me78[]  PROGMEM = "Noradrenaline: sympathetic postganglionic";

const char mq79[]  PROGMEM = "Alcohol impairs balance: which region?";
const char mo79a[] PROGMEM = "Cerebrum";   const char mo79b[] PROGMEM = "Pons";
const char mo79c[] PROGMEM = "Cerebellum"; const char mo79d[] PROGMEM = "Medulla";
const char me79[]  PROGMEM = "Alcohol depresses cerebellum";

const char mq80[]  PROGMEM = "Neuron part lacking Nissl granules?";
const char mo80a[] PROGMEM = "Cyton";    const char mo80b[] PROGMEM = "Axon";
const char mo80c[] PROGMEM = "Dendrite"; const char mo80d[] PROGMEM = "Cell body";
const char me80[]  PROGMEM = "Nissl absent in axon only";

const char mq81[]  PROGMEM = "Forebrain structures?";
const char mo81a[] PROGMEM = "Cerebrum+Diencephalon"; const char mo81b[] PROGMEM = "Cerebellum+Pons";
const char mo81c[] PROGMEM = "Medulla+Pons";           const char mo81d[] PROGMEM = "Cerebrum+Cerebellum";
const char me81[]  PROGMEM = "Forebrain: Cerebrum, olfactory, Diencephalon";

const char mq82[]  PROGMEM = "Hindbrain structures?";
const char mo82a[] PROGMEM = "Cerebrum+Pons+Med";      const char mo82b[] PROGMEM = "Cerebellum+Pons+Med";
const char mo82c[] PROGMEM = "Thalamus+Hypo+Pons";     const char mo82d[] PROGMEM = "OpticLobes+Cerebrum";
const char me82[]  PROGMEM = "Hindbrain: Cerebellum, Pons, Medulla";

const char mq83[]  PROGMEM = "Impulse ENTERS spinal cord via?";
const char mo83a[] PROGMEM = "Ventral root"; const char mo83b[] PROGMEM = "Dorsal root";
const char mo83c[] PROGMEM = "White column"; const char mo83d[] PROGMEM = "Central canal";
const char me83[]  PROGMEM = "Sensory fibers enter via dorsal root";

const char mq84[]  PROGMEM = "Motor fibers EMERGE from spinal cord?";
const char mo84a[] PROGMEM = "Dorsal root";  const char mo84b[] PROGMEM = "Ventral root";
const char mo84c[] PROGMEM = "Meninges";     const char mo84d[] PROGMEM = "Grey commissure";
const char me84[]  PROGMEM = "Motor exit via ventral (anterior) root";

const char mq85[]  PROGMEM = "Relay (association) neuron links what?";
const char mo85a[] PROGMEM = "Two relay neurons"; const char mo85b[] PROGMEM = "Two afferent";
const char mo85c[] PROGMEM = "Sensory to motor";  const char mo85d[] PROGMEM = "Two receptors";
const char me85[]  PROGMEM = "Interneuron within CNS: sensory to motor";

const char mq86[]  PROGMEM = "Synapse is one-way because?";
const char mo86a[] PROGMEM = "Myelin blocks back";   const char mo86b[] PROGMEM = "Receptors postsynaptic only";
const char mo86c[] PROGMEM = "Axons longer";          const char mo86d[] PROGMEM = "CSF forces direction";
const char me86[]  PROGMEM = "Vesicles pre, receptors post only";

const char mq87[]  PROGMEM = "Example of voluntary action?";
const char mo87a[] PROGMEM = "Sneezing";   const char mo87b[] PROGMEM = "Knee-jerk";
const char mo87c[] PROGMEM = "Writing exam"; const char mo87d[] PROGMEM = "Peristalsis";
const char me87[]  PROGMEM = "Writing: consciously decided by cortex";

const char mq88[]  PROGMEM = "Cushions brain inside skull?";
const char mo88a[] PROGMEM = "Neuroplasm"; const char mo88b[] PROGMEM = "CSF";
const char mo88c[] PROGMEM = "Myelin";     const char mo88d[] PROGMEM = "Corpus callosum";
const char me88[]  PROGMEM = "CSF: liquid cushion for brain";

const char mq89[]  PROGMEM = "Adult human spinal cord length?";
const char mo89a[] PROGMEM = "43-45 cm"; const char mo89b[] PROGMEM = "10-12 cm";
const char mo89c[] PROGMEM = "70-75 cm"; const char mo89d[] PROGMEM = "150 cm";
const char me89[]  PROGMEM = "~43-45 cm from medulla down";

const char mq90[]  PROGMEM = "Bony box protecting human brain?";
const char mo90a[] PROGMEM = "Vertebral col"; const char mo90b[] PROGMEM = "Ribcage";
const char mo90c[] PROGMEM = "Cranium";       const char mo90d[] PROGMEM = "Sternum";
const char me90[]  PROGMEM = "Cranium: 8 bones forming skull vault";

const char mq91[]  PROGMEM = "Grooves on cerebral cortex?";
const char mo91a[] PROGMEM = "Gyri";    const char mo91b[] PROGMEM = "Sulci";
const char mo91c[] PROGMEM = "Ganglia"; const char mo91d[] PROGMEM = "Vesicles";
const char me91[]  PROGMEM = "Sulci: shallow grooves separating gyri";

const char mq92[]  PROGMEM = "Bridge connecting cerebellar lobes?";
const char mo92a[] PROGMEM = "Corpus callosum"; const char mo92b[] PROGMEM = "Pons";
const char mo92c[] PROGMEM = "Thalamus";         const char mo92d[] PROGMEM = "Hypothalamus";
const char me92[]  PROGMEM = "Pons carries fiber tracts between regions";

const char mq93[]  PROGMEM = "Cerebellar damage result?";
const char mo93a[] PROGMEM = "Paralysis";    const char mo93b[] PROGMEM = "Loss of coordination";
const char mo93c[] PROGMEM = "Memory loss";  const char mo93d[] PROGMEM = "Blindness";
const char me93[]  PROGMEM = "Movement OK but coordination/timing lost";

const char mq94[]  PROGMEM = "Hypothalamus connects NS to?";
const char mo94a[] PROGMEM = "Circulatory";  const char mo94b[] PROGMEM = "Endocrine";
const char mo94c[] PROGMEM = "Digestive";    const char mo94d[] PROGMEM = "Lymphatic";
const char me94[]  PROGMEM = "Controls pituitary: NS to hormones";

const char mq95[]  PROGMEM = "Micturition refers to?";
const char mo95a[] PROGMEM = "Defecation"; const char mo95b[] PROGMEM = "Urinating";
const char mo95c[] PROGMEM = "Deamination"; const char mo95d[] PROGMEM = "Diapedesis";
const char me95[]  PROGMEM = "Micturition: urinating, spinal reflexes";

const char mq96[]  PROGMEM = "ANS neurons to reach effectors?";
const char mo96a[] PROGMEM = "One";   const char mo96b[] PROGMEM = "Two";
const char mo96c[] PROGMEM = "Three"; const char mo96d[] PROGMEM = "Four";
const char me96[]  PROGMEM = "Pre and postganglionic at ganglion";

const char mq97[]  PROGMEM = "Dorsal root damage causes?";
const char mo97a[] PROGMEM = "Paralysis only"; const char mo97b[] PROGMEM = "Loss sensation only";
const char mo97c[] PROGMEM = "Both";           const char mo97d[] PROGMEM = "No effect";
const char me97[]  PROGMEM = "Dorsal=sensory: damage=no sensation";

const char mq98[]  PROGMEM = "Ventral root damage causes?";
const char mo98a[] PROGMEM = "Loss sensation"; const char mo98b[] PROGMEM = "Paralysis only";
const char mo98c[] PROGMEM = "Both";           const char mo98d[] PROGMEM = "No effect";
const char me98[]  PROGMEM = "Ventral=motor: damage=paralysis";

const char mq99[]  PROGMEM = "Alleles are at?";
const char mo99a[] PROGMEM = "Different chromo"; const char mo99b[] PROGMEM = "Same locus,homolog";
const char mo99c[] PROGMEM = "Random positions"; const char mo99d[] PROGMEM = "Sex chromo only";
const char me99[]  PROGMEM = "Same locus on homologous chromosomes";

const MCQ MCQ_TABLE[] PROGMEM = {
  {mq0,mo0a,mo0b,mo0c,mo0d,0,me0},   {mq1,mo1a,mo1b,mo1c,mo1d,1,me1},
  {mq2,mo2a,mo2b,mo2c,mo2d,2,me2},   {mq3,mo3a,mo3b,mo3c,mo3d,2,me3},
  {mq4,mo4a,mo4b,mo4c,mo4d,0,me4},   {mq5,mo5a,mo5b,mo5c,mo5d,1,me5},
  {mq6,mo6a,mo6b,mo6c,mo6d,2,me6},   {mq7,mo7a,mo7b,mo7c,mo7d,1,me7},
  {mq8,mo8a,mo8b,mo8c,mo8d,3,me8},   {mq9,mo9a,mo9b,mo9c,mo9d,2,me9},
  {mq10,mo10a,mo10b,mo10c,mo10d,0,me10}, {mq11,mo11a,mo11b,mo11c,mo11d,2,me11},
  {mq12,mo12a,mo12b,mo12c,mo12d,0,me12}, {mq13,mo13a,mo13b,mo13c,mo13d,1,me13},
  {mq14,mo14a,mo14b,mo14c,mo14d,1,me14}, {mq15,mo15a,mo15b,mo15c,mo15d,1,me15},
  {mq16,mo16a,mo16b,mo16c,mo16d,1,me16}, {mq17,mo17a,mo17b,mo17c,mo17d,2,me17},
  {mq18,mo18a,mo18b,mo18c,mo18d,0,me18}, {mq19,mo19a,mo19b,mo19c,mo19d,2,me19},
  {mq20,mo20a,mo20b,mo20c,mo20d,2,me20}, {mq21,mo21a,mo21b,mo21c,mo21d,1,me21},
  {mq22,mo22a,mo22b,mo22c,mo22d,2,me22}, {mq23,mo23a,mo23b,mo23c,mo23d,1,me23},
  {mq24,mo24a,mo24b,mo24c,mo24d,1,me24}, {mq25,mo25a,mo25b,mo25c,mo25d,1,me25},
  {mq26,mo26a,mo26b,mo26c,mo26d,2,me26}, {mq27,mo27a,mo27b,mo27c,mo27d,2,me27},
  {mq28,mo28a,mo28b,mo28c,mo28d,1,me28}, {mq29,mo29a,mo29b,mo29c,mo29d,2,me29},
  {mq30,mo30a,mo30b,mo30c,mo30d,1,me30}, {mq31,mo31a,mo31b,mo31c,mo31d,2,me31},
  {mq32,mo32a,mo32b,mo32c,mo32d,1,me32}, {mq33,mo33a,mo33b,mo33c,mo33d,1,me33},
  {mq34,mo34a,mo34b,mo34c,mo34d,2,me34}, {mq35,mo35a,mo35b,mo35c,mo35d,1,me35},
  {mq36,mo36a,mo36b,mo36c,mo36d,0,me36}, {mq37,mo37a,mo37b,mo37c,mo37d,1,me37},
  {mq38,mo38a,mo38b,mo38c,mo38d,2,me38}, {mq39,mo39a,mo39b,mo39c,mo39d,2,me39},
  {mq40,mo40a,mo40b,mo40c,mo40d,1,me40}, {mq41,mo41a,mo41b,mo41c,mo41d,2,me41},
  {mq42,mo42a,mo42b,mo42c,mo42d,2,me42}, {mq43,mo43a,mo43b,mo43c,mo43d,1,me43},
  {mq44,mo44a,mo44b,mo44c,mo44d,3,me44}, {mq45,mo45a,mo45b,mo45c,mo45d,2,me45},
  {mq46,mo46a,mo46b,mo46c,mo46d,1,me46}, {mq47,mo47a,mo47b,mo47c,mo47d,0,me47},
  {mq48,mo48a,mo48b,mo48c,mo48d,1,me48}, {mq49,mo49a,mo49b,mo49c,mo49d,2,me49},
  {mq50,mo50a,mo50b,mo50c,mo50d,1,me50}, {mq51,mo51a,mo51b,mo51c,mo51d,2,me51},
  {mq52,mo52a,mo52b,mo52c,mo52d,2,me52}, {mq53,mo53a,mo53b,mo53c,mo53d,1,me53},
  {mq54,mo54a,mo54b,mo54c,mo54d,0,me54}, {mq55,mo55a,mo55b,mo55c,mo55d,1,me55},
  {mq56,mo56a,mo56b,mo56c,mo56d,2,me56}, {mq57,mo57a,mo57b,mo57c,mo57d,1,me57},
  {mq58,mo58a,mo58b,mo58c,mo58d,0,me58}, {mq59,mo59a,mo59b,mo59c,mo59d,1,me59},
  {mq60,mo60a,mo60b,mo60c,mo60d,2,me60}, {mq61,mo61a,mo61b,mo61c,mo61d,0,me61},
  {mq62,mo62a,mo62b,mo62c,mo62d,1,me62}, {mq63,mo63a,mo63b,mo63c,mo63d,0,me63},
  {mq64,mo64a,mo64b,mo64c,mo64d,2,me64}, {mq65,mo65a,mo65b,mo65c,mo65d,2,me65},
  {mq66,mo66a,mo66b,mo66c,mo66d,3,me66}, {mq67,mo67a,mo67b,mo67c,mo67d,0,me67},
  {mq68,mo68a,mo68b,mo68c,mo68d,1,me68}, {mq69,mo69a,mo69b,mo69c,mo69d,0,me69},
  {mq70,mo70a,mo70b,mo70c,mo70d,1,me70}, {mq71,mo71a,mo71b,mo71c,mo71d,1,me71},
  {mq72,mo72a,mo72b,mo72c,mo72d,0,me72}, {mq73,mo73a,mo73b,mo73c,mo73d,1,me73},
  {mq74,mo74a,mo74b,mo74c,mo74d,1,me74}, {mq75,mo75a,mo75b,mo75c,mo75d,1,me75},
  {mq76,mo76a,mo76b,mo76c,mo76d,1,me76}, {mq77,mo77a,mo77b,mo77c,mo77d,1,me77},
  {mq78,mo78a,mo78b,mo78c,mo78d,1,me78}, {mq79,mo79a,mo79b,mo79c,mo79d,2,me79},
  {mq80,mo80a,mo80b,mo80c,mo80d,1,me80}, {mq81,mo81a,mo81b,mo81c,mo81d,0,me81},
  {mq82,mo82a,mo82b,mo82c,mo82d,1,me82}, {mq83,mo83a,mo83b,mo83c,mo83d,1,me83},
  {mq84,mo84a,mo84b,mo84c,mo84d,1,me84}, {mq85,mo85a,mo85b,mo85c,mo85d,2,me85},
  {mq86,mo86a,mo86b,mo86c,mo86d,1,me86}, {mq87,mo87a,mo87b,mo87c,mo87d,2,me87},
  {mq88,mo88a,mo88b,mo88c,mo88d,1,me88}, {mq89,mo89a,mo89b,mo89c,mo89d,0,me89},
  {mq90,mo90a,mo90b,mo90c,mo90d,2,me90}, {mq91,mo91a,mo91b,mo91c,mo91d,1,me91},
  {mq92,mo92a,mo92b,mo92c,mo92d,1,me92}, {mq93,mo93a,mo93b,mo93c,mo93d,1,me93},
  {mq94,mo94a,mo94b,mo94c,mo94d,1,me94}, {mq95,mo95a,mo95b,mo95c,mo95d,1,me95},
  {mq96,mo96a,mo96b,mo96c,mo96d,1,me96}, {mq97,mo97a,mo97b,mo97c,mo97d,1,me97},
  {mq98,mo98a,mo98b,mo98c,mo98d,1,me98}, {mq99,mo99a,mo99b,mo99c,mo99d,1,me99}
};
#define MCQ_COUNT 100

// ============================================================
//  FACTS (50 unique)
// ============================================================
struct Fact { const char* fact; const char* detail; };

const char ff0[]  PROGMEM = "Alluvial: ex-situ soil";     const char fd0[]  PROGMEM = "Transported by rivers";
const char ff1[]  PROGMEM = "Khadar is fertile";          const char fd1[]  PROGMEM = "New alluvium, fine";
const char ff2[]  PROGMEM = "Black soil self-ploughs";    const char fd2[]  PROGMEM = "Cracks when dry";
const char ff3[]  PROGMEM = "Red = iron oxide";           const char fd3[]  PROGMEM = "Fe2O3 diffusion";
const char ff4[]  PROGMEM = "Laterite = leached";         const char fd4[]  PROGMEM = "Acidic, tropical";
const char ff5[]  PROGMEM = "LIMCAP = Black soil";        const char fd5[]  PROGMEM = "Cotton crop best";
const char ff6[]  PROGMEM = "Gully erosion = badland";    const char fd6[]  PROGMEM = "Chambal ravines";
const char ff7[]  PROGMEM = "Terrace farming";            const char fd7[]  PROGMEM = "Slows hill runoff";
const char ff8[]  PROGMEM = "Alluvial 40% India";         const char fd8[]  PROGMEM = "Most widespread";
const char ff9[]  PROGMEM = "Humus = fertility";          const char fd9[]  PROGMEM = "Organic matter key";
const char ff10[] PROGMEM = "INC founded 1885";           const char fd10[] PROGMEM = "Hume + Indian leaders";
const char ff11[] PROGMEM = "Drain of Wealth";            const char fd11[] PROGMEM = "Naoroji's theory";
const char ff12[] PROGMEM = "Vernacular Act 1878";        const char fd12[] PROGMEM = "Lytton suppression";
const char ff13[] PROGMEM = "Ilbert Bill 1883";           const char fd13[] PROGMEM = "Indian judges EU";
const char ff14[] PROGMEM = "Brahmo Samaj 1828";          const char fd14[] PROGMEM = "Raja Ram Mohan Roy";
const char ff15[] PROGMEM = "Phule 1873 Samaj";           const char fd15[] PROGMEM = "Anti-caste movement";
const char ff16[] PROGMEM = "Ghulamgiri = slavery";       const char fd16[] PROGMEM = "Phule's book 1873";
const char ff17[] PROGMEM = "Mendel: Pisum sativum";      const char fd17[] PROGMEM = "Garden pea genetics";
const char ff18[] PROGMEM = "F2 mono = 3:1";              const char fd18[] PROGMEM = "Dom:Rec phenotype";
const char ff19[] PROGMEM = "Alleles: same locus";        const char fd19[] PROGMEM = "Homologous chromo";
const char ff20[] PROGMEM = "Father decides sex";         const char fd20[] PROGMEM = "X or Y sperm";
const char ff21[] PROGMEM = "Haemophilia: X-linked";      const char fd21[] PROGMEM = "Recessive disorder";
const char ff22[] PROGMEM = "Y-linked: father->son";      const char fd22[] PROGMEM = "Holandric traits";
const char ff23[] PROGMEM = "Mutation = DNA change";      const char fd23[] PROGMEM = "Permanent, heritable";
const char ff24[] PROGMEM = "H3PO3: dibasic acid";        const char fd24[] PROGMEM = "One H on P: no ionize";
const char ff25[] PROGMEM = "Methyl orange: acid=red";    const char fd25[] PROGMEM = "Base=yellow";
const char ff26[] PROGMEM = "FeCl3: Fe + dry Cl2";        const char fd26[] PROGMEM = "Anhydrous method";
const char ff27[] PROGMEM = "Efflorescence: lose H2O";    const char fd27[] PROGMEM = "Na2CO3.10H2O";
const char ff28[] PROGMEM = "Deliquescence: dissolves";   const char fd28[] PROGMEM = "NaOH absorbs+dissolves";
const char ff29[] PROGMEM = "Hygroscopic: absorbs H2O";   const char fd29[] PROGMEM = "H2SO4 drying agent";
const char ff30[] PROGMEM = "NO2: mixed anhydride";       const char fd30[] PROGMEM = "HNO2 + HNO3";
const char ff31[] PROGMEM = "Neuron = NS unit";           const char fd31[] PROGMEM = "Structural+functional";
const char ff32[] PROGMEM = "Nissl granules: protein";    const char fd32[] PROGMEM = "Ribosomes in cyton";
const char ff33[] PROGMEM = "Myelin: speeds impulse";     const char fd33[] PROGMEM = "Insulates axon";
const char ff34[] PROGMEM = "Ranvier nodes: saltatory";   const char fd34[] PROGMEM = "Jumping conduction";
const char ff35[] PROGMEM = "Synapse: one-way";           const char fd35[] PROGMEM = "Vesicles one side";
const char ff36[] PROGMEM = "Meninges: DAP layers";       const char fd36[] PROGMEM = "Dura Arachnoid Pia";
const char ff37[] PROGMEM = "CSF: cushions brain";        const char fd37[] PROGMEM = "Shock absorber";
const char ff38[] PROGMEM = "Gyri: cortex ridges";        const char fd38[] PROGMEM = "Increase surface area";
const char ff39[] PROGMEM = "Corpus callosum";            const char fd39[] PROGMEM = "Connects hemispheres";
const char ff40[] PROGMEM = "Cerebrum: intelligence";     const char fd40[] PROGMEM = "Memory willpower";
const char ff41[] PROGMEM = "Cerebellum: balance";        const char fd41[] PROGMEM = "Coordination center";
const char ff42[] PROGMEM = "Medulla: vital center";      const char fd42[] PROGMEM = "Heart+breathing";
const char ff43[] PROGMEM = "Hypothalamus: homeo";        const char fd43[] PROGMEM = "Hunger thirst temp";
const char ff44[] PROGMEM = "Sympathetic: fight-flight";  const char fd44[] PROGMEM = "Noradrenaline";
const char ff45[] PROGMEM = "Parasympathetic: rest";      const char fd45[] PROGMEM = "Acetylcholine";
const char ff46[] PROGMEM = "Reflex: involuntary";        const char fd46[] PROGMEM = "Protective rapid";
const char ff47[] PROGMEM = "Afferent: toward CNS";       const char fd47[] PROGMEM = "Sensory neuron";
const char ff48[] PROGMEM = "Efferent: away CNS";         const char fd48[] PROGMEM = "Motor neuron";
const char ff49[] PROGMEM = "12 cranial + 31 spinal";     const char fd49[] PROGMEM = "Human PNS nerves";

const Fact FACT_TABLE[] PROGMEM = {
  {ff0,fd0},{ff1,fd1},{ff2,fd2},{ff3,fd3},{ff4,fd4},{ff5,fd5},{ff6,fd6},{ff7,fd7},{ff8,fd8},{ff9,fd9},
  {ff10,fd10},{ff11,fd11},{ff12,fd12},{ff13,fd13},{ff14,fd14},{ff15,fd15},{ff16,fd16},{ff17,fd17},{ff18,fd18},{ff19,fd19},
  {ff20,fd20},{ff21,fd21},{ff22,fd22},{ff23,fd23},{ff24,fd24},{ff25,fd25},{ff26,fd26},{ff27,fd27},{ff28,fd28},{ff29,fd29},
  {ff30,fd30},{ff31,fd31},{ff32,fd32},{ff33,fd33},{ff34,fd34},{ff35,fd35},{ff36,fd36},{ff37,fd37},{ff38,fd38},{ff39,fd39},
  {ff40,fd40},{ff41,fd41},{ff42,fd42},{ff43,fd43},{ff44,fd44},{ff45,fd45},{ff46,fd46},{ff47,fd47},{ff48,fd48},{ff49,fd49}
};
#define FACT_COUNT 50

// ============================================================
//  SHOP SYSTEM
// ============================================================
struct ShopItem { const char* name; uint16_t cost; uint8_t effect; uint8_t type; };
const ShopItem SHOP_ITEMS[] = {
  {"Apple",        5,  10, 0},
  {"Pizza",       15,  25, 0},
  {"Burger",      25,  40, 0},
  {"Energy Drink",10,  15, 1},
  {"Coffee",      20,  30, 1},
  {"Vitamin",     30,  50, 1},
  {"Gift Box",    50,  20, 2},
  {"Star",       100,  50, 2}
};
#define SHOP_ITEM_COUNT 8

// ============================================================
//  TIMER STATE
// ============================================================
struct TimerState {
  uint32_t      total;
  uint32_t      remaining;
  bool          running;
  unsigned long lastTick;
  int           presetIdx;
};
TimerState gTimer = {0, 0, false, 0, 2};
const uint16_t TIMER_PRESETS[] = {300, 900, 1500, 2700, 3600};
const char* TIMER_NAMES[] = {"5m Break","15m Quick","25m Study","45m Lecture","60m Exam"};

// ============================================================
//  DISPLAY HELPERS
// ============================================================
void oledClear() {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
}

void oledHeader(const char* title) {
  display.setCursor(0, 0);
  display.println(title);
  display.drawFastHLine(0, 9, 128, SSD1306_WHITE);
}

void renderMenu(const char* title, const char* items[], int count, int sel) {
  oledClear();
  oledHeader(title);
  int start = (sel > 3) ? sel - 3 : 0;
  for (int i = 0; i < 4 && (start + i) < count; i++) {
    int idx = start + i;
    int y   = 12 + i * 13;
    if (idx == sel) {
      display.fillRect(0, y - 1, 128, 12, SSD1306_WHITE);
      display.setTextColor(SSD1306_BLACK);
    } else {
      display.setTextColor(SSD1306_WHITE);
    }
    display.setCursor(4, y);
    display.print(idx == sel ? F("> ") : F("  "));
    display.print(items[idx]);
    display.setTextColor(SSD1306_WHITE);
  }
  // Scroll indicator
  if (count > 4) {
    int barH = max(4, 52 * 4 / count);
    int barY = 10 + (sel * (52 - barH) / (count - 1));
    display.fillRect(125, barY, 3, barH, SSD1306_WHITE);
  }
  display.display();
}

void showMsg(const char* line1, const char* line2 = nullptr, uint16_t ms = 2000) {
  oledClear();
  display.setTextSize(1);
  display.setCursor(0, 18);
  display.println(line1);
  if (line2) display.println(line2);
  display.display();
  delay(ms);
}

// ============================================================
//  WEATHER ICONS (16x16 bitmaps)
// ============================================================
const unsigned char ICO_SUN[] PROGMEM = {
  0x02,0x40,0x02,0x40,0x00,0x00,0x0F,0x78,
  0x1F,0xFC,0x3F,0xFE,0x7F,0xFF,0xFF,0xFF,
  0xFF,0xFF,0x7F,0xFF,0x3F,0xFE,0x1F,0xFC,
  0x0F,0x78,0x00,0x00,0x02,0x40,0x02,0x40
};
const unsigned char ICO_CLOUD[] PROGMEM = {
  0x00,0x00,0x00,0x00,0x0F,0x00,0x1F,0x80,
  0x3F,0xC0,0x7F,0xE0,0xFF,0xF0,0xFF,0xF8,
  0xFF,0xF8,0xFF,0xF8,0xFF,0xF8,0x7F,0xF0,
  0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00
};
const unsigned char ICO_RAIN[] PROGMEM = {
  0x00,0x00,0x0F,0x00,0x1F,0x80,0x3F,0xC0,
  0xFF,0xF0,0xFF,0xF8,0xFF,0xF8,0x7F,0xF0,
  0x22,0x40,0x14,0x80,0x22,0x40,0x14,0x80,
  0x22,0x40,0x00,0x00,0x00,0x00,0x00,0x00
};

void drawWeatherIcon() {
  const unsigned char* bmp = ICO_CLOUD;
  if (g_icon == "sun")  bmp = ICO_SUN;
  if (g_icon == "rain") bmp = ICO_RAIN;
  display.drawBitmap(110, 0, bmp, 16, 16, SSD1306_WHITE);
}

// ============================================================
//  WEATHER DISPLAY
// ============================================================
void renderWeather() {
  oledClear();
  drawWeatherIcon();
  display.setTextSize(2);
  display.setCursor(0, 0);
  display.print(g_time);
  display.setTextSize(1);
  display.setCursor(0, 18);
  display.print(g_date);
  display.drawFastHLine(0, 27, 108, SSD1306_WHITE);
  display.setTextSize(2);
  display.setCursor(0, 30);
  display.print(g_temp, 1);
  display.print((char)247); display.print('C');
  display.setTextSize(1);
  display.setCursor(76, 30); display.print(F("Feel"));
  display.setCursor(76, 40); display.print(g_feels, 0); display.print((char)247); display.print('C');
  display.setCursor(0, 50);
  display.print(F("H:")); display.print(g_humidity); display.print(F("% W:"));
  display.print(g_windKph); display.print(F("kph"));
  if (g_wifiOk && WiFi.status() == WL_CONNECTED) {
    int r = WiFi.RSSI();
    display.setCursor(100, 56);
    display.print(r > -55 ? F("[3]") : r > -70 ? F("[2]") : F("[1]"));
  }
  display.display();
}

// ============================================================
//  WIFI & WEATHER FETCH
// ============================================================
String jsonVal(const String& json, const String& key) {
  String tok = "\""; tok += key; tok += "\":";
  int ki = json.indexOf(tok);
  if (ki < 0) return "";
  int si = ki + tok.length();
  while (si < (int)json.length() && json[si] == ' ') si++;
  if (si >= (int)json.length()) return "";
  char fc = json[si];
  if (fc == '"') { si++; int ei = json.indexOf('"', si); return ei < 0 ? "" : json.substring(si, ei); }
  if (fc == 't') return "1"; if (fc == 'f') return "0"; if (fc == 'n') return "";
  int ei = si;
  while (ei < (int)json.length()) { char c = json[ei]; if (isDigit(c)||c=='.'||c=='-') ei++; else break; }
  return json.substring(si, ei);
}

bool syncNTP() {
  configTime(GMT_OFFSET_SEC, DST_OFFSET_SEC, "pool.ntp.org", "time.google.com");
  struct tm ti;
  for (int i = 0; i < 20; i++) {
    if (getLocalTime(&ti, 500)) {
      char tb[8], db[14];
      strftime(tb, sizeof(tb), "%H:%M", &ti);
      strftime(db, sizeof(db), "%d %b %Y", &ti);
      g_time = tb; g_date = db;
      return true;
    }
  }
  return false;
}

bool fetchWeather() {
  if (!g_wifiOk) return false;
  String url = F("https://api.weatherapi.com/v1/current.json?key=");
  url += API_KEY; url += F("&q="); url += LOCATION; url += F("&aqi=no");
  WiFiClientSecure client; client.setInsecure();
  HTTPClient http; http.begin(client, url); http.setTimeout(15000);
  int code = http.GET();
  if (code != 200) { http.end(); return false; }
  String payload = http.getString(); http.end();
  g_temp     = jsonVal(payload, "temp_c").toFloat();
  g_humidity = jsonVal(payload, "humidity").toInt();
  g_feels    = jsonVal(payload, "feelslike_c").toFloat();
  g_windKph  = (int)jsonVal(payload, "wind_kph").toFloat();
  g_desc     = jsonVal(payload, "text");
  String lc  = g_desc; lc.toLowerCase();
  g_icon = (lc.indexOf("sun")>=0||lc.indexOf("clear")>=0) ? "sun"
          : (lc.indexOf("rain")>=0||lc.indexOf("drizzle")>=0) ? "rain" : "cloud";
  g_hasData  = true;
  g_lastWeatherFetch = millis();
  return true;
}

bool connectWiFi(const char* ssid, const char* pass) {
  WiFi.disconnect(true); delay(100);
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, pass);
  for (int i = 0; i < 20; i++) {
    if (WiFi.status() == WL_CONNECTED) return true;
    oledClear();
    display.setCursor(0, 20);
    display.print(F("WiFi: ")); display.print(i * 5); display.println(F("%"));
    display.display();
    delay(500);
  }
  return false;
}

// ============================================================
//  MCQ MODE
// ============================================================
void runMCQMode() {
  // Shuffle index selection
  int qIdx = random(0, MCQ_COUNT);
  MCQ item; memcpy_P(&item, &MCQ_TABLE[qIdx], sizeof(MCQ));
  char q[48], a[28], b[28], c[28], d[28], e[48];
  strncpy_P(q, item.question, 47); q[47]=0;
  strncpy_P(a, item.optA, 27); a[27]=0;
  strncpy_P(b, item.optB, 27); b[27]=0;
  strncpy_P(c, item.optC, 27); c[27]=0;
  strncpy_P(d, item.optD, 27); d[27]=0;
  strncpy_P(e, item.explain, 47); e[47]=0;

  // Phase 1: Show question (10s)
  unsigned long t0 = millis();
  while (millis()-t0 < 10000) {
    oledClear(); oledHeader("MCQ Quiz");
    display.setCursor(0, 12);
    display.println(q);
    display.setCursor(0, 56);
    int rem = 10 - (millis()-t0)/1000;
    display.print(F("Think...")); display.print(rem); display.print('s');
    display.display();
    ButtonAction ba = getButtonAction();
    if (ba == ACTION_BACK || ba == ACTION_LEFT) { g_state = STATE_STUDY_MENU; return; }
    delay(100);
  }

  // Phase 2: Show options with selection
  int sel = 0;
  bool answered = false;
  const char* opts[] = {a, b, c, d};
  unsigned long t1 = millis();

  while (!answered && millis()-t1 < 15000) {
    oledClear(); oledHeader("Select Answer");
    for (int i = 0; i < 4; i++) {
      int y = 12 + i*13;
      if (i == sel) {
        display.fillRect(0, y-1, 128, 12, SSD1306_WHITE);
        display.setTextColor(SSD1306_BLACK);
      } else display.setTextColor(SSD1306_WHITE);
      display.setCursor(2, y);
      display.print(char('A'+i)); display.print(')');
      display.print(opts[i]);
      display.setTextColor(SSD1306_WHITE);
    }
    int rem = 15 - (millis()-t1)/1000;
    display.setCursor(100,56); display.print(rem); display.print('s');
    display.display();

    ButtonAction ba = getButtonAction();
    if (ba == ACTION_UP)    sel = (sel + 3) % 4;
    if (ba == ACTION_DOWN)  sel = (sel + 1) % 4;
    if (ba == ACTION_RIGHT) answered = true;
    if (ba == ACTION_BACK || ba == ACTION_LEFT) { g_state = STATE_STUDY_MENU; return; }
    delay(100);
  }

  // Phase 3: Show result
  bool correct = (sel == item.answer);
  player.totalMCQs++;
  if (correct) { player.correctMCQs++; player.coins += 3; player.xp += 10; }
  else          { player.coins += 1; player.xp += 2; }

  unsigned long t2 = millis();
  while (millis()-t2 < 5000) {
    oledClear();
    display.setCursor(0,0);
    display.println(correct ? F("CORRECT!") : F("WRONG!"));
    display.drawFastHLine(0,9,128,SSD1306_WHITE);
    display.setCursor(0,12);
    display.print(F("You: ")); display.println(char('A'+sel));
    display.print(F("Ans: ")); display.println(char('A'+item.answer));
    display.setCursor(0,35); display.println(e);
    display.setCursor(0,56);
    display.print(correct ? F("+3 coins +10xp") : F("+1 coin +2xp"));
    display.display();
    if (getButtonAction() != ACTION_NONE) break;
    delay(100);
  }

  // Level up check
  if (player.xp >= player.level * 50) {
    player.xp -= player.level * 50;
    player.level++;
    player.coins += 20;
    showMsg("LEVEL UP!", ("Level " + String(player.level)).c_str(), 2000);
  }
  savePrefs();
  g_state = STATE_STUDY_MENU;
}

// ============================================================
//  FACTS MODE
// ============================================================
void runFactsMode() {
  int fIdx = random(0, FACT_COUNT);
  Fact item; memcpy_P(&item, &FACT_TABLE[fIdx], sizeof(Fact));
  char f[40], det[40];
  strncpy_P(f, item.fact, 39); f[39]=0;
  strncpy_P(det, item.detail, 39); det[39]=0;

  unsigned long t0 = millis();
  while (millis()-t0 < 8000) {
    oledClear(); oledHeader("Fact Mode");
    display.setTextSize(1);
    display.setCursor(0, 14);
    display.println(f);
    display.setCursor(0, 38); display.println(det);
    display.setCursor(0,56); display.print(F("Fact #")); display.print(fIdx+1);
    display.display();
    ButtonAction ba = getButtonAction();
    if (ba == ACTION_RIGHT) { fIdx = random(0,FACT_COUNT); memcpy_P(&item,&FACT_TABLE[fIdx],sizeof(Fact)); strncpy_P(f,item.fact,39); strncpy_P(det,item.detail,39); t0=millis(); }
    if (ba == ACTION_BACK || ba == ACTION_LEFT) break;
    delay(100);
  }
  player.coins += 1; player.totalFacts++; player.xp += 3;
  savePrefs();
  g_state = STATE_STUDY_MENU;
}

// ============================================================
//  STUDY STATS
// ============================================================
void renderStudyStats() {
  while (true) {
    oledClear(); oledHeader("Study Stats");
    display.setCursor(0, 12);
    display.print(F("MCQs: ")); display.print(player.totalMCQs);
    if (player.totalMCQs > 0) {
      display.print(F(" (")); display.print(player.correctMCQs*100/player.totalMCQs); display.print(F("%)"));
    }
    display.setCursor(0,24); display.print(F("Facts: ")); display.print(player.totalFacts);
    display.setCursor(0,36); display.print(F("Coins: ")); display.print(player.coins);
    display.setCursor(0,48); display.print(F("Level: ")); display.print(player.level);
    display.print(F("  XP: ")); display.print(player.xp);
    display.display();
    ButtonAction ba = getButtonAction();
    if (ba == ACTION_LEFT || ba == ACTION_BACK) { g_state = STATE_STUDY_MENU; return; }
    delay(100);
  }
}

// ============================================================
//  SHOP
// ============================================================
void renderShop() {
  while (true) {
    oledClear();
    display.setCursor(0,0); display.print(F("Shop  Coins:")); display.print(player.coins);
    display.drawFastHLine(0,9,128,SSD1306_WHITE);
    int start = g_menuSel > 3 ? g_menuSel-3 : 0;
    for (int i = 0; i < 4 && start+i < SHOP_ITEM_COUNT; i++) {
      int idx = start+i, y = 12+i*13;
      if (idx==g_menuSel) { display.fillRect(0,y-1,128,12,SSD1306_WHITE); display.setTextColor(SSD1306_BLACK); }
      else display.setTextColor(SSD1306_WHITE);
      display.setCursor(2,y); display.print(SHOP_ITEMS[idx].name);
      display.setCursor(85,y); display.print(SHOP_ITEMS[idx].cost); display.print('c');
      display.setTextColor(SSD1306_WHITE);
    }
    display.display();
    ButtonAction ba = getButtonAction();
    if (ba==ACTION_UP)   g_menuSel=(g_menuSel-1+SHOP_ITEM_COUNT)%SHOP_ITEM_COUNT;
    if (ba==ACTION_DOWN) g_menuSel=(g_menuSel+1)%SHOP_ITEM_COUNT;
    if (ba==ACTION_RIGHT||ba==ACTION_SELECT) {
      const ShopItem& it = SHOP_ITEMS[g_menuSel];
      if (player.coins < it.cost) { showMsg("Not enough coins!","",1500); continue; }
      player.coins -= it.cost;
      switch(it.type) {
        case 0: player.hunger    = min(100, player.hunger+it.effect); break;
        case 1: player.energy    = min(100, player.energy+it.effect); break;
        case 2: player.happiness = min(100, player.happiness+it.effect);
                if (g_menuSel==7) player.level++;
                break;
      }
      showMsg("Bought!", it.name, 1500);
      savePrefs();
    }
    if (ba==ACTION_LEFT||ba==ACTION_BACK) { g_state=STATE_MAIN_MENU; g_menuSel=3; return; }
    delay(50);
  }
}

// ============================================================
//  TIMER
// ============================================================
void renderTimer() {
  while (true) {
    if (!gTimer.running) {
      oledClear(); oledHeader("Timer");
      display.setCursor(0,14);
      display.print(F("Preset: ")); display.println(TIMER_NAMES[gTimer.presetIdx]);
      display.setCursor(0,28);
      int m = TIMER_PRESETS[gTimer.presetIdx]/60;
      display.print(m); display.println(F(" minutes"));
      display.setCursor(0,42); display.println(F("UP/DN: Change"));
      display.setCursor(0,52); display.println(F("RIGHT: Start  LEFT: Back"));
      display.display();
      ButtonAction ba = getButtonAction();
      if (ba==ACTION_UP)   gTimer.presetIdx=(gTimer.presetIdx+1)%5;
      if (ba==ACTION_DOWN) gTimer.presetIdx=(gTimer.presetIdx-1+5)%5;
      if (ba==ACTION_RIGHT) {
        gTimer.total     = TIMER_PRESETS[gTimer.presetIdx];
        gTimer.remaining = gTimer.total;
        gTimer.running   = true;
        gTimer.lastTick  = millis();
      }
      if (ba==ACTION_LEFT||ba==ACTION_BACK) { g_state=STATE_APPS_MENU; g_menuSel=0; return; }
    } else {
      if (millis()-gTimer.lastTick >= 1000) {
        gTimer.lastTick = millis();
        if (gTimer.remaining > 0) gTimer.remaining--;
        else { gTimer.running=false; player.coins++; savePrefs(); showMsg("Timer Done!","+1 coin",2000); continue; }
      }
      oledClear(); oledHeader("Timer Running");
      display.setTextSize(2); display.setCursor(22,16);
      int mn=gTimer.remaining/60, sc=gTimer.remaining%60;
      if(mn<10) display.print('0'); display.print(mn);
      display.print(':');
      if(sc<10) display.print('0'); display.print(sc);
      display.setTextSize(1);
      int bw = (gTimer.remaining*120)/gTimer.total;
      display.drawRect(4,46,120,10,SSD1306_WHITE);
      display.fillRect(4,46,bw,10,SSD1306_WHITE);
      display.setCursor(0,58); display.print(F("LEFT: Stop"));
      display.display();
      ButtonAction ba = getButtonAction();
      if (ba==ACTION_LEFT||ba==ACTION_BACK) { gTimer.running=false; g_state=STATE_APPS_MENU; g_menuSel=0; return; }
    }
    delay(50);
  }
}

// ============================================================
//  GOALS
// ============================================================
void renderGoals() {
  int sel = 0;
  while (true) {
    oledClear(); oledHeader("Goal Tracker");
    for (int i=0;i<4;i++) {
      int y=12+i*13;
      if(i==sel) { display.fillRect(0,y-1,80,12,SSD1306_WHITE); display.setTextColor(SSD1306_BLACK); }
      else display.setTextColor(SSD1306_WHITE);
      display.setCursor(2,y); display.print(goals[i].name);
      display.setTextColor(SSD1306_WHITE);
      display.drawRect(60,y,60,8,SSD1306_WHITE);
      int bw=goals[i].progress*58/100;
      display.fillRect(60,y,bw,8,SSD1306_WHITE);
      display.setCursor(122,y); display.setTextColor(goals[i].completed?SSD1306_WHITE:SSD1306_WHITE);
      display.print(goals[i].completed ? F("*") : F(" "));
    }
    display.display();
    ButtonAction ba = getButtonAction();
    if (ba==ACTION_UP)   sel=(sel-1+4)%4;
    if (ba==ACTION_DOWN) sel=(sel+1)%4;
    if (ba==ACTION_RIGHT) {
      goals[sel].progress = min(100, goals[sel].progress+25);
      if (goals[sel].progress>=100 && !goals[sel].completed) {
        goals[sel].completed=true; player.coins+=5; savePrefs();
        showMsg("Goal Complete!","+5 coins",1500);
      }
    }
    if (ba==ACTION_LEFT) {
      goals[sel].progress = max(0, goals[sel].progress-25);
      goals[sel].completed = false;
    }
    if (ba==ACTION_BACK) { g_state=STATE_APPS_MENU; g_menuSel=1; return; }
    delay(50);
  }
}

// ============================================================
//  COIN FLIP
// ============================================================
void renderCoinFlip() {
  // Animation
  for (int i=0;i<6;i++) {
    oledClear();
    display.setTextSize(2); display.setCursor(28,22);
    display.println(i%2==0 ? F("HEADS") : F("TAILS"));
    display.display(); delay(200);
  }
  bool heads = esp_random()%2==0;
  oledClear(); display.setTextSize(2); display.setCursor(28,18);
  display.println(heads ? F("HEADS!") : F("TAILS!"));
  display.setTextSize(1); display.setCursor(30,42);
  display.println(heads ? F("You win") : F("Try again"));
  display.display(); delay(3000);
  g_state=STATE_APPS_MENU; g_menuSel=2;
}

// ============================================================
//  FORTUNE
// ============================================================
void renderFortune() {
  const char* fortunes[] = {
    "Today is your day!", "Surprise incoming!", "Learn something new",
    "Someone thinks of you","Stay hydrated!","Trust your gut",
    "Patience pays off","A stranger smiles","Good things coming",
    "You are enough","Take the next step","Rest, then conquer"
  };
  int idx = esp_random()%12;
  oledClear(); oledHeader("Fortune Cookie");
  display.setCursor(0,18); display.println(fortunes[idx]);
  display.setCursor(0,48); display.println(F("RIGHT for another"));
  display.display();
  unsigned long t=millis();
  while(millis()-t<8000) {
    ButtonAction ba=getButtonAction();
    if(ba==ACTION_RIGHT) { idx=esp_random()%12; t=millis();
      oledClear(); oledHeader("Fortune Cookie");
      display.setCursor(0,18); display.println(fortunes[idx]);
      display.setCursor(0,48); display.println(F("RIGHT for another"));
      display.display(); }
    if(ba==ACTION_LEFT||ba==ACTION_BACK) break;
    delay(100);
  }
  g_state=STATE_APPS_MENU; g_menuSel=3;
}

// ============================================================
//  MOOD CHECK
// ============================================================
void renderMood() {
  const char* moods[] = {
    "Life is great!","Feeling playful!","Have some fun!",
    "You are awesome!","Coffee time!","Believe in yourself!",
    "It's YOUR day!","Keep pushing!","Almost there!","You've got this!"
  };
  int idx=esp_random()%10;
  oledClear(); oledHeader("Mood Check");
  display.setCursor(0,22); display.println(moods[idx]);
  display.setCursor(0,50); display.print(F("Happy: ")); display.print(player.happiness); display.print('%');
  display.display(); delay(5000);
  g_state=STATE_APPS_MENU; g_menuSel=4;
}

// ============================================================
//  HEALTH CHECK
// ============================================================
void renderHealth() {
  while(true) {
    oledClear(); oledHeader("Health Check");
    display.setCursor(0,12);
    display.print(F("Hunger:  ")); display.print(player.hunger);  display.println('%');
    display.print(F("Energy:  ")); display.print(player.energy);  display.println('%');
    display.print(F("Happy:   ")); display.print(player.happiness); display.println('%');
    display.print(F("Coins:   ")); display.println(player.coins);
    unsigned long sec=millis()/1000;
    display.print(F("Uptime: ")); display.print(sec/3600); display.print(F("h "));
    display.print((sec%3600)/60); display.println('m');
    display.display();
    ButtonAction ba=getButtonAction();
    if(ba==ACTION_LEFT||ba==ACTION_BACK) { g_state=STATE_APPS_MENU; g_menuSel=5; return; }
    delay(200);
  }
}

// ============================================================
//  WIFI SCANNER
// ============================================================
void renderScanner() {
  oledClear(); oledHeader("WiFi Scanner");
  display.setCursor(0,14); display.println(F("Scanning..."));
  display.display();
  int n = WiFi.scanNetworks();
  oledClear(); oledHeader("WiFi Scanner");
  display.setCursor(0,12); display.print(n); display.println(F(" networks found"));
  for(int i=0;i<min(n,3);i++) {
    display.setCursor(0,24+i*13);
    String s=WiFi.SSID(i);
    if(s.length()>14) s=s.substring(0,11)+"...";
    display.print(s);
    display.setCursor(100,24+i*13); display.print(WiFi.RSSI(i));
  }
  display.display(); delay(6000);
  WiFi.scanDelete();
  g_state=STATE_APPS_MENU; g_menuSel=6;
}

// ============================================================
//  KEYBOARD
// ============================================================
void renderKeyboard() {
  const char* rows[3] = {"QWERTYUIOP","ASDFGHJKL","ZXCVBNM"};
  int row=0, col=0;
  while(true) {
    oledClear(); oledHeader("Keyboard");
    for(int r=0;r<3;r++) {
      int rlen=strlen(rows[r]);
      int xoff=(r==1)?5:(r==2)?10:0;
      for(int c=0;c<rlen;c++) {
        int x=xoff+c*12, y=12+r*13;
        bool sel=(r==row&&c==col);
        if(sel) { display.fillRect(x-1,y-1,11,11,SSD1306_WHITE); display.setTextColor(SSD1306_BLACK); }
        else display.setTextColor(SSD1306_WHITE);
        display.setCursor(x,y); display.print(rows[r][c]);
        display.setTextColor(SSD1306_WHITE);
      }
    }
    // Text area
    display.drawRect(0,51,128,13,SSD1306_WHITE);
    display.setCursor(2,54);
    int blen=strlen(keyboardBuffer);
    if(blen>16) display.print(keyboardBuffer+blen-16);
    else display.print(keyboardBuffer);
    display.print('_');
    display.display();

    ButtonAction ba=getButtonAction();
    if(ba==ACTION_UP)   row=(row-1+3)%3;
    if(ba==ACTION_DOWN) row=(row+1)%3;
    if(ba==ACTION_LEFT) col=max(0,col-1);
    if(ba==ACTION_RIGHT){
      int rlen=strlen(rows[row]);
      if(col<rlen-1) col++;
      else {
        // Type character
        int len=strlen(keyboardBuffer);
        if(len<32){ keyboardBuffer[len]=rows[row][col]; keyboardBuffer[len+1]='\0'; }
      }
    }
    if(ba==ACTION_SELECT) { // Long right: type
      int len=strlen(keyboardBuffer);
      if(len<32){ keyboardBuffer[len]=rows[row][col]; keyboardBuffer[len+1]='\0'; }
    }
    if(ba==ACTION_SPECIAL) { // Long up: delete
      int len=strlen(keyboardBuffer);
      if(len>0) keyboardBuffer[len-1]='\0';
    }
    if(ba==ACTION_BACK) { g_state=STATE_APPS_MENU; g_menuSel=7; keyboardBuffer[0]='\0'; return; }
    delay(50);
  }
}

// ============================================================
//  SLEEP MODE
// ============================================================
const char* QUOTES[] = {
  "It's all yours","Be kind today","Night: escape mode",
  "Built by Soumyajit","Boards incoming!","Fix weak spots",
  "Not the villain!","Born for purpose","More darkness first",
  "Don't waste today","Coffee? :)","I believe in you",
  "No hormone chaos","Don't ruin the day","Why do you exist?",
  "Die standing!","It'll be okay","Level increasing",
  "Someone's watching","Are you ok?","Study counts friend",
  "Code is poetry","No errors tonight","Future starts now",
  "Night ends at dawn","One line changes all","Make days matter",
  "Why did you start?","Main character mode","Keep that energy",
  "Real > perfect","You are the plan"
};
#define QUOTE_COUNT 32

void runSleepMode() {
  int qi = 0;
  while(true) {
    oledClear(); oledHeader("Sleep Mode");
    display.setCursor(0,16); display.println(QUOTES[qi%QUOTE_COUNT]);
    display.setCursor(80,56); display.print(qi%QUOTE_COUNT+1); display.print('/'); display.print(QUOTE_COUNT);
    display.display();
    unsigned long t=millis();
    while(millis()-t<5000) {
      ButtonAction ba=getButtonAction();
      if(ba!=ACTION_NONE) { g_state=STATE_WEATHER; return; }
      delay(100);
    }
    qi++;
    if(qi>=QUOTE_COUNT) { g_state=STATE_WEATHER; return; }
  }
}

// ============================================================
//  HIGH SCORES
// ============================================================
void renderHighScores() {
  while(true) {
    oledClear(); oledHeader("High Scores");
    display.setCursor(0,12);
    display.print(F("Space War: ")); display.println(player.hsSpaceWar);
    display.print(F("Snake:     ")); display.println(player.hsSnake);
    display.print(F("Pong:      ")); display.println(player.hsPong);
    display.print(F("Car:       ")); display.println(player.hsCar);
    display.display();
    ButtonAction ba=getButtonAction();
    if(ba==ACTION_DOWN||ba==ACTION_RIGHT) {
      oledClear(); oledHeader("High Scores 2");
      display.setCursor(0,12);
      display.print(F("Maze:    ")); display.println(player.hsMaze);
      display.print(F("Flappy:  ")); display.println(player.hsFlappy);
      display.print(F("2048:    ")); display.println(player.hs2048);
      display.print(F("Tetris:  ")); display.println(player.hsTetris);
      display.display();
      delay(3000);
    }
    if(ba==ACTION_LEFT||ba==ACTION_BACK) { g_state=STATE_GAMES_MENU; g_menuSel=8; return; }
    delay(100);
  }
}

// ============================================================
//  SETTINGS
// ============================================================
void renderSettingsBrightness() {
  while(true) {
    oledClear(); oledHeader("Brightness");
    display.setCursor(0,18);
    const char* lvls[]={"25%","50%","75%","100%"};
    for(int i=0;i<4;i++) {
      if(i==player.brightness) { display.fillRect(0,16+i*12,80,11,SSD1306_WHITE); display.setTextColor(SSD1306_BLACK); }
      else display.setTextColor(SSD1306_WHITE);
      display.setCursor(4,18+i*12); display.print(lvls[i]);
      display.setTextColor(SSD1306_WHITE);
    }
    display.display();
    ButtonAction ba=getButtonAction();
    if(ba==ACTION_UP)   player.brightness=(player.brightness-1+4)%4;
    if(ba==ACTION_DOWN) player.brightness=(player.brightness+1)%4;
    if(ba==ACTION_RIGHT||ba==ACTION_SELECT) { applyBrightness(); savePrefs(); showMsg("Saved!","",1000); }
    if(ba==ACTION_LEFT||ba==ACTION_BACK) { g_state=STATE_SETTINGS_MENU; g_menuSel=0; return; }
    delay(50);
  }
}

void renderSettingsTimeout() {
  const uint32_t TIMEOUTS[]={5000,10000,30000,60000,0};
  const char* TNAMES[]={"5s","10s","30s","60s","Never"};
  int sel=0;
  for(int i=0;i<5;i++) if(TIMEOUTS[i]==player.menuTimeoutMs) sel=i;
  while(true) {
    oledClear(); oledHeader("Menu Timeout");
    for(int i=0;i<5;i++) {
      int y=12+i*10;
      if(i==sel) { display.fillRect(0,y-1,90,10,SSD1306_WHITE); display.setTextColor(SSD1306_BLACK); }
      else display.setTextColor(SSD1306_WHITE);
      display.setCursor(4,y); display.print(TNAMES[i]);
      display.setTextColor(SSD1306_WHITE);
    }
    display.display();
    ButtonAction ba=getButtonAction();
    if(ba==ACTION_UP)   sel=(sel-1+5)%5;
    if(ba==ACTION_DOWN) sel=(sel+1)%5;
    if(ba==ACTION_RIGHT||ba==ACTION_SELECT) {
      player.menuTimeoutMs=TIMEOUTS[sel]; savePrefs(); showMsg("Timeout saved!","",1000);
    }
    if(ba==ACTION_LEFT||ba==ACTION_BACK) { g_state=STATE_SETTINGS_MENU; g_menuSel=1; return; }
    delay(50);
  }
}

void renderSettingsWifi() {
  // Show current + prompt reconnect
  while(true) {
    oledClear(); oledHeader("WiFi Settings");
    display.setCursor(0,12);
    display.print(F("SSID: ")); display.println(player.wifiSSID);
    display.setCursor(0,26);
    display.print(F("Status: ")); display.println(g_wifiOk?F("Connected"):F("Disconnected"));
    if(g_wifiOk) { display.setCursor(0,38); display.print(F("IP: ")); display.print(WiFi.localIP()); }
    display.setCursor(0,52); display.println(F("RIGHT: Reconnect"));
    display.display();
    ButtonAction ba=getButtonAction();
    if(ba==ACTION_RIGHT) {
      showMsg("Reconnecting...","",500);
      g_wifiOk=connectWiFi(player.wifiSSID,player.wifiPass);
      if(g_wifiOk) { syncNTP(); fetchWeather(); showMsg("Connected!","",1500); }
      else showMsg("Failed!","Check SSID/Pass",2000);
    }
    if(ba==ACTION_LEFT||ba==ACTION_BACK) { g_state=STATE_SETTINGS_MENU; g_menuSel=2; return; }
    delay(100);
  }
}

void renderSettingsReset() {
  oledClear(); oledHeader("Reset All?");
  display.setCursor(0,20); display.println(F("RIGHT: Confirm"));
  display.setCursor(0,34); display.println(F("LEFT:  Cancel"));
  display.display();
  unsigned long t=millis();
  while(millis()-t<10000) {
    ButtonAction ba=getButtonAction();
    if(ba==ACTION_RIGHT) {
      prefs.begin("homie",false); prefs.clear(); prefs.end();
      showMsg("Reset Done!","Restart device",3000);
      ESP.restart();
    }
    if(ba==ACTION_LEFT||ba==ACTION_BACK) { g_state=STATE_SETTINGS_MENU; g_menuSel=4; return; }
    delay(100);
  }
  g_state=STATE_SETTINGS_MENU; g_menuSel=4;
}

void renderAbout() {
  while(true) {
    oledClear(); oledHeader("About HOMIE");
    display.setCursor(0,12); display.println(F("Version: 7.0 Final"));
    display.setCursor(0,24); display.println(F("By: Soumyajit"));
    display.setCursor(0,36); display.println(F("ESP32 + SSD1306"));
    display.setCursor(0,48); display.println(F("8 Games, 100 MCQs"));
    display.display();
    ButtonAction ba=getButtonAction();
    if(ba==ACTION_LEFT||ba==ACTION_BACK) { g_state=STATE_SETTINGS_MENU; g_menuSel=5; return; }
    delay(100);
  }
}

// ============================================================
//  GAME HELPER: show game over screen
// ============================================================
void gameOver(const char* gameName, int score, uint16_t& hs) {
  if(score > (int)hs) { hs=(uint16_t)score; savePrefs(); }
  player.coins += score/10 + 1; savePrefs();
  oledClear(); display.setTextSize(2); display.setCursor(8,10);
  display.println(F("GAME OVER")); display.setTextSize(1);
  display.setCursor(20,34); display.print(F("Score: ")); display.print(score);
  display.setCursor(20,46); display.print(F("Best:  ")); display.print(hs);
  display.setCursor(0,56); display.print(F("Coins+")); display.print(score/10+1);
  display.display(); delay(3000);
}

void gamePause() {
  oledClear(); display.setTextSize(2); display.setCursor(18,24);
  display.println(F("PAUSED"));
  display.setTextSize(1); display.setCursor(0,52); display.println(F("Any button: Resume"));
  display.display();
  while(getButtonAction()==ACTION_NONE) delay(100);
}

// ============================================================
//  GAME 1: SPACE WAR
// ============================================================
void gameSpaceWar() {
  // Init
  int px=60, py=56;
  int bx=-1, by=-1;
  int exX[6], exY[6], exSpd[6];
  int score=0, lives=3;
  unsigned long lastUpdate=millis();
  bool paused=false;

  for(int i=0;i<6;i++){
    exX[i]=random(4,122); exY[i]=random(-60,-5); exSpd[i]=random(1,3);
  }

  while(true) {
    if(!paused && millis()-lastUpdate>=50) {
      lastUpdate=millis();

      ButtonAction ba=getButtonAction();
      if(ba==ACTION_LEFT)  px=max(5,px-4);
      if(ba==ACTION_RIGHT) px=min(123,px+4);
      if(ba==ACTION_UP && bx<0){ bx=px; by=py-5; }
      if(ba==ACTION_DOWN)  paused=true;
      if(ba==ACTION_BACK)  { g_state=STATE_GAME_SPACE_WAR; gameOver("SpaceWar",score,player.hsSpaceWar); g_state=STATE_GAMES_MENU; g_menuSel=0; return; }

      // Move bullet
      if(bx>=0){ by-=6; if(by<10) bx=-1; }

      // Move enemies
      for(int i=0;i<6;i++){
        exY[i]+=exSpd[i];
        if(exY[i]>64){
          exX[i]=random(4,122); exY[i]=random(-60,-10); exSpd[i]=random(1,3);
          lives--;
          if(lives<=0){ gameOver("SpaceWar",score,player.hsSpaceWar); g_state=STATE_GAMES_MENU; g_menuSel=0; return; }
        }
        // Bullet collision
        if(bx>=0 && abs(bx-exX[i])<5 && abs(by-exY[i])<5){
          score+=10; bx=-1;
          exX[i]=random(4,122); exY[i]=random(-60,-10); exSpd[i]=random(1,3);
          if(score%50==0 && score>0) for(int j=0;j<6;j++) if(exSpd[j]<5) exSpd[j]++;
        }
        // Player collision
        if(abs(px-exX[i])<5 && abs(py-exY[i])<8){
          lives--;
          exX[i]=random(4,122); exY[i]=random(-60,-10);
          if(lives<=0){ gameOver("SpaceWar",score,player.hsSpaceWar); g_state=STATE_GAMES_MENU; g_menuSel=0; return; }
        }
      }

      // Draw
      oledClear();
      display.setCursor(0,0); display.print(F("S:")); display.print(score);
      display.print(F(" L:")); display.print(lives);
      display.setCursor(100,0); display.print(F("BW:"));
      display.drawFastHLine(0,9,128,SSD1306_WHITE);

      // Enemies (UFO shape)
      for(int i=0;i<6;i++){
        if(exY[i]>9 && exY[i]<64){
          display.drawPixel(exX[i],exY[i],SSD1306_WHITE);
          display.drawFastHLine(exX[i]-3,exY[i]+1,7,SSD1306_WHITE);
          display.drawPixel(exX[i]-2,exY[i]+2,SSD1306_WHITE);
          display.drawPixel(exX[i]+2,exY[i]+2,SSD1306_WHITE);
        }
      }

      // Bullet
      if(bx>=0) display.fillRect(bx,by,1,4,SSD1306_WHITE);

      // Player ship
      display.fillTriangle(px,py-5, px-4,py+3, px+4,py+3, SSD1306_WHITE);
      display.drawFastHLine(px-3,py+3,7,SSD1306_WHITE);
      display.display();
    }

    if(paused) { gamePause(); paused=false; lastUpdate=millis(); }
    delay(10);
  }
}

// ============================================================
//  GAME 2: CAR GAME
// ============================================================
void gameCarGame() {
  // Road: 3 lanes, obstacles from top
  int laneX[3]={20,64,108};
  int playerLane=1;
  int obsLane[4],obsY[4];
  int speed=2, score=0, lives=3;
  unsigned long lastUpdate=millis(), lastScore=millis();
  bool paused=false;

  for(int i=0;i<4;i++){ obsLane[i]=random(0,3); obsY[i]=random(-80,-10); }
  // Stagger
  for(int i=0;i<4;i++) obsY[i]-=i*20;

  while(true) {
    if(!paused && millis()-lastUpdate>=60) {
      lastUpdate=millis();
      ButtonAction ba=getButtonAction();
      if(ba==ACTION_LEFT)  playerLane=max(0,playerLane-1);
      if(ba==ACTION_RIGHT) playerLane=min(2,playerLane+1);
      if(ba==ACTION_DOWN)  paused=true;
      if(ba==ACTION_BACK)  { gameOver("CarGame",score,player.hsCar); g_state=STATE_GAMES_MENU; g_menuSel=1; return; }

      // Move obstacles
      for(int i=0;i<4;i++){
        obsY[i]+=speed;
        if(obsY[i]>64){ obsLane[i]=random(0,3); obsY[i]=random(-80,-20); }
        // Collision
        if(obsLane[i]==playerLane && obsY[i]>44 && obsY[i]<60){
          lives--;
          obsY[i]=random(-80,-20); obsLane[i]=random(0,3);
          if(lives<=0){ gameOver("CarGame",score,player.hsCar); g_state=STATE_GAMES_MENU; g_menuSel=1; return; }
        }
      }

      if(millis()-lastScore>=1000){ score+=speed; lastScore=millis(); }
      if(score>0 && score%20==0) speed=min(6,speed+1);

      // Draw
      oledClear();
      // Road
      display.drawFastVLine(8,10,54,SSD1306_WHITE);
      display.drawFastVLine(42,10,54,SSD1306_WHITE);
      display.drawFastVLine(86,10,54,SSD1306_WHITE);
      display.drawFastVLine(120,10,54,SSD1306_WHITE);
      // Dashes
      for(int y=10;y<64;y+=12){ display.drawFastVLine(21,y,6,SSD1306_WHITE); display.drawFastVLine(65,y,6,SSD1306_WHITE); display.drawFastVLine(99,y,6,SSD1306_WHITE); }

      // Player car
      int px=laneX[playerLane];
      display.fillRect(px-6,48,12,14,SSD1306_WHITE);
      display.fillRect(px-4,51,4,6,SSD1306_BLACK);
      display.fillRect(px+1,51,4,6,SSD1306_BLACK);

      // Obstacles
      for(int i=0;i<4;i++){
        if(obsY[i]>9&&obsY[i]<64){
          int ox=laneX[obsLane[i]];
          display.fillRect(ox-6,obsY[i]-12,12,12,SSD1306_WHITE);
          display.fillRect(ox-4,obsY[i]-10,4,4,SSD1306_BLACK);
          display.fillRect(ox+1,obsY[i]-10,4,4,SSD1306_BLACK);
        }
      }

      display.setCursor(0,0); display.print(F("Score:")); display.print(score);
      display.print(F(" L:")); display.print(lives);
      display.display();
    }
    if(paused){ gamePause(); paused=false; lastUpdate=millis(); }
    delay(10);
  }
}

// ============================================================
//  GAME 3: SNAKE
// ============================================================
void gameSnake() {
  int sx[120], sy[120];
  sx[0]=64; sy[0]=36; int slen=4;
  for(int i=1;i<4;i++){ sx[i]=sx[i-1]-4; sy[i]=36; }
  int dir=1, nx=4, ny=0;
  int fx=random(12,120), fy=random(14,60);
  int score=0;
  unsigned long lastMove=millis();
  int speed=220;
  bool paused=false;

  while(true) {
    ButtonAction ba=getButtonAction();
    if(ba==ACTION_UP    && dir!=2){ nx=0; ny=-4; }
    if(ba==ACTION_DOWN  && dir!=0){ nx=0; ny=4;  }
    if(ba==ACTION_LEFT  && dir!=1){ nx=-4;ny=0; dir=3; }
    if(ba==ACTION_RIGHT && dir!=3){ nx=4; ny=0; dir=1; }
    if(ba==ACTION_DOWN)  paused=true;
    if(ba==ACTION_BACK)  { gameOver("Snake",score,player.hsSnake); g_state=STATE_GAMES_MENU; g_menuSel=2; return; }

    if(paused){ gamePause(); paused=false; lastMove=millis(); }

    if(millis()-lastMove>=speed) {
      lastMove=millis();
      // Determine dir
      if(nx==0&&ny==0){ nx=4;ny=0;dir=1; }

      for(int i=slen-1;i>0;i--){ sx[i]=sx[i-1]; sy[i]=sy[i-1]; }
      sx[0]+=nx; sy[0]+=ny;

      // Wall collision
      if(sx[0]<4||sx[0]>123||sy[0]<12||sy[0]>62){ gameOver("Snake",score,player.hsSnake); g_state=STATE_GAMES_MENU; g_menuSel=2; return; }

      // Self collision
      for(int i=2;i<slen;i++) if(sx[0]==sx[i]&&sy[0]==sy[i]){ gameOver("Snake",score,player.hsSnake); g_state=STATE_GAMES_MENU; g_menuSel=2; return; }

      // Food
      if(abs(sx[0]-fx)<4&&abs(sy[0]-fy)<4){
        if(slen<119) slen++;
        score+=10;
        fx=random(8,118); fy=random(14,58);
        if(speed>80) speed-=5;
      }

      oledClear();
      display.setCursor(0,0); display.print(F("Snake  Score:")); display.print(score);
      display.drawFastHLine(0,9,128,SSD1306_WHITE);
      display.drawRect(2,11,124,52,SSD1306_WHITE);

      // Snake body
      for(int i=0;i<slen;i++){
        if(i==0) display.fillRect(sx[i]-1,sy[i]-1,4,4,SSD1306_WHITE);
        else     display.fillRect(sx[i],sy[i],3,3,SSD1306_WHITE);
      }

      // Food (blinking)
      display.fillRect(fx,fy,4,4,SSD1306_WHITE);
      display.fillRect(fx+1,fy+1,2,2,SSD1306_BLACK);
      display.display();
    }
    delay(10);
  }
}

// ============================================================
//  GAME 4: PONG
// ============================================================
void gamePong() {
  int ballX=64, ballY=36;
  int bdx=2, bdy=2;
  int pY=28, aiY=28;
  int score=0, aiScore=0;
  unsigned long lastUpdate=millis();
  bool paused=false;
  const int PADDLE_H=14, PADDLE_W=3;

  while(true) {
    if(!paused && millis()-lastUpdate>=40) {
      lastUpdate=millis();
      ButtonAction ba=getButtonAction();
      if(ba==ACTION_UP)   pY=max(12,pY-4);
      if(ba==ACTION_DOWN) pY=min(52,pY+4);
      if(ba==ACTION_RIGHT) paused=true;
      if(ba==ACTION_BACK){ gameOver("Pong",score*10,player.hsPong); g_state=STATE_GAMES_MENU; g_menuSel=3; return; }

      ballX+=bdx; ballY+=bdy;

      // Top/bottom wall
      if(ballY<=11||ballY>=62) bdy=-bdy;

      // Player paddle (left)
      if(ballX<=8 && ballY>=pY && ballY<=pY+PADDLE_H){
        bdx=abs(bdx); int mid=pY+PADDLE_H/2; bdy=(ballY-mid)/3; if(bdy==0) bdy=1;
      }

      // AI paddle (right)
      if(ballX>=118 && ballY>=aiY && ballY<=aiY+PADDLE_H){
        bdx=-abs(bdx);
      }

      // Score
      if(ballX<2){ aiScore++; ballX=64; ballY=36; bdx=2; bdy=random(0,2)?1:-1; }
      if(ballX>126){ score++; player.coins++; ballX=64; ballY=36; bdx=-2; bdy=random(0,2)?1:-1; }

      // AI logic
      int aiMid=aiY+PADDLE_H/2;
      if(aiMid<ballY-1) aiY=min(50,aiY+2);
      if(aiMid>ballY+1) aiY=max(12,aiY-2);

      if(score>=7||aiScore>=7){
        gameOver("Pong",score*10,player.hsPong);
        oledClear(); display.setTextSize(2); display.setCursor(8,10);
        display.println(score>=7?F("YOU WIN!"):F("AI WINS!"));
        display.setTextSize(1); display.setCursor(30,40);
        display.print(score); display.print(F(" - ")); display.print(aiScore);
        display.display(); delay(3000);
        g_state=STATE_GAMES_MENU; g_menuSel=3; return;
      }

      oledClear();
      display.setCursor(40,0); display.print(score); display.print(F("  ")); display.print(aiScore);
      display.drawFastHLine(0,9,128,SSD1306_WHITE);
      display.fillRect(4,pY,PADDLE_W,PADDLE_H,SSD1306_WHITE);
      display.fillRect(121,aiY,PADDLE_W,PADDLE_H,SSD1306_WHITE);
      display.fillRect(ballX,ballY,3,3,SSD1306_WHITE);
      // Center line
      for(int y=10;y<64;y+=5) display.drawPixel(64,y,SSD1306_WHITE);
      display.display();
    }
    if(paused){ gamePause(); paused=false; lastUpdate=millis(); }
    delay(10);
  }
}

// ============================================================
//  GAME 5: MAZE
// ============================================================
// Simple 8-cell maze using walls bitmask
// Cells: 8x4 grid, each 14x12 pixels, walls: N=1 S=2 E=4 W=8
const uint8_t MAZE_W=8, MAZE_H=4;
const uint8_t maze[MAZE_H][MAZE_W] = {
  {6,5,14,5,14,13,6,9},
  {10,6,9,6,11,2,5,4},
  {8,3,12,3,12,7,14,5},
  {12,13,4,13,4,13,4,12}
};
// 6=S|E, 5=N|E... walls: N=1,S=2,E=4,W=8

void gameMaze() {
  int px=0,py=0,ex=MAZE_W-1,ey=MAZE_H-1;
  int moves=0,score=0;
  unsigned long start=millis();
  bool paused=false;

  while(true) {
    // Draw maze
    oledClear();
    display.setCursor(0,0); display.print(F("Maze  M:")); display.print(moves);
    display.print(F(" T:")); display.print((millis()-start)/1000);
    display.drawFastHLine(0,9,128,SSD1306_WHITE);

    for(int row=0;row<MAZE_H;row++){
      for(int col=0;col<MAZE_W;col++){
        int x=col*16, y=10+row*13;
        uint8_t walls=maze[row][col];
        if(walls&1) display.drawFastHLine(x,y,16,SSD1306_WHITE);       // N
        if(walls&2) display.drawFastHLine(x,y+13,16,SSD1306_WHITE);    // S
        if(walls&4) display.drawFastVLine(x+15,y,13,SSD1306_WHITE);    // E
        if(walls&8) display.drawFastVLine(x,y,13,SSD1306_WHITE);       // W
      }
    }

    // Player
    display.fillRect(px*16+5, 10+py*13+4, 6,6, SSD1306_WHITE);
    // Exit star
    display.drawRect(ex*16+4, 10+ey*13+3, 8,8, SSD1306_WHITE);

    display.display();

    if(px==ex && py==ey){
      int timeSec=(millis()-start)/1000;
      score=max(10,500-moves*2-timeSec*3);
      if(score>(int)player.hsMaze){ player.hsMaze=(uint16_t)score; savePrefs(); }
      player.coins+=5; savePrefs();
      showMsg("Maze Solved!", ("Score:"+String(score)).c_str(), 3000);
      g_state=STATE_GAMES_MENU; g_menuSel=4; return;
    }

    ButtonAction ba=getButtonAction();
    uint8_t walls=maze[py][px];

    if(ba==ACTION_UP    && py>0        && !(walls&1)){ py--; moves++; }
    if(ba==ACTION_DOWN  && py<MAZE_H-1 && !(walls&2)){ py++; moves++; }
    if(ba==ACTION_RIGHT && px<MAZE_W-1 && !(walls&4)){ px++; moves++; }
    if(ba==ACTION_LEFT  && px>0        && !(walls&8)){ px--; moves++; }
    if(ba==ACTION_BACK){ g_state=STATE_GAMES_MENU; g_menuSel=4; return; }
    delay(50);
  }
}

// ============================================================
//  GAME 6: FLAPPY BIRD
// ============================================================
void gameFlappy() {
  float by=32, bv=0;
  const float GRAV=0.5, JUMP=-4.0;
  int pipeX[2]={128,192}, pipeGap[2]={20,30};
  int pipeGapY[2]={20,15};
  int score=0;
  unsigned long lastUpdate=millis();
  bool started=false, paused=false;

  oledClear(); display.setTextSize(1); display.setCursor(10,25);
  display.println(F("Press UP to start!"));
  display.display();
  while(getButtonAction()!=ACTION_UP) delay(50);
  started=true;

  while(true) {
    if(!paused && millis()-lastUpdate>=30){
      lastUpdate=millis();
      ButtonAction ba=getButtonAction();
      if(ba==ACTION_UP)    bv=JUMP;
      if(ba==ACTION_DOWN)  paused=true;
      if(ba==ACTION_BACK){ gameOver("Flappy",score,player.hsFlappy); g_state=STATE_GAMES_MENU; g_menuSel=5; return; }

      bv+=GRAV; by+=bv;

      // Pipes move
      for(int i=0;i<2;i++){
        pipeX[i]-=2;
        if(pipeX[i]<-16){
          pipeX[i]=128+random(0,40);
          pipeGap[i]=random(14,26);
          pipeGapY[i]=random(12,30);
          score++; player.coins++;
        }
      }

      // Collision: walls
      if(by>62||by<10){ gameOver("Flappy",score,player.hsFlappy); g_state=STATE_GAMES_MENU; g_menuSel=5; return; }

      // Pipe collision
      for(int i=0;i<2;i++){
        if((int)by+3>=pipeGapY[i] && (int)by<=pipeGapY[i]+pipeGap[i]){ /* in gap */ }
        else if(10>pipeX[i] && 10<pipeX[i]+12){
          gameOver("Flappy",score,player.hsFlappy); g_state=STATE_GAMES_MENU; g_menuSel=5; return;
        }
      }

      oledClear();
      // Ground/sky
      display.drawFastHLine(0,9,128,SSD1306_WHITE);
      display.drawFastHLine(0,63,128,SSD1306_WHITE);

      // Pipes
      for(int i=0;i<2;i++){
        // Top pipe
        display.fillRect(pipeX[i],10,12,pipeGapY[i]-10,SSD1306_WHITE);
        // Bottom pipe
        display.fillRect(pipeX[i],pipeGapY[i]+pipeGap[i],12,64-(pipeGapY[i]+pipeGap[i]),SSD1306_WHITE);
      }

      // Bird
      display.fillCircle(10,(int)by,4,SSD1306_WHITE);
      display.fillCircle(12,(int)by-1,1,SSD1306_BLACK); // eye
      display.drawPixel(14,(int)by,SSD1306_WHITE); // beak

      display.setCursor(50,0); display.print(score);
      display.display();
    }
    if(paused){ gamePause(); paused=false; lastUpdate=millis(); }
    delay(10);
  }
}

// ============================================================
//  GAME 7: 2048
// ============================================================
void game2048() {
  uint16_t grid[4][4]={};
  uint32_t score=0;
  bool changed=true;

  auto addTile = [&](){
    int empty[16][2]; int ec=0;
    for(int r=0;r<4;r++) for(int c=0;c<4;c++) if(!grid[r][c]){ empty[ec][0]=r; empty[ec][1]=c; ec++; }
    if(ec>0){ int idx=random(0,ec); grid[empty[idx][0]][empty[idx][1]]=(random(0,5)<4)?2:4; }
  };

  auto draw2048=[&](){
    oledClear();
    display.setCursor(0,0); display.print(F("2048 Score:")); display.print(score);
    display.drawFastHLine(0,9,128,SSD1306_WHITE);
    for(int r=0;r<4;r++) for(int c=0;c<4;c++){
      int x=c*31+1, y=10+r*14;
      display.drawRect(x,y,30,13,SSD1306_WHITE);
      if(grid[r][c]){
        display.setCursor(x+2,y+3);
        if(grid[r][c]>=1000) display.print(grid[r][c]/1000); // abbreviate
        display.print(grid[r][c]);
      }
    }
    display.display();
  };

  addTile(); addTile();

  while(true) {
    draw2048();
    ButtonAction ba=getButtonAction();
    if(ba==ACTION_BACK){ if((uint32_t)score>player.hs2048){ player.hs2048=(uint32_t)score; savePrefs(); } player.coins+=score/100+1; savePrefs(); g_state=STATE_GAMES_MENU; g_menuSel=6; return; }

    uint16_t old[4][4]; memcpy(old,grid,sizeof(grid));

    if(ba==ACTION_LEFT||ba==ACTION_RIGHT){
      for(int r=0;r<4;r++){
        uint16_t row[4]={};  int ri=0;
        int s=ba==ACTION_LEFT?0:3, step=ba==ACTION_LEFT?1:-1;
        for(int c=s;c>=0&&c<4;c+=step) if(grid[r][c]) row[ri++]=grid[r][c];
        for(int i=0;i<3;i++) if(row[i]&&row[i]==row[i+1]){ row[i]*=2; score+=row[i]; row[i+1]=0; }
        uint16_t row2[4]={}; ri=0;
        for(int i=0;i<4;i++) if(row[i]) row2[ri++]=row[i];
        for(int c=0;c<4;c++) grid[r][ba==ACTION_LEFT?c:3-c]=row2[c];
      }
    }
    if(ba==ACTION_UP||ba==ACTION_DOWN){
      for(int c=0;c<4;c++){
        uint16_t col[4]={}; int ci=0;
        int s=ba==ACTION_UP?0:3, step=ba==ACTION_UP?1:-1;
        for(int r=s;r>=0&&r<4;r+=step) if(grid[r][c]) col[ci++]=grid[r][c];
        for(int i=0;i<3;i++) if(col[i]&&col[i]==col[i+1]){ col[i]*=2; score+=col[i]; col[i+1]=0; }
        uint16_t col2[4]={}; ci=0;
        for(int i=0;i<4;i++) if(col[i]) col2[ci++]=col[i];
        for(int r=0;r<4;r++) grid[ba==ACTION_UP?r:3-r][c]=col2[r];
      }
    }

    if(memcmp(old,grid,sizeof(grid))!=0) addTile();

    // Check 2048
    for(int r=0;r<4;r++) for(int c=0;c<4;c++) if(grid[r][c]==2048){
      if((uint32_t)score>player.hs2048){ player.hs2048=(uint32_t)score; savePrefs(); }
      showMsg("2048 REACHED!",("Score:"+String(score)).c_str(),4000);
      g_state=STATE_GAMES_MENU; g_menuSel=6; return;
    }

    // Check no moves
    bool any=false;
    for(int r=0;r<4;r++) for(int c=0;c<4;c++){
      if(!grid[r][c]) any=true;
      if(c<3&&grid[r][c]==grid[r][c+1]) any=true;
      if(r<3&&grid[r][c]==grid[r+1][c]) any=true;
    }
    if(!any){
      if((uint32_t)score>player.hs2048){ player.hs2048=(uint32_t)score; savePrefs(); }
      showMsg("No moves!",("Score:"+String(score)).c_str(),3000);
      g_state=STATE_GAMES_MENU; g_menuSel=6; return;
    }
    delay(50);
  }
}

// ============================================================
//  GAME 8: TETRIS
// ============================================================
const uint8_t TETRIS_W=10, TETRIS_H=18;
uint8_t tBoard[TETRIS_H][TETRIS_W];

const int8_t PIECES[7][4][2] = {
  {{0,0},{0,1},{0,2},{0,3}},  // I
  {{0,0},{1,0},{0,1},{1,1}},  // O
  {{0,1},{1,0},{1,1},{1,2}},  // T
  {{0,0},{1,0},{1,1},{1,2}},  // L
  {{0,1},{0,2},{1,0},{1,1}},  // S
  {{0,0},{0,1},{1,1},{1,2}},  // Z
  {{0,2},{1,0},{1,1},{1,2}},  // J
};

void gameTetris() {
  memset(tBoard,0,sizeof(tBoard));
  int pieceType=random(0,7);
  int px=4, py=0;
  int8_t piece[4][2];
  int score=0, linesCleared=0;
  unsigned long lastDrop=millis(), lastInput=millis();
  int dropInterval=500;
  bool paused=false;

  auto newPiece=[&](){
    pieceType=random(0,7);
    px=4; py=0;
    for(int i=0;i<4;i++){ piece[i][0]=PIECES[pieceType][i][0]; piece[i][1]=PIECES[pieceType][i][1]; }
  };

  auto canPlace=[&](int offX, int offY)->bool{
    for(int i=0;i<4;i++){
      int r=piece[i][0]+offY+py;
      int c=piece[i][1]+offX+px;
      if(c<0||c>=TETRIS_W||r<0||r>=TETRIS_H) return false;
      if(tBoard[r][c]) return false;
    }
    return true;
  };

  auto placePiece=[&](){
    for(int i=0;i<4;i++) tBoard[piece[i][0]+py][piece[i][1]+px]=1;
  };

  auto clearLines=[&](){
    for(int r=TETRIS_H-1;r>=0;r--){
      bool full=true;
      for(int c=0;c<TETRIS_W;c++) if(!tBoard[r][c]){full=false;break;}
      if(full){
        for(int rr=r;rr>0;rr--) memcpy(tBoard[rr],tBoard[rr-1],TETRIS_W);
        memset(tBoard[0],0,TETRIS_W);
        linesCleared++; score+=100; r++;
        if(dropInterval>100) dropInterval-=10;
      }
    }
  };

  auto rotatePiece=[&](){
    int8_t tmp[4][2];
    for(int i=0;i<4;i++){ tmp[i][0]=piece[i][1]; tmp[i][1]=-piece[i][0]; }
    // Normalize
    int8_t minR=127,minC=127;
    for(int i=0;i<4;i++){ if(tmp[i][0]<minR)minR=tmp[i][0]; if(tmp[i][1]<minC)minC=tmp[i][1]; }
    int8_t test[4][2];
    for(int i=0;i<4;i++){ test[i][0]=tmp[i][0]-minR; test[i][1]=tmp[i][1]-minC; }
    int8_t savePiece[4][2]; memcpy(savePiece,piece,sizeof(piece));
    memcpy(piece,test,sizeof(piece));
    if(!canPlace(0,0)) memcpy(piece,savePiece,sizeof(piece));
  };

  // Init first piece
  for(int i=0;i<4;i++){ piece[i][0]=PIECES[pieceType][i][0]; piece[i][1]=PIECES[pieceType][i][1]; }

  const int BX=46, BY=9, CW=8, CH=3; // Board starts at x=46, y=9, cell 8x3

  auto drawBoard=[&](){
    oledClear();
    display.setCursor(0,0); display.print(F("Tetris"));
    display.setCursor(0,14); display.print(F("Score"));
    display.setCursor(0,24); display.print(score);
    display.setCursor(0,36); display.print(F("Lines"));
    display.setCursor(0,46); display.print(linesCleared);
    display.drawFastHLine(0,9,128,SSD1306_WHITE);
    display.drawRect(BX-1,BY,TETRIS_W*CW+2,TETRIS_H*CH+1,SSD1306_WHITE);

    for(int r=0;r<TETRIS_H;r++) for(int c=0;c<TETRIS_W;c++){
      if(tBoard[r][c]) display.fillRect(BX+c*CW, BY+r*CH, CW-1, CH-1, SSD1306_WHITE);
    }
    for(int i=0;i<4;i++){
      int r=piece[i][0]+py, c=piece[i][1]+px;
      if(r>=0) display.fillRect(BX+c*CW, BY+r*CH, CW-1, CH-1, SSD1306_WHITE);
    }
    display.display();
  };

  while(true) {
    if(!paused) {
      ButtonAction ba=getButtonAction();
      if(ba==ACTION_LEFT  && canPlace(-1,0)) px--;
      if(ba==ACTION_RIGHT && canPlace(1,0))  px++;
      if(ba==ACTION_DOWN  && canPlace(0,1))  py++;
      if(ba==ACTION_UP)   rotatePiece();
      if(ba==ACTION_SELECT || ba==ACTION_SPECIAL) paused=true;
      if(ba==ACTION_BACK){ if(score>(int)player.hsTetris){player.hsTetris=(uint16_t)score;savePrefs();} player.coins+=score/50+1; savePrefs(); g_state=STATE_GAMES_MENU; g_menuSel=7; return; }

      if(millis()-lastDrop>=dropInterval){
        lastDrop=millis();
        if(canPlace(0,1)){ py++; }
        else {
          placePiece(); clearLines(); newPiece();
          if(!canPlace(0,0)){
            if(score>(int)player.hsTetris){player.hsTetris=(uint16_t)score;savePrefs();}
            player.coins+=score/50+1; savePrefs();
            gameOver("Tetris",score,player.hsTetris);
            g_state=STATE_GAMES_MENU; g_menuSel=7; return;
          }
        }
      }
      drawBoard();
    }
    if(paused){ gamePause(); paused=false; lastDrop=millis(); }
    delay(20);
  }
}

// ============================================================
//  BOOT SCREEN
// ============================================================
void showBoot() {
  oledClear();
  display.setTextSize(2); display.setCursor(12,4);
  display.println(F("HOMIE"));
  display.setTextSize(1);
  display.setCursor(2,26); display.println(F("v7.0 FINAL EDITION"));
  display.setCursor(10,38); display.println(F("by Soumyajit"));
  display.setCursor(0,52); display.println(F("Loading..."));
  display.display();
  // Boot progress bar animation
  for(int i=0;i<128;i+=4){
    display.fillRect(0,56,i,6,SSD1306_WHITE);
    display.display(); delay(15);
  }
}

// ============================================================
//  MENU TIMEOUT CHECK
// ============================================================
void checkMenuTimeout() {
  if(player.menuTimeoutMs==0) return;
  if(g_state>=STATE_MAIN_MENU && g_state<=STATE_SETTINGS_MENU){
    if(millis()-g_lastActivity >= player.menuTimeoutMs){
      g_state=STATE_WEATHER; g_menuSel=0;
    }
  }
}

// ============================================================
//  SETUP
// ============================================================
void setup() {
  Serial.begin(115200);

  // Button pins
  pinMode(BTN_UP,    INPUT_PULLUP);
  pinMode(BTN_LEFT,  INPUT_PULLUP);
  pinMode(BTN_RIGHT, INPUT_PULLUP);
  pinMode(BTN_DOWN,  INPUT_PULLUP);

  // I2C + OLED
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println(F("OLED FAILED")); while(true) delay(1000);
  }
  display.clearDisplay(); display.display();

  // Load saved prefs
  loadPrefs();
  applyBrightness();

  // Seed random
  randomSeed(esp_random());

  showBoot();

  // WiFi
  oledClear(); display.setCursor(0,20);
  display.println(F("Connecting WiFi..."));
  display.display();

  g_wifiOk = connectWiFi(player.wifiSSID, player.wifiPass);
  if (g_wifiOk) {
    syncNTP();
    fetchWeather();
  } else {
    showMsg("WiFi Failed","Offline mode",2000);
  }

  g_state = STATE_WEATHER;
  g_lastActivity = millis();
}

// ============================================================
//  MAIN LOOP
// ============================================================
void loop() {
  // Periodic NTP sync and weather refresh
  static unsigned long lastNTP = 0;
  if (g_wifiOk && WiFi.status()==WL_CONNECTED) {
    if (millis()-lastNTP >= 60000) { syncNTP(); lastNTP=millis(); }
    if (millis()-g_lastWeatherFetch >= WEATHER_INTERVAL_MS) fetchWeather();
  }

  ButtonAction action = getButtonAction();
  if (action != ACTION_NONE) g_lastActivity = millis();

  switch (g_state) {

    // ── WEATHER ──────────────────────────────────────────────
    case STATE_WEATHER:
      renderWeather();
      if (action==ACTION_RIGHT||action==ACTION_DOWN) {
        g_state=STATE_MAIN_MENU; g_menuSel=0; g_lastActivity=millis();
      }
      delay(10000); // Refresh every 10s (non-blocking in real use)
      break;

    // ── MAIN MENU ─────────────────────────────────────────────
    case STATE_MAIN_MENU:
      renderMenu("HOMIE v7.0", MAIN_MENU, MAIN_MENU_COUNT, g_menuSel);
      if (action==ACTION_UP)   g_menuSel=(g_menuSel-1+MAIN_MENU_COUNT)%MAIN_MENU_COUNT;
      if (action==ACTION_DOWN) g_menuSel=(g_menuSel+1)%MAIN_MENU_COUNT;
      if (action==ACTION_RIGHT||action==ACTION_SELECT) {
        switch(g_menuSel){
          case 0: g_state=STATE_WEATHER; break;
          case 1: g_state=STATE_GAMES_MENU;    g_menuSel=0; break;
          case 2: g_state=STATE_STUDY_MENU;    g_menuSel=0; break;
          case 3: g_state=STATE_SHOP;          g_menuSel=0; break;
          case 4: g_state=STATE_APPS_MENU;     g_menuSel=0; break;
          case 5: g_state=STATE_SETTINGS_MENU; g_menuSel=0; break;
        }
      }
      if (action==ACTION_LEFT||action==ACTION_BACK) g_state=STATE_WEATHER;
      checkMenuTimeout();
      break;

    // ── GAMES MENU ────────────────────────────────────────────
    case STATE_GAMES_MENU:
      renderMenu("Games", GAMES_MENU, GAMES_MENU_COUNT, g_menuSel);
      if (action==ACTION_UP)   g_menuSel=(g_menuSel-1+GAMES_MENU_COUNT)%GAMES_MENU_COUNT;
      if (action==ACTION_DOWN) g_menuSel=(g_menuSel+1)%GAMES_MENU_COUNT;
      if (action==ACTION_RIGHT||action==ACTION_SELECT) {
        switch(g_menuSel){
          case 0: gameSpaceWar(); break;
          case 1: gameCarGame();  break;
          case 2: gameSnake();    break;
          case 3: gamePong();     break;
          case 4: gameMaze();     break;
          case 5: gameFlappy();   break;
          case 6: game2048();     break;
          case 7: gameTetris();   break;
          case 8: renderHighScores(); break;
        }
      }
      if (action==ACTION_LEFT||action==ACTION_BACK) { g_state=STATE_MAIN_MENU; g_menuSel=1; }
      checkMenuTimeout();
      break;

    // ── STUDY MENU ────────────────────────────────────────────
    case STATE_STUDY_MENU:
      renderMenu("Study", STUDY_MENU, STUDY_MENU_COUNT, g_menuSel);
      if (action==ACTION_UP)   g_menuSel=(g_menuSel-1+STUDY_MENU_COUNT)%STUDY_MENU_COUNT;
      if (action==ACTION_DOWN) g_menuSel=(g_menuSel+1)%STUDY_MENU_COUNT;
      if (action==ACTION_RIGHT||action==ACTION_SELECT) {
        switch(g_menuSel){
          case 0: runMCQMode();      break;
          case 1: runFactsMode();    break;
          case 2: renderStudyStats(); break;
        }
      }
      if (action==ACTION_LEFT||action==ACTION_BACK) { g_state=STATE_MAIN_MENU; g_menuSel=2; }
      checkMenuTimeout();
      break;

    // ── SHOP ──────────────────────────────────────────────────
    case STATE_SHOP:
      renderShop(); // Self-contained loop
      break;

    // ── APPS MENU ─────────────────────────────────────────────
    case STATE_APPS_MENU:
      renderMenu("System Apps", APPS_MENU, APPS_MENU_COUNT, g_menuSel);
      if (action==ACTION_UP)   g_menuSel=(g_menuSel-1+APPS_MENU_COUNT)%APPS_MENU_COUNT;
      if (action==ACTION_DOWN) g_menuSel=(g_menuSel+1)%APPS_MENU_COUNT;
      if (action==ACTION_RIGHT||action==ACTION_SELECT) {
        switch(g_menuSel){
          case 0: renderTimer();    break;
          case 1: renderGoals();    break;
          case 2: renderCoinFlip(); break;
          case 3: renderFortune();  break;
          case 4: renderMood();     break;
          case 5: renderHealth();   break;
          case 6: renderScanner();  break;
          case 7: renderKeyboard(); break;
          case 8: runSleepMode();   break;
        }
      }
      if (action==ACTION_LEFT||action==ACTION_BACK) { g_state=STATE_MAIN_MENU; g_menuSel=4; }
      checkMenuTimeout();
      break;

    // ── SETTINGS MENU ─────────────────────────────────────────
    case STATE_SETTINGS_MENU:
      renderMenu("Settings", SETTINGS_MENU, SETTINGS_MENU_COUNT, g_menuSel);
      if (action==ACTION_UP)   g_menuSel=(g_menuSel-1+SETTINGS_MENU_COUNT)%SETTINGS_MENU_COUNT;
      if (action==ACTION_DOWN) g_menuSel=(g_menuSel+1)%SETTINGS_MENU_COUNT;
      if (action==ACTION_RIGHT||action==ACTION_SELECT) {
        switch(g_menuSel){
          case 0: renderSettingsBrightness(); break;
          case 1: renderSettingsTimeout();    break;
          case 2: renderSettingsWifi();       break;
          case 3: renderSettingsWifi();       break; // same screen shows pass hint
          case 4: renderSettingsReset();      break;
          case 5: renderAbout();              break;
        }
      }
      if (action==ACTION_LEFT||action==ACTION_BACK) { g_state=STATE_MAIN_MENU; g_menuSel=5; }
      checkMenuTimeout();
      break;

    default:
      g_state=STATE_WEATHER;
      break;
  }

  delay(20);
}