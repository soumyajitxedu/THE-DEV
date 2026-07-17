#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <time.h>

// ====================================================
//  CONFIGURATION
// ====================================================
const char* API_KEY   = "api !"; // replace later
const char* LOCATION  = "Dhulian,India";
const char* WIFI_SSID = "C4F143FED775F4E1705DB899EF02";
const char* WIFI_PASS = "1915211325110920";

// ====================================================
//  OLED
// ====================================================
#define SCREEN_W    128
#define SCREEN_H     64
#define OLED_ADDR   0x3C
#define SDA_PIN      21
#define SCL_PIN      22

Adafruit_SSD1306 display(SCREEN_W, SCREEN_H, &Wire, -1);

// ====================================================
//  TIMING CONSTANTS  (all in milliseconds)
// ====================================================
static const unsigned long AWAKE_MS        =  2UL * 60000UL; // 2 min ON
static const unsigned long CYCLE_MS        =  5UL * 60000UL; // full 5 min cycle
static const unsigned long RETRY_MS        =       30000UL;  // 30 s retry on fail
static const unsigned long CLOCK_TICK_MS   =       60000UL;  // soft-clock refresh

// ====================================================
//  NTP
// ====================================================
const long  GMT_OFFSET_SEC  = 19800;   // IST  UTC+5:30
const int   DST_OFFSET_SEC  = 0;

// ====================================================
//  WEATHER DATA
// ====================================================
float   g_temp      = 0.0f;
int     g_humidity  = 0;
float   g_feels     = 0.0f;
int     g_windKph   = 0;
String  g_desc      = "---";
String  g_icon      = "cloud";
String  g_time      = "--:--";
String  g_date      = "---";
bool    g_hasData   = false;

// ====================================================
//  STATE FLAGS
// ====================================================
bool g_wifiOk       = false;
bool g_screenOn     = true;

// ====================================================
//  16×16 WEATHER ICONS  (PROGMEM bitmaps)
// ====================================================
const unsigned char ICO_SUN[] PROGMEM = {
  0x02,0x40,0x06,0x60,0x04,0x20,0x00,0x00,
  0x11,0x88,0x39,0x9C,0x7F,0xFE,0xFF,0xFF,
  0xFF,0xFF,0x7F,0xFE,0x39,0x9C,0x11,0x88,
  0x00,0x00,0x04,0x20,0x06,0x60,0x02,0x40
};
const unsigned char ICO_CLOUD[] PROGMEM = {
  0x00,0x00,0x00,0x00,0x0F,0x00,0x1F,0x80,
  0x30,0xC0,0x20,0x40,0x60,0x60,0x47,0xE0,
  0xCF,0xF0,0xFF,0xF8,0xFF,0xF8,0xFF,0xF8,
  0xFF,0xF8,0x7F,0xF0,0x00,0x00,0x00,0x00
};
const unsigned char ICO_RAIN[] PROGMEM = {
  0x00,0x00,0x0F,0x00,0x1F,0x80,0x30,0xC0,
  0x60,0x60,0x47,0xE0,0xCF,0xF0,0xFF,0xF8,
  0xFF,0xF8,0x7F,0xF0,0x00,0x00,0x22,0x44,
  0x11,0x22,0x22,0x44,0x11,0x22,0x00,0x00
};
const unsigned char ICO_NIGHT[] PROGMEM = {
  0x00,0x00,0x07,0x00,0x0F,0x80,0x1F,0x80,
  0x3F,0x00,0x3E,0x00,0x7C,0x1E,0x7C,0x3F,
  0x78,0x3F,0x78,0x3F,0x7C,0x3F,0x3C,0x1E,
  0x3E,0x00,0x1F,0x00,0x0F,0x80,0x03,0x00
};
const unsigned char ICO_THUNDER[] PROGMEM = {
  0x00,0x00,0x0F,0x00,0x1F,0x80,0x30,0xC0,
  0x60,0x60,0xCF,0xF0,0xFF,0xF8,0xFF,0xF8,
  0x7F,0xF0,0x0F,0x00,0x1E,0x00,0x3C,0x00,
  0x18,0x00,0x38,0x00,0x30,0x00,0x00,0x00
};
const unsigned char ICO_MIST[] PROGMEM = {
  0x00,0x00,0x00,0x00,0x7F,0xFE,0x00,0x00,
  0x00,0x00,0x7F,0xFE,0x00,0x00,0x00,0x00,
  0x7F,0xFE,0x00,0x00,0x00,0x00,0x7F,0xFE,
  0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00
};

// ====================================================
//  FORWARD DECLARATIONS
// ====================================================
void     oledOn();
void     oledOff();
void     oledMsg(const char* l1, const char* l2 = "", int waitMs = 0);
void     oledError(const char* title, const char* detail, int waitMs = 2000);
void     showBoot();
void     scanNetworks();
bool     connectWiFi();
bool     syncNTP();
String   jsonExtract(const String& json, const String& key);
String   chooseIcon(const String& cond, int isDay);
void     renderWeather();
bool     fetchWeather();
void     softClockTick();

// ====================================================
// ===          OLED  CONTROL HELPERS               ===
// ====================================================

// Hardware ON via SSD1306 command register
void oledOn() {
  if (!g_screenOn) {
    Wire.beginTransmission(OLED_ADDR);
    Wire.write(0x00);              // Co=0, D/C#=0 → command byte
    Wire.write(0xAF);              // SSD1306_DISPLAYON
    Wire.endTransmission();
    g_screenOn = true;
    Serial.println(F("[OLED] Display ON"));
  }
}

// Hardware OFF via SSD1306 command register
void oledOff() {
  if (g_screenOn) {
    display.clearDisplay();
    display.display();             // blank first
    Wire.beginTransmission(OLED_ADDR);
    Wire.write(0x00);
    Wire.write(0xAE);              // SSD1306_DISPLAYOFF
    Wire.endTransmission();
    g_screenOn = false;
    Serial.println(F("[OLED] Display OFF (sleep)"));
  }
}

void oledMsg(const char* l1, const char* l2, int waitMs) {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 18);
  display.println(l1);
  if (l2 && strlen(l2)) display.println(l2);
  display.display();
  if (waitMs > 0) delay(waitMs);
}

void oledError(const char* title, const char* detail, int waitMs) {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println(F("!!! ERROR !!!"));
  display.println();
  display.println(title);
  if (detail && strlen(detail)) display.println(detail);
  display.display();
  Serial.print(F("[ERR] "));
  Serial.print(title);
  Serial.print(F(" | "));
  Serial.println(detail);
  if (waitMs > 0) delay(waitMs);
}

// ====================================================
// ===              BOOT  SCREEN                    ===
// ====================================================
void showBoot() {
  // --- Title frame ---
  display.clearDisplay();
  display.drawRect(0, 0, 128, 64, SSD1306_WHITE);
  display.setTextSize(2);
  display.setCursor(8, 8);
  display.println(F("WEATHER"));
  display.setTextSize(1);
  display.setCursor(20, 34);
  display.println(F("STATION  v3.0"));
  display.setCursor(16, 48);
  display.println(F("ESP32 + SSD1306"));
  display.display();
  delay(2500);

  // --- Progress bar ---
  display.clearDisplay();
  display.setTextSize(1);
  display.setCursor(28, 6);
  display.println(F("Initializing..."));
  for (int p = 0; p <= 100; p += 4) {
    int barW = p;                  // 100 px wide bar maximum
    display.fillRect(14, 30, barW, 10, SSD1306_WHITE);
    display.drawRect(12, 28, 104, 14, SSD1306_WHITE);
    display.fillRect(46, 48, 36, 10, SSD1306_BLACK);
    display.setCursor(52, 50);
    display.print(p);
    display.print(F("%"));
    display.display();
    delay(30);
  }
  delay(400);

  // --- System checks ---
  display.clearDisplay();
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println(F("System Check:"));
  display.println();
  const char* chk[] = {
    "OLED  ....... OK",
    "Memory ...... OK",
    "WiFi  ....... --"
  };
  for (int i = 0; i < 3; i++) {
    display.println(chk[i]);
    display.display();
    delay(350);
  }
  delay(600);
}

// ====================================================
// ===           WIFI  SCAN  &  CONNECT             ===
// ====================================================
void scanNetworks() {
  Serial.println(F("\n=== WiFi Scan ==="));
  oledMsg("Scanning WiFi...");
  int n = WiFi.scanNetworks();
  if (n <= 0) {
    Serial.println(F("No networks found."));
  } else {
    Serial.printf("%d networks found:\n", n);
    Serial.println(F("  #  RSSI  SSID"));
    Serial.println(F("  -  ----  --------------------"));
    for (int i = 0; i < n; i++) {
      Serial.printf("  %d  %4d  %s\n", i + 1, WiFi.RSSI(i), WiFi.SSID(i).c_str());
    }
  }
  Serial.println(F("=================\n"));
  WiFi.scanDelete();
}

bool connectWiFi() {
  scanNetworks();

  Serial.print(F("Connecting to: "));
  Serial.println(WIFI_SSID);

  display.clearDisplay();
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println(F("Connecting WiFi:"));
  display.println();
  // Truncate SSID if long
  String s = String(WIFI_SSID);
  if (s.length() > 20) s = s.substring(0, 17) + F("...");
  display.println(s);
  display.display();

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  int  tries = 0;
  int  dots  = 0;
  while (tries < 40 && WiFi.status() != WL_CONNECTED) {
    display.fillRect(0, 46, 128, 16, SSD1306_BLACK);
    display.setCursor(0, 50);
    display.print(F("Connecting"));
    for (int d = 0; d < (dots % 4); d++) display.print('.');
    display.display();
    delay(500);
    Serial.print('.');
    tries++;
    dots++;
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print(F("Connected! IP: "));
    Serial.print(WiFi.localIP());
    Serial.print(F("  RSSI: "));
    Serial.print(WiFi.RSSI());
    Serial.println(F(" dBm"));

    display.clearDisplay();
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.println(F("WiFi Connected!"));
    display.println();
    display.print(F("IP: "));
    display.println(WiFi.localIP().toString());
    display.print(F("RSSI: "));
    display.print(WiFi.RSSI());
    display.println(F(" dBm"));
    display.display();
    delay(2000);
    return true;
  }

  oledError("WiFi Failed", "Check SSID/Pass", 3000);
  return false;
}

// ====================================================
// ===              NTP  SYNC                       ===
// ====================================================
bool syncNTP() {
  oledMsg("Syncing time...");
  configTime(GMT_OFFSET_SEC, DST_OFFSET_SEC, "pool.ntp.org", "time.nist.gov");
  struct tm ti;
  for (int i = 0; i < 20; i++) {   // up to 10 s
    if (getLocalTime(&ti, 500)) {
      char tb[8], db[14];
      strftime(tb, sizeof(tb), "%H:%M",    &ti);
      strftime(db, sizeof(db), "%d %b %Y", &ti);
      g_time = tb;
      g_date = db;
      Serial.print(F("NTP OK: "));
      Serial.print(g_time);
      Serial.print(F("  "));
      Serial.println(g_date);
      return true;
    }
    Serial.print('.');
    delay(500);
  }
  Serial.println(F("\nNTP failed."));
  return false;
}

// ====================================================
// ===          JSON  FIELD  EXTRACTOR              ===
// ====================================================
String jsonExtract(const String& json, const String& key) {
  // Search for "key":
  String token = "\"";
  token += key;
  token += "\":";

  int ki = json.indexOf(token);
  if (ki < 0) return F("");

  int si = ki + token.length();
  // skip whitespace
  while (si < (int)json.length() && json[si] == ' ') si++;
  if (si >= (int)json.length()) return F("");

  char fc = json[si];

  if (fc == '"') {
    // quoted string
    si++;
    int ei = json.indexOf('"', si);
    return (ei < 0) ? String("") : json.substring(si, ei);
  }
  if (fc == 't') return F("1");   // true  → 1
  if (fc == 'f') return F("0");   // false → 0
  if (fc == 'n') return F("");    // null

  // numeric
  int ei = si;
  while (ei < (int)json.length()) {
    char c = json[ei];
    if (isDigit(c) || c == '.' || c == '-') ei++;
    else break;
  }
  return json.substring(si, ei);
}

// ====================================================
// ===          ICON  SELECTION                     ===
// ====================================================
String chooseIcon(const String& cond, int isDay) {
  if (isDay == 0) return F("night");
  String c = cond;
  c.toLowerCase();
  if (c.indexOf(F("thunder")) >= 0 || c.indexOf(F("storm"))   >= 0) return F("thunder");
  if (c.indexOf(F("rain"))    >= 0 || c.indexOf(F("drizzle")) >= 0 ||
      c.indexOf(F("shower"))  >= 0)                                  return F("rain");
  if (c.indexOf(F("mist"))    >= 0 || c.indexOf(F("fog"))     >= 0 ||
      c.indexOf(F("haze"))    >= 0)                                  return F("mist");
  if (c.indexOf(F("cloud"))   >= 0 || c.indexOf(F("overcast")) >= 0) return F("cloud");
  if (c.indexOf(F("sunny"))   >= 0 || c.indexOf(F("clear"))   >= 0) return F("sun");
  return F("cloud");
}

// ====================================================
// ===          SOFT  CLOCK  TICK                   ===
// ====================================================
// Call this every loop iteration; updates g_time every 60 s
// without an API call, then re-renders the display.
void softClockTick() {
  static unsigned long lastTick = 0;
  if (millis() - lastTick < CLOCK_TICK_MS) return;
  lastTick = millis();

  struct tm ti;
  if (getLocalTime(&ti, 300)) {
    char tb[8];
    strftime(tb, sizeof(tb), "%H:%M", &ti);
    g_time = tb;
  }
  if (g_screenOn && g_hasData) renderWeather();
}

// ====================================================
// ===          RENDER  WEATHER  SCREEN             ===
// ====================================================
void renderWeather() {
  display.clearDisplay();

  // --- Icon (top-right 16×16) ---
  const unsigned char* bmp = ICO_CLOUD;
  if      (g_icon == F("sun"))     bmp = ICO_SUN;
  else if (g_icon == F("rain"))    bmp = ICO_RAIN;
  else if (g_icon == F("night"))   bmp = ICO_NIGHT;
  else if (g_icon == F("thunder")) bmp = ICO_THUNDER;
  else if (g_icon == F("mist"))    bmp = ICO_MIST;
  display.drawBitmap(110, 0, bmp, 16, 16, SSD1306_WHITE);

  // --- Time (large, top-left) ---
  display.setTextSize(2);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.print(g_time);

  // --- Date ---
  display.setTextSize(1);
  display.setCursor(0, 18);
  display.print(g_date);

  // --- Separator line ---
  display.drawFastHLine(0, 27, 128, SSD1306_WHITE);

  // --- Temperature (large) ---
  display.setTextSize(2);
  display.setCursor(0, 30);
  display.print(g_temp, 1);
  display.print((char)247);   // °
  display.print('C');

  // --- Feels like (right side) ---
  display.setTextSize(1);
  display.setCursor(76, 30);
  display.print(F("Feels"));
  display.setCursor(76, 40);
  display.print(g_feels, 0);
  display.print((char)247);
  display.print('C');

  // --- Humidity & Wind ---
  display.setTextSize(1);
  display.setCursor(0, 50);
  display.print(F("H:"));
  display.print(g_humidity);
  display.print(F("% W:"));
  display.print(g_windKph);
  display.print(F("kph"));

  // --- WiFi signal indicator ---
  if (WiFi.status() == WL_CONNECTED) {
    int rssi = WiFi.RSSI();
    display.setCursor(100, 56);
    if      (rssi > -55) display.print(F("[3]"));
    else if (rssi > -70) display.print(F("[2]"));
    else                 display.print(F("[1]"));
  }

  display.display();
}

// ====================================================
// ===          FETCH  WEATHER  (HTTPS)             ===
// ====================================================
bool fetchWeather() {
  if (!g_wifiOk) {
    oledError("No WiFi", "Cannot fetch", 2000);
    return false;
  }

  Serial.println(F("\n[API] Fetching weather via HTTPS..."));
  oledMsg("Fetching data...", "", 0);

  // Build HTTPS URL
  String url = F("https://api.weatherapi.com/v1/current.json?key=");
  url += API_KEY;
  url += F("&q=");
  url += LOCATION;
  url += F("&aqi=no");

  // Use WiFiClientSecure — skip cert verification (fine for this use-case)
  WiFiClientSecure client;
  client.setInsecure();           // no root-CA needed

  HTTPClient http;
  http.begin(client, url);
  http.setTimeout(15000);
  http.addHeader(F("Accept"), F("application/json"));

  int code = http.GET();
  Serial.print(F("[API] HTTP code: "));
  Serial.println(code);

  if (code <= 0) {
    String e = "Code:" + String(code);
    oledError("HTTP Failed", e.c_str(), 2000);
    http.end();
    return false;
  }
  if (code == 401 || code == 403) {
    oledError("Bad API Key", "Check config", 4000);
    http.end();
    return false;
  }
  if (code != 200) {
    String e = "HTTP " + String(code);
    oledError("Server Err", e.c_str(), 2000);
    http.end();
    return false;
  }

  String payload = http.getString();
  http.end();

  if (payload.length() == 0) {
    oledError("Empty payload", "", 2000);
    return false;
  }

  Serial.print(F("[API] Payload len: "));
  Serial.println(payload.length());

  // --- Parse ---
  String tStr = jsonExtract(payload, F("temp_c"));
  String hStr = jsonExtract(payload, F("humidity"));
  String fStr = jsonExtract(payload, F("feelslike_c"));
  String wStr = jsonExtract(payload, F("wind_kph"));
  String dStr = jsonExtract(payload, F("text"));
  String iStr = jsonExtract(payload, F("is_day"));
  payload = "";                   // free RAM immediately

  if (tStr.length() == 0 || hStr.length() == 0) {
    oledError("Parse Err", "Missing fields", 2000);
    return false;
  }

  g_temp     = tStr.toFloat();
  g_humidity = hStr.toInt();
  g_feels    = fStr.toFloat();
  g_windKph  = (int)wStr.toFloat();
  g_desc     = (dStr.length() > 0) ? dStr : F("Unknown");
  g_icon     = chooseIcon(g_desc, iStr.toInt());
  g_hasData  = true;

  // Refresh time while we're at it
  struct tm ti;
  if (getLocalTime(&ti, 500)) {
    char tb[8], db[14];
    strftime(tb, sizeof(tb), "%H:%M",    &ti);
    strftime(db, sizeof(db), "%d %b %Y", &ti);
    g_time = tb;
    g_date = db;
  }

  Serial.printf("[API] Temp:%.1f  Hum:%d  Feels:%.1f  Wind:%d  Cond:%s  Icon:%s\n",
                g_temp, g_humidity, g_feels, g_windKph,
                g_desc.c_str(), g_icon.c_str());
  return true;
}

// ====================================================
// ===                  SETUP                       ===
// ====================================================
void setup() {
  Serial.begin(115200);
  delay(800);
  Serial.println(F("\n=== Weather Station v3.0 ==="));

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);

  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println(F("OLED INIT FAILED — check wiring!"));
    while (true) delay(1000);
  }
  display.setTextColor(SSD1306_WHITE);
  display.clearDisplay();
  display.display();
  g_screenOn = true;

  showBoot();

  g_wifiOk = connectWiFi();
  if (!g_wifiOk) {
    oledError("No WiFi", "Restart device", 0);
    // Keep alive — still tracks millis for retry
  } else {
    syncNTP();
    bool ok = fetchWeather();
    if (ok) {
      renderWeather();
    } else {
      oledMsg("Data fetch fail", "Will retry...", 2000);
    }
  }
}

// ====================================================
// ===                  LOOP                        ===
// ====================================================
/*
  5-minute duty cycle:
  ┌─────────────────────────────────────────┐
  │  0:00 → 2:00  AWAKE  (screen ON)        │
  │   ↳ API fetch at cycle start            │
  │   ↳ soft-clock tick every 60 s          │
  │   ↳ retry every 30 s if fetch failed    │
  │  2:00 → 5:00  SLEEP  (screen OFF)       │
  │   ↳ ESP32 awake, NTP ticks silently     │
  │  5:00 → repeat                          │
  └─────────────────────────────────────────┘
*/
void loop() {
  // ── WiFi watchdog ──────────────────────────────────
  if (WiFi.status() != WL_CONNECTED) {
    if (g_wifiOk) {
      g_wifiOk = false;
      Serial.println(F("[WiFi] Lost connection!"));
      if (g_screenOn) oledError("WiFi Lost", "Reconnecting...", 1000);
    }
    g_wifiOk = connectWiFi();
    if (g_wifiOk) syncNTP();
    delay(5000);
    return;
  }
  g_wifiOk = true;

  // ── Cycle timing ───────────────────────────────────
  static unsigned long cycleStart    = 0;      // when the current 5-min cycle began
  static bool          fetchDone     = false;  // did we successfully fetch this cycle?
  static unsigned long lastRetry     = 0;      // last retry timestamp
  static bool          inSleep       = false;  // are we in sleep phase?

  unsigned long now     = millis();
  unsigned long elapsed = now - cycleStart;

  // ── Detect new cycle ───────────────────────────────
  if (elapsed >= CYCLE_MS) {
    cycleStart = now;
    elapsed    = 0;
    fetchDone  = false;
    inSleep    = false;
    Serial.println(F("\n[CYCLE] New 5-min cycle started."));

    // Wake screen if it was off
    if (!g_screenOn) {
      oledOn();
    }
  }

  // ── AWAKE PHASE  (0 → 2 min) ───────────────────────
  if (elapsed < AWAKE_MS) {
    // Ensure screen is on
    if (!g_screenOn) {
      oledOn();
    }
    inSleep = false;

    // Attempt one successful fetch at cycle start (or retry every 30 s)
    if (!fetchDone) {
      bool shouldFetch = (lastRetry == 0) ||            // very first attempt
                         (now - lastRetry >= RETRY_MS); // retry after 30 s

      if (shouldFetch) {
        lastRetry = now;
        Serial.println(F("[CYCLE] Awake: attempting API fetch..."));
        bool ok = fetchWeather();
        if (ok) {
          fetchDone = true;
          renderWeather();
          Serial.println(F("[CYCLE] Fetch SUCCESS."));
        } else {
          Serial.println(F("[CYCLE] Fetch FAILED — will retry in 30 s."));
        }
      }
    }

    // Soft clock tick (updates time on screen every 60 s)
    softClockTick();
  }

  // ── SLEEP PHASE  (2 → 5 min) ───────────────────────
  else {
    if (!inSleep) {
      inSleep = true;
      oledOff();
      Serial.println(F("[CYCLE] Sleep phase — screen OFF."));
    }

    // Silent NTP tick (keeps g_time accurate while screen is off)
    struct tm ti;
    static unsigned long lastSilentTick = 0;
    if (now - lastSilentTick >= CLOCK_TICK_MS) {
      lastSilentTick = now;
      if (getLocalTime(&ti, 300)) {
        char tb[8];
        strftime(tb, sizeof(tb), "%H:%M", &ti);
        g_time = tb;
        Serial.print(F("[SLEEP] Silent tick: "));
        Serial.println(g_time);
      }
    }
  }

  // ── Small yield delay (prevent WDT) ────────────────
  delay(500);
}