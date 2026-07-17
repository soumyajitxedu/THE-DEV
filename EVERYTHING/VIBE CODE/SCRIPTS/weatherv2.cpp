// ============================================================
//  ESP32 ULTIMATE STUDY & WEATHER STATION
//  By Soumyajit | v4.0
//  Board: ESP32 Dev Module
//  Display: 0.96" SSD1306 OLED (128x64, I2C)
// ============================================================

#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <time.h>

// ============================================================
//  CONFIGURATION
// ============================================================
const char* API_KEY   = "d6ccc068b65f4cee99f104921261707";
const char* LOCATION  = "Dhulian,India";
const char* WIFI_SSID = "C4F143FED775F4E1705DB899EF02";
const char* WIFI_PASS = "1915211325110920";

// ============================================================
//  OLED
// ============================================================
#define SCREEN_W   128
#define SCREEN_H    64
#define OLED_ADDR 0x3C
#define SDA_PIN     21
#define SCL_PIN     22
Adafruit_SSD1306 display(SCREEN_W, SCREEN_H, &Wire, -1);

// ============================================================
//  TIMING
// ============================================================
const long          GMT_OFFSET_SEC      = 19800;
const int           DST_OFFSET_SEC      = 0;
const unsigned long WEATHER_INTERVAL_MS = 1800000UL; // 30 min
const unsigned long CLOCK_TICK_MS       =   60000UL; // 1 min
const unsigned long WIFI_RETRY_DELAY_MS =    5000UL;

// ============================================================
//  WEATHER DATA
// ============================================================
float  g_temp     = 0.0f;
int    g_humidity = 0;
float  g_feels    = 0.0f;
int    g_windKph  = 0;
String g_desc     = "---";
String g_icon     = "cloud";
String g_time     = "--:--";
String g_date     = "---";
bool   g_hasData  = false;
bool   g_wifiOk   = false;

// ============================================================
//  WEATHER ICONS  16x16 PROGMEM BITMAPS
// ============================================================
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

// ============================================================
//  32 MOTIVATIONAL QUOTES  (PROGMEM)
// ============================================================
const char Q00[] PROGMEM = "It's all yours";
const char Q01[] PROGMEM = "Be kind";
const char Q02[] PROGMEM = "Sleeping is the best way to escape reality";
const char Q03[] PROGMEM = "I am built by code, made by Soumyajit";
const char Q04[] PROGMEM = "Don't awake too much at night";
const char Q05[] PROGMEM = "Boards only few months left! Think!";
const char Q06[] PROGMEM = "You're weak in math! Fix it.";
const char Q07[] PROGMEM = "Your morals make you weak";
const char Q08[] PROGMEM = "Don't be the villain of your own story";
const char Q09[] PROGMEM = "You were born for a purpose! Do it now!";
const char Q10[] PROGMEM = "You will see more darkness in the future";
const char Q11[] PROGMEM = "Don't waste the day!";
const char Q12[] PROGMEM = "Coffee??? :)";
const char Q13[] PROGMEM = "Wanna be yours";
const char Q14[] PROGMEM = "You don't need GF, it's just hormones";
const char Q15[] PROGMEM = "Don't ruin the day!";
const char Q16[] PROGMEM = "Wait! You got distracted again";
const char Q17[] PROGMEM = "Why do we exist?";
const char Q18[] PROGMEM = "I'd rather die standing than live kneeling";
const char Q19[] PROGMEM = "It's gonna be okay";
const char Q20[] PROGMEM = "Not decreasing! Your level is increasing!";
const char Q21[] PROGMEM = "There is always someone behind you";
const char Q22[] PROGMEM = "Are you ok?";
const char Q23[] PROGMEM = "Study bro! Every second counts";
const char Q24[] PROGMEM = "Code is poetry written in logic";
const char Q25[] PROGMEM = "Best error: the one that never shows up";
const char Q26[] PROGMEM = "Future built by today, not tomorrow";
const char Q27[] PROGMEM = "Even the darkest night ends at sunrise";
const char Q28[] PROGMEM = "One line of code can change everything";
const char Q29[] PROGMEM = "Don't count days, make the days count";
const char Q30[] PROGMEM = "Remember why you started this project";
const char Q31[] PROGMEM = "You are the main character of your journey";

const char* const QUOTES[] PROGMEM = {
  Q00,Q01,Q02,Q03,Q04,Q05,Q06,Q07,
  Q08,Q09,Q10,Q11,Q12,Q13,Q14,Q15,
  Q16,Q17,Q18,Q19,Q20,Q21,Q22,Q23,
  Q24,Q25,Q26,Q27,Q28,Q29,Q30,Q31
};
#define QUOTE_COUNT 32

// ============================================================
//  7 APOLOGY MESSAGES  (PROGMEM)
// ============================================================
const char AP0[] PROGMEM = "Ohh no... I can't find her... I meant internet!";
const char AP1[] PROGMEM = "Sadly, I can't find internet...";
const char AP2[] PROGMEM = "Where is internet? Did it run away?";
const char AP3[] PROGMEM = "WiFi signal: gone, reduced to atoms.";
const char AP4[] PROGMEM = "Did you forget to turn on the router?";
const char AP5[] PROGMEM = "Internet... we had something special.";
const char AP6[] PROGMEM = "Maybe check the cable? Wait, I have none.";
const char* const APOLOGIES[] PROGMEM = {
  AP0,AP1,AP2,AP3,AP4,AP5,AP6
};
#define APOLOGY_COUNT 7

// ============================================================
//  MCQ DATA  (50 Questions embedded — subset shown for size)
//  Full 100 stored as structs in PROGMEM
// ============================================================
struct MCQ {
  const char* question;
  const char* optA;
  const char* optB;
  const char* optC;
  const char* optD;
  uint8_t     answer;   // 0=A 1=B 2=C 3=D
  const char* explain;
};

// --- Geography / Soil MCQs (Q0–Q12) ---
const char mq0[]  PROGMEM = "Which soil is classified as ex-situ (transported)?";
const char mo0a[] PROGMEM = "Alluvial"; const char mo0b[] PROGMEM = "Black";
const char mo0c[] PROGMEM = "Red";      const char mo0d[] PROGMEM = "Laterite";
const char me0[]  PROGMEM = "Alluvial soil is transported by rivers from parent rock.";

const char mq1[]  PROGMEM = "Khadar soil compared to Bhangar is:";
const char mo1a[] PROGMEM = "Older, coarser";     const char mo1b[] PROGMEM = "Newer, finer, fertile";
const char mo1c[] PROGMEM = "In-situ, volcanic";  const char mo1d[] PROGMEM = "Leached, acidic";
const char me1[]  PROGMEM = "Khadar is new alluvium: fine, fertile, flood-replenished.";

const char mq2[]  PROGMEM = "What gives Red Soil its red color?";
const char mo2a[] PROGMEM = "Alumina";  const char mo2b[] PROGMEM = "Magnesium";
const char mo2c[] PROGMEM = "Iron oxide"; const char mo2d[] PROGMEM = "Silica";
const char me2[]  PROGMEM = "Wide diffusion of iron oxide (Fe2O3) causes the red color.";

const char mq3[]  PROGMEM = "Which soil is called 'self-ploughing'?";
const char mo3a[] PROGMEM = "Alluvial"; const char mo3b[] PROGMEM = "Red";
const char mo3c[] PROGMEM = "Black";    const char mo3d[] PROGMEM = "Laterite";
const char me3[]  PROGMEM = "Black soil cracks deeply when dry, aerating itself.";

const char mq4[]  PROGMEM = "Laterite soil forms primarily by:";
const char mo4a[] PROGMEM = "Leaching"; const char mo4b[] PROGMEM = "River deposition";
const char mo4c[] PROGMEM = "Volcanic"; const char mo4d[] PROGMEM = "Wind";
const char me4[]  PROGMEM = "Heavy rain leaches silica/lime, leaving acidic laterite.";

const char mq5[]  PROGMEM = "LIMCAP represents minerals of which soil?";
const char mo5a[] PROGMEM = "Alluvial"; const char mo5b[] PROGMEM = "Black";
const char mo5c[] PROGMEM = "Red";      const char mo5d[] PROGMEM = "Laterite";
const char me5[]  PROGMEM = "Lime, Iron, Mg, Ca, Alumina, Potash = Black soil.";

const char mq6[]  PROGMEM = "Deep ravines/badlands in Chambal are caused by:";
const char mo6a[] PROGMEM = "Sheet erosion"; const char mo6b[] PROGMEM = "Rill erosion";
const char mo6c[] PROGMEM = "Gully erosion"; const char mo6d[] PROGMEM = "Wind erosion";
const char me6[]  PROGMEM = "Gully erosion cuts deep channels into unprotected soil.";

const char mq7[]  PROGMEM = "Best soil conservation for steep hilly slopes:";
const char mo7a[] PROGMEM = "Shelter belts"; const char mo7b[] PROGMEM = "Terrace farming";
const char mo7c[] PROGMEM = "Strip cropping"; const char mo7d[] PROGMEM = "Monoculture";
const char me7[]  PROGMEM = "Terraces slow water runoff on steep slopes.";

const char mq8[]  PROGMEM = "Most widespread soil in India:";
const char mo8a[] PROGMEM = "Black"; const char mo8b[] PROGMEM = "Red";
const char mo8c[] PROGMEM = "Laterite"; const char mo8d[] PROGMEM = "Alluvial";
const char me8[]  PROGMEM = "Alluvial covers 40% of India including Indo-Gangetic plain.";

const char mq9[]  PROGMEM = "Most critical organic element for soil fertility:";
const char mo9a[] PROGMEM = "Silt"; const char mo9b[] PROGMEM = "Kankar";
const char mo9c[] PROGMEM = "Humus"; const char mo9d[] PROGMEM = "Clay";
const char me9[]  PROGMEM = "Humus from decayed organic matter determines fertility.";

const char mq10[]  PROGMEM = "Wind erosion check in Rajasthan uses:";
const char mo10a[] PROGMEM = "Shelter belts"; const char mo10b[] PROGMEM = "Gully plugging";
const char mo10c[] PROGMEM = "Terrace farming"; const char mo10d[] PROGMEM = "Contour ploughing";
const char me10[]  PROGMEM = "Rows of trees break wind force, protecting sandy soil.";

const char mq11[]  PROGMEM = "Red soil has poor water retention because:";
const char mo11a[] PROGMEM = "Highly clayey"; const char mo11b[] PROGMEM = "Rich in humus";
const char mo11c[] PROGMEM = "Porous, coarse-grained"; const char mo11d[] PROGMEM = "Compacted by rain";
const char me11[]  PROGMEM = "Coarse, porous texture makes water drain away rapidly.";

const char mq12[]  PROGMEM = "First President of Indian National Congress (1885):";
const char mo12a[] PROGMEM = "W.C. Bonnerjee"; const char mo12b[] PROGMEM = "D. Naoroji";
const char mo12c[] PROGMEM = "A.O. Hume"; const char mo12d[] PROGMEM = "S. Banerjee";
const char me12[]  PROGMEM = "W.C. Bonnerjee presided over the first INC session in Bombay.";

// --- History MCQs (Q13–Q24) ---
const char mq13[]  PROGMEM = "Who founded Satya Shodhak Samaj in 1873?";
const char mo13a[] PROGMEM = "R.R.M. Roy"; const char mo13b[] PROGMEM = "Jyotiba Phule";
const char mo13c[] PROGMEM = "S. Dayanand"; const char mo13d[] PROGMEM = "D. Naoroji";
const char me13[]  PROGMEM = "Phule founded it to resist caste oppression in Maharashtra.";

const char mq14[]  PROGMEM = "Vernacular Press Act 1878 passed by which Viceroy?";
const char mo14a[] PROGMEM = "Lord Ripon"; const char mo14b[] PROGMEM = "Lord Lytton";
const char mo14c[] PROGMEM = "Lord Curzon"; const char mo14d[] PROGMEM = "Lord Dufferin";
const char me14[]  PROGMEM = "Lord Lytton (1876-80) suppressed anti-British local press.";

const char mq15[]  PROGMEM = "Ilbert Bill Controversy 1883 under which Viceroy?";
const char mo15a[] PROGMEM = "Lord Lytton"; const char mo15b[] PROGMEM = "Lord Ripon";
const char mo15c[] PROGMEM = "Lord Curzon"; const char mo15d[] PROGMEM = "Lord Dalhousie";
const char me15[]  PROGMEM = "Lord Ripon's Ilbert Bill allowed Indian judges to try Europeans.";

const char mq16[]  PROGMEM = "Author of 'Poverty and Un-British Rule in India':";
const char mo16a[] PROGMEM = "S. Banerjee"; const char mo16b[] PROGMEM = "D. Naoroji";
const char mo16c[] PROGMEM = "W.C. Bonnerjee"; const char mo16d[] PROGMEM = "A.O. Hume";
const char me16[]  PROGMEM = "Naoroji's Drain of Wealth theory exposed British exploitation.";

const char mq17[]  PROGMEM = "Indian National Congress was founded in:";
const char mo17a[] PROGMEM = "1857"; const char mo17b[] PROGMEM = "1876";
const char mo17c[] PROGMEM = "1885"; const char mo17d[] PROGMEM = "1905";
const char me17[]  PROGMEM = "INC founded December 1885 by A.O. Hume and Indian leaders.";

const char mq18[]  PROGMEM = "Father of Indian Renaissance who founded Brahmo Samaj:";
const char mo18a[] PROGMEM = "R.R.M. Roy"; const char mo18b[] PROGMEM = "J. Phule";
const char mo18c[] PROGMEM = "S. Vivekananda"; const char mo18d[] PROGMEM = "D. Tagore";
const char me18[]  PROGMEM = "Raja Ram Mohan Roy founded Brahmo Samaj in 1828.";

const char mq19[]  PROGMEM = "Amrit Bazar Patrika escaped Vernacular Press Act by:";
const char mo19a[] PROGMEM = "Closing down"; const char mo19b[] PROGMEM = "Filing petition";
const char mo19c[] PROGMEM = "Switching to English overnight"; const char mo19d[] PROGMEM = "Moving to Bombay";
const char me19[]  PROGMEM = "They converted to English overnight to bypass the 1878 Act.";

const char mq20[]  PROGMEM = "East India Association in London founded in 1866 by:";
const char mo20a[] PROGMEM = "W.C. Bonnerjee"; const char mo20b[] PROGMEM = "S. Banerjee";
const char mo20c[] PROGMEM = "D. Naoroji"; const char mo20d[] PROGMEM = "A.O. Hume";
const char me20[]  PROGMEM = "Naoroji raised Indian grievances among British parliamentarians.";

const char mq21[]  PROGMEM = "Jyotiba Phule's book exposing caste oppression:";
const char mo21a[] PROGMEM = "Satyarth Prakash"; const char mo21b[] PROGMEM = "Ghulamgiri";
const char mo21c[] PROGMEM = "Discovery of India"; const char mo21d[] PROGMEM = "Poverty & Un-British Rule";
const char me21[]  PROGMEM = "Ghulamgiri (1873) compared caste discrimination to US slavery.";

const char mq22[]  PROGMEM = "First INC session held in December 1885 at:";
const char mo22a[] PROGMEM = "Calcutta"; const char mo22b[] PROGMEM = "Madras";
const char mo22c[] PROGMEM = "Bombay"; const char mo22d[] PROGMEM = "Delhi";
const char me22[]  PROGMEM = "First session at Gokuldas Tejpal College, Bombay, 72 delegates.";

const char mq23[]  PROGMEM = "Lytton reduced ICS exam max age to prevent Indians from:";
const char mo23a[] PROGMEM = "Learning English"; const char mo23b[] PROGMEM = "Competing successfully";
const char mo23c[] PROGMEM = "Joining army"; const char mo23d[] PROGMEM = "Voting";
const char me23[]  PROGMEM = "Reducing age from 21 to 19 prevented Indians from travelling to London.";

const char mq24[]  PROGMEM = "Mendel's plant for genetics experiments:";
const char mo24a[] PROGMEM = "Sweet pea"; const char mo24b[] PROGMEM = "Garden pea (Pisum sativum)";
const char mo24c[] PROGMEM = "Wild pea"; const char mo24d[] PROGMEM = "Pigeon pea";
const char me24[]  PROGMEM = "Pisum sativum: clear traits, short cycle, self-pollinating.";

// --- Genetics MCQs (Q25–Q36) ---
const char mq25[]  PROGMEM = "F2 phenotypic ratio in monohybrid cross:";
const char mo25a[] PROGMEM = "1:2:1"; const char mo25b[] PROGMEM = "3:1";
const char mo25c[] PROGMEM = "9:3:3:1"; const char mo25d[] PROGMEM = "1:1:1:1";
const char me25[]  PROGMEM = "Tt x Tt gives 3 dominant : 1 recessive in F2.";

const char mq26[]  PROGMEM = "F2 phenotypic ratio in dihybrid cross:";
const char mo26a[] PROGMEM = "3:1"; const char mo26b[] PROGMEM = "1:2:1";
const char mo26c[] PROGMEM = "9:3:3:1"; const char mo26d[] PROGMEM = "1:1";
const char me26[]  PROGMEM = "RrYy x RrYy yields 9:3:3:1 showing independent assortment.";

const char mq27[]  PROGMEM = "Alternative gene forms on homologous chromosomes are:";
const char mo27a[] PROGMEM = "Chromatid"; const char mo27b[] PROGMEM = "Autosome";
const char mo27c[] PROGMEM = "Allele"; const char mo27d[] PROGMEM = "Phenotype";
const char me27[]  PROGMEM = "Alleles (e.g. T and t) determine different expressions of a trait.";

const char mq28[]  PROGMEM = "In humans, sex of offspring is determined by:";
const char mo28a[] PROGMEM = "Mother"; const char mo28b[] PROGMEM = "Father";
const char mo28c[] PROGMEM = "Both equally"; const char mo28d[] PROGMEM = "Temperature";
const char me28[]  PROGMEM = "Father produces X and Y sperm; mother only produces X eggs.";

const char mq29[]  PROGMEM = "Haemophilia is inherited as:";
const char mo29a[] PROGMEM = "Autosomal dominant"; const char mo29b[] PROGMEM = "Autosomal recessive";
const char mo29c[] PROGMEM = "X-linked recessive"; const char mo29d[] PROGMEM = "Y-linked dominant";
const char me29[]  PROGMEM = "Defective allele on X chromosome; more common in males.";

const char mq30[]  PROGMEM = "Carrier female genotype for haemophilia:";
const char mo30a[] PROGMEM = "XX"; const char mo30b[] PROGMEM = "X(h)X";
const char mo30c[] PROGMEM = "X(h)X(h)"; const char mo30d[] PROGMEM = "XY";
const char me30[]  PROGMEM = "One normal X + one defective X: carrier but unaffected.";

const char mq31[]  PROGMEM = "Y-linked (holandric) trait example:";
const char mo31a[] PROGMEM = "Colour blindness"; const char mo31b[] PROGMEM = "Haemophilia";
const char mo31c[] PROGMEM = "Hypertrichosis (ear hair)"; const char mo31d[] PROGMEM = "Albinism";
const char me31[]  PROGMEM = "Y-linked traits pass exclusively from father to son.";

const char mq32[]  PROGMEM = "Physical observable appearance of an organism is its:";
const char mo32a[] PROGMEM = "Genotype"; const char mo32b[] PROGMEM = "Phenotype";
const char mo32c[] PROGMEM = "Heterozygous"; const char mo32d[] PROGMEM = "Homozygous";
const char me32[]  PROGMEM = "Phenotype is outward appearance; genotype is genetic makeup.";

const char mq33[]  PROGMEM = "Normal human somatic cell has how many autosome pairs?";
const char mo33a[] PROGMEM = "23 pairs"; const char mo33b[] PROGMEM = "22 pairs";
const char mo33c[] PROGMEM = "1 pair"; const char mo33d[] PROGMEM = "44 pairs";
const char me33[]  PROGMEM = "23 pairs total: 22 autosomes + 1 sex chromosome pair.";

const char mq34[]  PROGMEM = "Sudden permanent inheritable change in gene/chromosome:";
const char mo34a[] PROGMEM = "Heredity"; const char mo34b[] PROGMEM = "Segregation";
const char mo34c[] PROGMEM = "Mutation"; const char mo34d[] PROGMEM = "Assortment";
const char me34[]  PROGMEM = "Mutations alter DNA sequence; can cause disorders like SCA.";

// --- Chemistry MCQs (Q35–Q48) ---
const char mq35[]  PROGMEM = "Why is H3PO3 a dibasic acid?";
const char mo35a[] PROGMEM = "Only one H atom"; const char mo35b[] PROGMEM = "Only 2 H bonded to O are ionizable";
const char mo35c[] PROGMEM = "Unstable in liquid"; const char mo35d[] PROGMEM = "Lacks phosphorus";
const char me35[]  PROGMEM = "Third H is bonded to P directly and cannot dissociate.";

const char mq36[]  PROGMEM = "Acetic acid (CH3COOH) is monobasic because:";
const char mo36a[] PROGMEM = "Only -COOH hydrogen is replaceable"; const char mo36b[] PROGMEM = "Doesn't dissolve in water";
const char mo36c[] PROGMEM = "Forms only acid salts"; const char mo36d[] PROGMEM = "It's a strong acid";
const char me36[]  PROGMEM = "Only the -COOH group hydrogen ionizes; other 3 H are on carbon.";

const char mq37[]  PROGMEM = "PbO2 is not a true base because:";
const char mo37a[] PROGMEM = "Doesn't react with acids";
const char mo37b[] PROGMEM = "Produces salt+water+Cl2 not just salt+water";
const char mo37c[] PROGMEM = "Turns litmus red"; const char mo37d[] PROGMEM = "Soluble in water";
const char me37[]  PROGMEM = "True base gives salt+water ONLY. PbO2 gives extra Cl2 gas.";

const char mq38[]  PROGMEM = "Methyl orange in acidic solution turns:";
const char mo38a[] PROGMEM = "Yellow"; const char mo38b[] PROGMEM = "Orange";
const char mo38c[] PROGMEM = "Red"; const char mo38d[] PROGMEM = "Pink";
const char me38[]  PROGMEM = "Methyl orange: red in acid, orange neutral, yellow in base.";

const char mq39[]  PROGMEM = "FeCl3 is prepared ONLY by:";
const char mo39a[] PROGMEM = "NaCl method"; const char mo39b[] PROGMEM = "FeCl2 method";
const char mo39c[] PROGMEM = "Direct combination (Fe + Cl2)"; const char mo39d[] PROGMEM = "CuSO4 method";
const char me39[]  PROGMEM = "Anhydrous FeCl3: heated iron + dry chlorine gas only.";

const char mq40[]  PROGMEM = "Na2CO3.10H2O left in dry air undergoes:";
const char mo40a[] PROGMEM = "Deliquescence"; const char mo40b[] PROGMEM = "Efflorescence";
const char mo40c[] PROGMEM = "Sublimation"; const char mo40d[] PROGMEM = "Hydration";
const char me40[]  PROGMEM = "Efflorescent: loses water of crystallization to dry atmosphere.";

const char mq41[]  PROGMEM = "NaOH absorbs moisture and dissolves: this is:";
const char mo41a[] PROGMEM = "Hygroscopic"; const char mo41b[] PROGMEM = "Efflorescence";
const char mo41c[] PROGMEM = "Deliquescence"; const char mo41d[] PROGMEM = "Sublimation";
const char me41[]  PROGMEM = "Deliquescent: absorbs moisture AND dissolves in it.";

const char mq42[]  PROGMEM = "Concentrated H2SO4 is used as drying agent because it is:";
const char mo42a[] PROGMEM = "Deliquescent"; const char mo42b[] PROGMEM = "Efflorescent";
const char mo42c[] PROGMEM = "Hygroscopic"; const char mo42d[] PROGMEM = "Reactive";
const char me42[]  PROGMEM = "Hygroscopic: absorbs moisture without dissolving or changing state.";

const char mq43[]  PROGMEM = "Salt giving CO2 that turns lime water milky but not K2Cr2O7:";
const char mo43a[] PROGMEM = "Sulphite"; const char mo43b[] PROGMEM = "Carbonate";
const char mo43c[] PROGMEM = "Chloride"; const char mo43d[] PROGMEM = "Sulphate";
const char me43[]  PROGMEM = "CO2 from carbonates turns limewater milky; doesn't change dichromate.";

const char mq44[]  PROGMEM = "pH of strongly basic solution at 25C:";
const char mo44a[] PROGMEM = "2"; const char mo44b[] PROGMEM = "7";
const char mo44c[] PROGMEM = "5"; const char mo44d[] PROGMEM = "13";
const char me44[]  PROGMEM = "pH scale 0-14; 7=neutral; 13-14 = strongly basic.";

const char mq45[]  PROGMEM = "NO2 is a mixed acid anhydride because it reacts with water to give:";
const char mo45a[] PROGMEM = "HNO3 only"; const char mo45b[] PROGMEM = "HNO2 only";
const char mo45c[] PROGMEM = "HNO2 and HNO3"; const char mo45d[] PROGMEM = "H2O and N2";
const char me45[]  PROGMEM = "Mixed anhydride: one oxide produces TWO different acids.";

const char mq46[]  PROGMEM = "NaCl is typically prepared in lab by:";
const char mo46a[] PROGMEM = "Lead sulfate method"; const char mo46b[] PROGMEM = "Titration/neutralization";
const char mo46c[] PROGMEM = "FeCl3 method"; const char mo46d[] PROGMEM = "Precipitation";
const char me46[]  PROGMEM = "NaOH + HCl titration gives soluble NaCl salt.";

// --- Neurology MCQs (Q47–Q99) ---
const char mq47[]  PROGMEM = "Structural and functional unit of the nervous system:";
const char mo47a[] PROGMEM = "Neuron"; const char mo47b[] PROGMEM = "Nephron";
const char mo47c[] PROGMEM = "Cyton"; const char mo47d[] PROGMEM = "Axon";
const char me47[]  PROGMEM = "Neuron: receives, conducts, and transmits impulses.";

const char mq48[]  PROGMEM = "Cytoplasmic matrix of neuron cell body is called:";
const char mo48a[] PROGMEM = "Axoplasm"; const char mo48b[] PROGMEM = "Neuroplasm";
const char mo48c[] PROGMEM = "Sarcoplasm"; const char mo48d[] PROGMEM = "Nucleoplasm";
const char me48[]  PROGMEM = "Neuroplasm is cytoplasm within the cyton of a nerve cell.";

const char mq49[]  PROGMEM = "Small ribosomes in neuron cell body are called:";
const char mo49a[] PROGMEM = "Mitochondria"; const char mo49b[] PROGMEM = "Centrosomes";
const char mo49c[] PROGMEM = "Nissl granules"; const char mo49d[] PROGMEM = "Neurofibrils";
const char me49[]  PROGMEM = "Nissl granules are ribosome-rich structures for protein synthesis.";

const char mq50[]  PROGMEM = "Primary function of dendrites:";
const char mo50a[] PROGMEM = "Carry impulses away from body"; const char mo50b[] PROGMEM = "Receive impulses toward body";
const char mo50c[] PROGMEM = "Insulate the axon"; const char mo50d[] PROGMEM = "Release neurotransmitters";
const char me50[]  PROGMEM = "Dendrites are short, branched receivers; conduct to cyton.";

const char mq51[]  PROGMEM = "Fatty insulating layer surrounding the axon:";
const char mo51a[] PROGMEM = "Axolemma"; const char mo51b[] PROGMEM = "Neurolemma";
const char mo51c[] PROGMEM = "Myelin sheath"; const char mo51d[] PROGMEM = "Meninges";
const char me51[]  PROGMEM = "Myelin sheath insulates axon and speeds impulse transmission.";

const char mq52[]  PROGMEM = "Gaps in myelin sheath between Schwann cells are:";
const char mo52a[] PROGMEM = "Synaptic clefts"; const char mo52b[] PROGMEM = "Dendritic nodes";
const char mo52c[] PROGMEM = "Nodes of Ranvier"; const char mo52d[] PROGMEM = "Axonal gaps";
const char me52[]  PROGMEM = "Nodes of Ranvier allow saltatory (jumping) conduction.";

const char mq53[]  PROGMEM = "Outer protective membrane of Schwann cells:";
const char mo53a[] PROGMEM = "Axolemma"; const char mo53b[] PROGMEM = "Neurolemma";
const char mo53c[] PROGMEM = "Pia mater"; const char mo53d[] PROGMEM = "Duramater";
const char me53[]  PROGMEM = "Neurolemma is the outermost nucleated Schwann cell sheath.";

const char mq54[]  PROGMEM = "Cluster of nerve cell bodies outside CNS:";
const char mo54a[] PROGMEM = "Ganglion"; const char mo54b[] PROGMEM = "Nerve";
const char mo54c[] PROGMEM = "Plexus"; const char mo54d[] PROGMEM = "Synapse";
const char me54[]  PROGMEM = "Ganglion: anatomically distinct cluster in peripheral NS.";

const char mq55[]  PROGMEM = "Junction between axon terminal and dendrite of next neuron:";
const char mo55a[] PROGMEM = "Ganglion"; const char mo55b[] PROGMEM = "Synapse";
const char mo55c[] PROGMEM = "Node of Ranvier"; const char mo55d[] PROGMEM = "Neuromuscular junc.";
const char me55[]  PROGMEM = "Synapse: gap bridged by chemical neurotransmitters.";

const char mq56[]  PROGMEM = "Neurotransmitter of parasympathetic nerve fiber:";
const char mo56a[] PROGMEM = "Noradrenaline"; const char mo56b[] PROGMEM = "Adrenaline";
const char mo56c[] PROGMEM = "Acetylcholine"; const char mo56d[] PROGMEM = "Dopamine";
const char me56[]  PROGMEM = "Acetylcholine: primary parasympathetic neurotransmitter.";

const char mq57[]  PROGMEM = "Sudden automatic involuntary response to stimulus:";
const char mo57a[] PROGMEM = "Voluntary action"; const char mo57b[] PROGMEM = "Reflex action";
const char mo57c[] PROGMEM = "Conditioned habit"; const char mo57d[] PROGMEM = "Cerebral response";
const char me57[]  PROGMEM = "Reflex actions protect body without conscious thought.";

const char mq58[]  PROGMEM = "Correct reflex arc pathway:";
const char mo58a[] PROGMEM = "Stimulus>Receptor>Sensory>Cord>Motor>Effector";
const char mo58b[] PROGMEM = "Stimulus>Receptor>Motor>Cord>Sensory>Effector";
const char mo58c[] PROGMEM = "Stimulus>Effector>Sensory>Cord>Motor>Receptor";
const char mo58d[] PROGMEM = "Stimulus>Receptor>Sensory>Brain>Motor>Effector";
const char me58[]  PROGMEM = "Standard arc: Receptor->Sensory->Cord->Motor->Effector.";

const char mq59[]  PROGMEM = "Three protective membranes around brain and spinal cord:";
const char mo59a[] PROGMEM = "Pleura"; const char mo59b[] PROGMEM = "Meninges";
const char mo59c[] PROGMEM = "Pericardium"; const char mo59d[] PROGMEM = "Peritoneum";
const char me59[]  PROGMEM = "Meninges: Dura mater, Arachnoid, Pia mater.";

const char mq60[]  PROGMEM = "Correct order of meninges outer to inner:";
const char mo60a[] PROGMEM = "Arachnoid>Dura>Pia"; const char mo60b[] PROGMEM = "Dura>Pia>Arachnoid";
const char mo60c[] PROGMEM = "Dura>Arachnoid>Pia"; const char mo60d[] PROGMEM = "Pia>Arachnoid>Dura";
const char me60[]  PROGMEM = "Mnemonic 'DAP': Dura mater, Arachnoid, Pia mater.";

const char mq61[]  PROGMEM = "CSF (Cerebrospinal Fluid) function:";
const char mo61a[] PROGMEM = "Shock absorber for brain"; const char mo61b[] PROGMEM = "Conducts impulses";
const char mo61c[] PROGMEM = "Secretes insulin"; const char mo61d[] PROGMEM = "Makes red blood cells";
const char me61[]  PROGMEM = "CSF cushions brain, exchanges nutrients, surrounds CNS.";

const char mq62[]  PROGMEM = "Ridges/elevations on cerebral cortex surface:";
const char mo62a[] PROGMEM = "Sulci"; const char mo62b[] PROGMEM = "Gyri";
const char mo62c[] PROGMEM = "Fissures"; const char mo62d[] PROGMEM = "Ventricles";
const char me62[]  PROGMEM = "Gyri are elevated ridges; increase surface area for neurons.";

const char mq63[]  PROGMEM = "Sheet connecting left and right cerebral hemispheres:";
const char mo63a[] PROGMEM = "Corpus callosum"; const char mo63b[] PROGMEM = "Pons";
const char mo63c[] PROGMEM = "Medulla oblongata"; const char mo63d[] PROGMEM = "Thalamus";
const char me63[]  PROGMEM = "Corpus callosum: thick band of 200M+ myelinated fibers.";

const char mq64[]  PROGMEM = "Seat of intelligence, memory, willpower, reasoning:";
const char mo64a[] PROGMEM = "Cerebellum"; const char mo64b[] PROGMEM = "Medulla oblongata";
const char mo64c[] PROGMEM = "Cerebrum"; const char mo64d[] PROGMEM = "Pons";
const char me64[]  PROGMEM = "Cerebrum: largest brain division, center of consciousness.";

const char mq65[]  PROGMEM = "Person can't maintain balance; which part affected?";
const char mo65a[] PROGMEM = "Cerebrum"; const char mo65b[] PROGMEM = "Thalamus";
const char mo65c[] PROGMEM = "Cerebellum"; const char mo65d[] PROGMEM = "Medulla oblongata";
const char me65[]  PROGMEM = "Cerebellum: coordinates voluntary movements and balance.";

const char mq66[]  PROGMEM = "Controls heartbeat, breathing, blood pressure:";
const char mo66a[] PROGMEM = "Cerebrum"; const char mo66b[] PROGMEM = "Cerebellum";
const char mo66c[] PROGMEM = "Pons"; const char mo66d[] PROGMEM = "Medulla oblongata";
const char me66[]  PROGMEM = "Medulla: vital reflex center for cardiac/respiratory functions.";

const char mq67[]  PROGMEM = "Destruction of medulla oblongata causes:";
const char mo67a[] PROGMEM = "Immediate death"; const char mo67b[] PROGMEM = "Memory loss";
const char mo67c[] PROGMEM = "Balance loss"; const char mo67d[] PROGMEM = "Blindness";
const char me67[]  PROGMEM = "Medulla houses cardiac and respiratory centers; damage = death.";

const char mq68[]  PROGMEM = "Primary control for hunger, thirst, body temperature:";
const char mo68a[] PROGMEM = "Thalamus"; const char mo68b[] PROGMEM = "Hypothalamus";
const char mo68c[] PROGMEM = "Cerebellum"; const char mo68d[] PROGMEM = "Pons";
const char me68[]  PROGMEM = "Hypothalamus: master of homeostasis and endocrine linkage.";

const char mq69[]  PROGMEM = "Grey matter location in BRAIN:";
const char mo69a[] PROGMEM = "Outside (cortex)"; const char mo69b[] PROGMEM = "Inside";
const char mo69c[] PROGMEM = "Mixed randomly"; const char mo69d[] PROGMEM = "Absent";
const char me69[]  PROGMEM = "Brain: grey matter outside (cortex), white matter inside.";

const char mq70[]  PROGMEM = "Grey matter location in SPINAL CORD:";
const char mo70a[] PROGMEM = "Outside"; const char mo70b[] PROGMEM = "Inside (H-shaped)";
const char mo70c[] PROGMEM = "Mixed randomly"; const char mo70d[] PROGMEM = "Only white matter";
const char me70[]  PROGMEM = "Spinal cord: inner H-shaped grey matter, outer white matter.";

const char mq71[]  PROGMEM = "Human PNS: cranial and spinal nerve pairs count:";
const char mo71a[] PROGMEM = "10 and 30"; const char mo71b[] PROGMEM = "12 and 31";
const char mo71c[] PROGMEM = "31 and 12"; const char mo71d[] PROGMEM = "12 and 12";
const char me71[]  PROGMEM = "12 cranial + 31 spinal pairs in the human PNS.";

const char mq72[]  PROGMEM = "Emergency 'fight-or-flight' division:";
const char mo72a[] PROGMEM = "Sympathetic NS"; const char mo72b[] PROGMEM = "Parasympathetic NS";
const char mo72c[] PROGMEM = "Somatic NS"; const char mo72d[] PROGMEM = "Central NS";
const char me72[]  PROGMEM = "Sympathetic: accelerates heart, dilates pupils in emergency.";

const char mq73[]  PROGMEM = "'Rest and digest' division of ANS:";
const char mo73a[] PROGMEM = "Sympathetic NS"; const char mo73b[] PROGMEM = "Parasympathetic NS";
const char mo73c[] PROGMEM = "Somatic NS"; const char mo73d[] PROGMEM = "Sensory NS";
const char me73[]  PROGMEM = "Parasympathetic: conserves energy, slows heart, aids digestion.";

const char mq74[]  PROGMEM = "Pavlov's dog salivating at bell sound is:";
const char mo74a[] PROGMEM = "Natural reflex"; const char mo74b[] PROGMEM = "Conditioned reflex";
const char mo74c[] PROGMEM = "Voluntary movement"; const char mo74d[] PROGMEM = "Spinal reflex";
const char me74[]  PROGMEM = "Conditioned: learned through association, not inborn.";

const char mq75[]  PROGMEM = "Example of natural (inborn) reflex:";
const char mo75a[] PROGMEM = "Applying car brakes"; const char mo75b[] PROGMEM = "Blinking when dust enters";
const char mo75c[] PROGMEM = "Typing on keyboard"; const char mo75d[] PROGMEM = "Playing instrument";
const char me75[]  PROGMEM = "Blinking is inborn protective reflex requiring no learning.";

const char mq76[]  PROGMEM = "Sensory neuron is also called:";
const char mo76a[] PROGMEM = "Efferent neuron"; const char mo76b[] PROGMEM = "Afferent neuron";
const char mo76c[] PROGMEM = "Association neuron"; const char mo76d[] PROGMEM = "Motor neuron";
const char me76[]  PROGMEM = "Afferent: carries impulses TOWARDS the CNS.";

const char mq77[]  PROGMEM = "Motor neuron is also called:";
const char mo77a[] PROGMEM = "Afferent neuron"; const char mo77b[] PROGMEM = "Efferent neuron";
const char mo77c[] PROGMEM = "Relay neuron"; const char mo77d[] PROGMEM = "Interneuron";
const char me77[]  PROGMEM = "Efferent: carries impulses AWAY from CNS to effectors.";

const char mq78[]  PROGMEM = "Neurotransmitter of sympathetic nerve fibers:";
const char mo78a[] PROGMEM = "Acetylcholine"; const char mo78b[] PROGMEM = "Noradrenaline";
const char mo78c[] PROGMEM = "Serotonin"; const char mo78d[] PROGMEM = "GABA";
const char me78[]  PROGMEM = "Noradrenaline: primary sympathetic postganglionic transmitter.";

const char mq79[]  PROGMEM = "Alcohol impairs balance: which brain region?";
const char mo79a[] PROGMEM = "Cerebrum"; const char mo79b[] PROGMEM = "Pons";
const char mo79c[] PROGMEM = "Cerebellum"; const char mo79d[] PROGMEM = "Medulla oblongata";
const char me79[]  PROGMEM = "Alcohol depresses cerebellum: causes ataxia and unsteady gait.";

const char mq80[]  PROGMEM = "Neuron part that LACKS Nissl granules:";
const char mo80a[] PROGMEM = "Cyton"; const char mo80b[] PROGMEM = "Axon";
const char mo80c[] PROGMEM = "Dendrite"; const char mo80d[] PROGMEM = "Cell body";
const char me80[]  PROGMEM = "Nissl granules present in cyton and dendrites; absent in axon.";

const char mq81[]  PROGMEM = "Forebrain structures:";
const char mo81a[] PROGMEM = "Cerebrum + Diencephalon"; const char mo81b[] PROGMEM = "Cerebellum + Pons";
const char mo81c[] PROGMEM = "Medulla + Pons"; const char mo81d[] PROGMEM = "Cerebrum + Cerebellum";
const char me81[]  PROGMEM = "Forebrain: Cerebrum, olfactory lobes, Diencephalon.";

const char mq82[]  PROGMEM = "Hindbrain structures:";
const char mo82a[] PROGMEM = "Cerebrum+Pons+Medulla"; const char mo82b[] PROGMEM = "Cerebellum+Pons+Medulla";
const char mo82c[] PROGMEM = "Thalamus+Hypothalamus+Pons"; const char mo82d[] PROGMEM = "Optic lobes+Cerebrum";
const char me82[]  PROGMEM = "Hindbrain: Cerebellum, Pons, and Medulla oblongata.";

const char mq83[]  PROGMEM = "Nerve impulse ENTERS spinal cord through:";
const char mo83a[] PROGMEM = "Ventral root"; const char mo83b[] PROGMEM = "Dorsal root";
const char mo83c[] PROGMEM = "White column"; const char mo83d[] PROGMEM = "Central canal";
const char me83[]  PROGMEM = "Sensory fibers enter via dorsal (posterior) root.";

const char mq84[]  PROGMEM = "Motor nerve fibers EMERGE from spinal cord through:";
const char mo84a[] PROGMEM = "Dorsal root"; const char mo84b[] PROGMEM = "Ventral root";
const char mo84c[] PROGMEM = "Meninges"; const char mo84d[] PROGMEM = "Grey commissure";
const char me84[]  PROGMEM = "Motor fibers exit via ventral (anterior) root to effectors.";

const char mq85[]  PROGMEM = "Association (relay) neuron links:";
const char mo85a[] PROGMEM = "Relay neuron"; const char mo85b[] PROGMEM = "Afferent neuron";
const char mo85c[] PROGMEM = "Efferent neuron"; const char mo85d[] PROGMEM = "Receptor";
const char me85[]  PROGMEM = "Interneuron entirely within CNS: connects sensory to motor.";

const char mq86[]  PROGMEM = "Synapse transmission is strictly ONE-WAY because:";
const char mo86a[] PROGMEM = "Myelin blocks backward flow";
const char mo86b[] PROGMEM = "Receptors only on postsynaptic membrane";
const char mo86c[] PROGMEM = "Axons are longer than dendrites";
const char mo86d[] PROGMEM = "CSF forces one direction";
const char me86[]  PROGMEM = "Vesicles only in presynaptic terminal; receptors on postsynaptic side.";

const char mq87[]  PROGMEM = "Voluntary action example:";
const char mo87a[] PROGMEM = "Sneezing from pepper"; const char mo87b[] PROGMEM = "Knee-jerk from tap";
const char mo87c[] PROGMEM = "Writing exam answer"; const char mo87d[] PROGMEM = "Peristalsis";
const char me87[]  PROGMEM = "Writing is consciously decided by the cerebral cortex.";

const char mq88[]  PROGMEM = "Prevents friction, cushions brain inside skull:";
const char mo88a[] PROGMEM = "Neuroplasm"; const char mo88b[] PROGMEM = "Cerebrospinal Fluid";
const char mo88c[] PROGMEM = "Myelin"; const char mo88d[] PROGMEM = "Corpus callosum";
const char me88[]  PROGMEM = "CSF: liquid cushion between brain tissue and hard skull.";

const char mq89[]  PROGMEM = "Approximate length of adult human spinal cord:";
const char mo89a[] PROGMEM = "43-45 cm"; const char mo89b[] PROGMEM = "10-12 cm";
const char mo89c[] PROGMEM = "70-75 cm"; const char mo89d[] PROGMEM = "150 cm";
const char me89[]  PROGMEM = "Spinal cord: approx 43-45 cm from medulla down.";

const char mq90[]  PROGMEM = "Bony box protecting the human brain:";
const char mo90a[] PROGMEM = "Vertebral column"; const char mo90b[] PROGMEM = "Ribcage";
const char mo90c[] PROGMEM = "Cranium"; const char mo90d[] PROGMEM = "Sternum";
const char me90[]  PROGMEM = "Cranium (brainbox): 8 bones forming rigid skull vault.";

const char mq91[]  PROGMEM = "Depressions/grooves on cerebral cortex:";
const char mo91a[] PROGMEM = "Gyri"; const char mo91b[] PROGMEM = "Sulci";
const char mo91c[] PROGMEM = "Ganglia"; const char mo91d[] PROGMEM = "Vesicles";
const char me91[]  PROGMEM = "Sulci are shallow grooves separating gyri on cortex.";

const char mq92[]  PROGMEM = "Bridge connecting two lobes of cerebellum:";
const char mo92a[] PROGMEM = "Corpus callosum"; const char mo92b[] PROGMEM = "Pons";
const char mo92c[] PROGMEM = "Thalamus"; const char mo92d[] PROGMEM = "Hypothalamus";
const char me92[]  PROGMEM = "Pons: carries fiber tracts connecting different brain regions.";

const char mq93[]  PROGMEM = "Cerebellar damage result:";
const char mo93a[] PROGMEM = "Paralysis"; const char mo93b[] PROGMEM = "Loss of coordination/balance";
const char mo93c[] PROGMEM = "Memory loss"; const char mo93d[] PROGMEM = "Blindness";
const char me93[]  PROGMEM = "Muscles can move but coordination, timing and smoothness lost.";

const char mq94[]  PROGMEM = "Hypothalamus connects nervous system to:";
const char mo94a[] PROGMEM = "Circulatory system"; const char mo94b[] PROGMEM = "Endocrine system";
const char mo94c[] PROGMEM = "Digestive system"; const char mo94d[] PROGMEM = "Lymphatic system";
const char me94[]  PROGMEM = "Hypothalamus controls pituitary gland, linking NS to hormones.";

const char mq95[]  PROGMEM = "Micturition refers to:";
const char mo95a[] PROGMEM = "Defecation"; const char mo95b[] PROGMEM = "Urinating";
const char mo95c[] PROGMEM = "Deamination"; const char mo95d[] PROGMEM = "Diapedesis";
const char me95[]  PROGMEM = "Micturition: process of urinating, coordinated by spinal reflexes.";

const char mq96[]  PROGMEM = "Autonomic system uses how many neurons to reach effectors?";
const char mo96a[] PROGMEM = "One"; const char mo96b[] PROGMEM = "Two";
const char mo96c[] PROGMEM = "Three"; const char mo96d[] PROGMEM = "Four";
const char me96[]  PROGMEM = "Pre and postganglionic neurons meet at an autonomic ganglion.";

const char mq97[]  PROGMEM = "Dorsal root damage causes:";
const char mo97a[] PROGMEM = "Paralysis only"; const char mo97b[] PROGMEM = "Loss of sensation only";
const char mo97c[] PROGMEM = "Both loss"; const char mo97d[] PROGMEM = "No effect";
const char me97[]  PROGMEM = "Dorsal = sensory only; damage = loss of sensation, not paralysis.";

const char mq98[]  PROGMEM = "Ventral root damage causes:";
const char mo98a[] PROGMEM = "Loss of sensation"; const char mo98b[] PROGMEM = "Paralysis only";
const char mo98c[] PROGMEM = "Both"; const char mo98d[] PROGMEM = "No effect";
const char me98[]  PROGMEM = "Ventral = motor only; damage = paralysis, sensation remains.";

const char mq99[]  PROGMEM = "Alleles are alternative gene forms at:";
const char mo99a[] PROGMEM = "Different chromosomes"; const char mo99b[] PROGMEM = "Same locus on homologous chromosomes";
const char mo99c[] PROGMEM = "Random positions"; const char mo99d[] PROGMEM = "Only sex chromosomes";
const char me99[]  PROGMEM = "Alleles occupy same locus on homologous chromosomes.";

// --- MCQ MASTER TABLE (100 entries) ---
const MCQ MCQ_TABLE[] PROGMEM = {
  {mq0,mo0a,mo0b,mo0c,mo0d,0,me0},   // 0
  {mq1,mo1a,mo1b,mo1c,mo1d,1,me1},   // 1
  {mq2,mo2a,mo2b,mo2c,mo2d,2,me2},   // 2
  {mq3,mo3a,mo3b,mo3c,mo3d,2,me3},   // 3
  {mq4,mo4a,mo4b,mo4c,mo4d,0,me4},   // 4
  {mq5,mo5a,mo5b,mo5c,mo5d,1,me5},   // 5
  {mq6,mo6a,mo6b,mo6c,mo6d,2,me6},   // 6
  {mq7,mo7a,mo7b,mo7c,mo7d,1,me7},   // 7
  {mq8,mo8a,mo8b,mo8c,mo8d,3,me8},   // 8
  {mq9,mo9a,mo9b,mo9c,mo9d,2,me9},   // 9
  {mq10,mo10a,mo10b,mo10c,mo10d,0,me10}, // 10
  {mq11,mo11a,mo11b,mo11c,mo11d,2,me11}, // 11
  {mq12,mo12a,mo12b,mo12c,mo12d,0,me12}, // 12
  {mq13,mo13a,mo13b,mo13c,mo13d,1,me13}, // 13
  {mq14,mo14a,mo14b,mo14c,mo14d,1,me14}, // 14
  {mq15,mo15a,mo15b,mo15c,mo15d,1,me15}, // 15
  {mq16,mo16a,mo16b,mo16c,mo16d,1,me16}, // 16
  {mq17,mo17a,mo17b,mo17c,mo17d,2,me17}, // 17
  {mq18,mo18a,mo18b,mo18c,mo18d,0,me18}, // 18
  {mq19,mo19a,mo19b,mo19c,mo19d,2,me19}, // 19
  {mq20,mo20a,mo20b,mo20c,mo20d,2,me20}, // 20
  {mq21,mo21a,mo21b,mo21c,mo21d,1,me21}, // 21
  {mq22,mo22a,mo22b,mo22c,mo22d,2,me22}, // 22
  {mq23,mo23a,mo23b,mo23c,mo23d,1,me23}, // 23
  {mq24,mo24a,mo24b,mo24c,mo24d,1,me24}, // 24
  {mq25,mo25a,mo25b,mo25c,mo25d,1,me25}, // 25
  {mq26,mo26a,mo26b,mo26c,mo26d,2,me26}, // 26
  {mq27,mo27a,mo27b,mo27c,mo27d,2,me27}, // 27
  {mq28,mo28a,mo28b,mo28c,mo28d,1,me28}, // 28
  {mq29,mo29a,mo29b,mo29c,mo29d,2,me29}, // 29
  {mq30,mo30a,mo30b,mo30c,mo30d,1,me30}, // 30
  {mq31,mo31a,mo31b,mo31c,mo31d,2,me31}, // 31
  {mq32,mo32a,mo32b,mo32c,mo32d,1,me32}, // 32
  {mq33,mo33a,mo33b,mo33c,mo33d,1,me33}, // 33
  {mq34,mo34a,mo34b,mo34c,mo34d,2,me34}, // 34
  {mq35,mo35a,mo35b,mo35c,mo35d,1,me35}, // 35
  {mq36,mo36a,mo36b,mo36c,mo36d,0,me36}, // 36
  {mq37,mo37a,mo37b,mo37c,mo37d,1,me37}, // 37
  {mq38,mo38a,mo38b,mo38c,mo38d,2,me38}, // 38
  {mq39,mo39a,mo39b,mo39c,mo39d,2,me39}, // 39
  {mq40,mo40a,mo40b,mo40c,mo40d,1,me40}, // 40
  {mq41,mo41a,mo41b,mo41c,mo41d,2,me41}, // 41
  {mq42,mo42a,mo42b,mo42c,mo42d,2,me42}, // 42
  {mq43,mo43a,mo43b,mo43c,mo43d,1,me43}, // 43
  {mq44,mo44a,mo44b,mo44c,mo44d,3,me44}, // 44
  {mq45,mo45a,mo45b,mo45c,mo45d,2,me45}, // 45
  {mq46,mo46a,mo46b,mo46c,mo46d,1,me46}, // 46
  {mq47,mo47a,mo47b,mo47c,mo47d,0,me47}, // 47
  {mq48,mo48a,mo48b,mo48c,mo48d,1,me48}, // 48
  {mq49,mo49a,mo49b,mo49c,mo49d,2,me49}, // 49
  {mq50,mo50a,mo50b,mo50c,mo50d,1,me50}, // 50
  {mq51,mo51a,mo51b,mo51c,mo51d,2,me51}, // 51
  {mq52,mo52a,mo52b,mo52c,mo52d,2,me52}, // 52
  {mq53,mo53a,mo53b,mo53c,mo53d,1,me53}, // 53
  {mq54,mo54a,mo54b,mo54c,mo54d,0,me54}, // 54
  {mq55,mo55a,mo55b,mo55c,mo55d,1,me55}, // 55
  {mq56,mo56a,mo56b,mo56c,mo56d,2,me56}, // 56
  {mq57,mo57a,mo57b,mo57c,mo57d,1,me57}, // 57
  {mq58,mo58a,mo58b,mo58c,mo58d,0,me58}, // 58
  {mq59,mo59a,mo59b,mo59c,mo59d,1,me59}, // 59
  {mq60,mo60a,mo60b,mo60c,mo60d,2,me60}, // 60
  {mq61,mo61a,mo61b,mo61c,mo61d,0,me61}, // 61
  {mq62,mo62a,mo62b,mo62c,mo62d,1,me62}, // 62
  {mq63,mo63a,mo63b,mo63c,mo63d,0,me63}, // 63
  {mq64,mo64a,mo64b,mo64c,mo64d,2,me64}, // 64
  {mq65,mo65a,mo65b,mo65c,mo65d,2,me65}, // 65
  {mq66,mo66a,mo66b,mo66c,mo66d,3,me66}, // 66
  {mq67,mo67a,mo67b,mo67c,mo67d,0,me67}, // 67
  {mq68,mo68a,mo68b,mo68c,mo68d,1,me68}, // 68
  {mq69,mo69a,mo69b,mo69c,mo69d,0,me69}, // 69
  {mq70,mo70a,mo70b,mo70c,mo70d,1,me70}, // 70
  {mq71,mo71a,mo71b,mo71c,mo71d,1,me71}, // 71
  {mq72,mo72a,mo72b,mo72c,mo72d,0,me72}, // 72
  {mq73,mo73a,mo73b,mo73c,mo73d,1,me73}, // 73
  {mq74,mo74a,mo74b,mo74c,mo74d,1,me74}, // 74
  {mq75,mo75a,mo75b,mo75c,mo75d,1,me75}, // 75
  {mq76,mo76a,mo76b,mo76c,mo76d,1,me76}, // 76
  {mq77,mo77a,mo77b,mo77c,mo77d,1,me77}, // 77
  {mq78,mo78a,mo78b,mo78c,mo78d,1,me78}, // 78
  {mq79,mo79a,mo79b,mo79c,mo79d,2,me79}, // 79
  {mq80,mo80a,mo80b,mo80c,mo80d,1,me80}, // 80
  {mq81,mo81a,mo81b,mo81c,mo81d,0,me81}, // 81
  {mq82,mo82a,mo82b,mo82c,mo82d,1,me82}, // 82
  {mq83,mo83a,mo83b,mo83c,mo83d,1,me83}, // 83
  {mq84,mo84a,mo84b,mo84c,mo84d,1,me84}, // 84
  {mq85,mo85a,mo85b,mo85c,mo85d,0,me85}, // 85
  {mq86,mo86a,mo86b,mo86c,mo86d,1,me86}, // 86
  {mq87,mo87a,mo87b,mo87c,mo87d,2,me87}, // 87
  {mq88,mo88a,mo88b,mo88c,mo88d,1,me88}, // 88
  {mq89,mo89a,mo89b,mo89c,mo89d,0,me89}, // 89
  {mq90,mo90a,mo90b,mo90c,mo90d,2,me90}, // 90
  {mq91,mo91a,mo91b,mo91c,mo91d,1,me91}, // 91
  {mq92,mo92a,mo92b,mo92c,mo92d,1,me92}, // 92
  {mq93,mo93a,mo93b,mo93c,mo93d,1,me93}, // 93
  {mq94,mo94a,mo94b,mo94c,mo94d,1,me94}, // 94
  {mq95,mo95a,mo95b,mo95c,mo95d,1,me95}, // 95
  {mq96,mo96a,mo96b,mo96c,mo96d,1,me96}, // 96
  {mq97,mo97a,mo97b,mo97c,mo97d,1,me97}, // 97
  {mq98,mo98a,mo98b,mo98c,mo98d,1,me98}, // 98
  {mq99,mo99a,mo99b,mo99c,mo99d,1,me99}, // 99
};
#define MCQ_COUNT 100

// ============================================================
//  FACTS DATA (100 facts — same PROGMEM pattern)
// ============================================================
struct Fact { const char* fact; const char* detail; };

const char ff0[]  PROGMEM = "Alluvial soil is ex-situ (transported).";
const char fd0[]  PROGMEM = "Formed by deposition of river sediments, not parent rock below.";
const char ff1[]  PROGMEM = "Khadar is more fertile than Bhangar.";
const char fd1[]  PROGMEM = "Khadar = new alluvium, fine-grained, replenished by floods.";
const char ff2[]  PROGMEM = "Black Soil minerals: LIMCAP.";
const char fd2[]  PROGMEM = "Lime, Iron, Mg, Ca, Alumina, Potash: ideal for cotton.";
const char ff3[]  PROGMEM = "Black Soil is self-ploughing.";
const char fd3[]  PROGMEM = "Expands when wet, cracks deeply when dry, aerating soil.";
const char ff4[]  PROGMEM = "Red Soil is red due to iron oxide (Fe2O3).";
const char fd4[]  PROGMEM = "Residual soil; coarse & porous; poor water retention.";
const char ff5[]  PROGMEM = "Laterite soil is highly acidic and infertile.";
const char fd5[]  PROGMEM = "Heavy rain leaches silica; leaves insoluble iron/Al oxides.";
const char ff6[]  PROGMEM = "Red and Laterite soils share similar properties.";
const char fd6[]  PROGMEM = "Both: reddish color, porous/coarse texture, low nutrients.";
const char ff7[]  PROGMEM = "Sheet erosion: uniform slow removal of topsoil.";
const char fd7[]  PROGMEM = "Occurs on gentle slopes; often unnoticed till fertility drops.";
const char ff8[]  PROGMEM = "Gully erosion causes badland topography.";
const char fd8[]  PROGMEM = "Chambal Valley = most famous ravine badland in India.";
const char ff9[]  PROGMEM = "Terrace farming ideal for mountainous regions.";
const char fd9[]  PROGMEM = "Step-like platforms reduce runoff velocity on steep slopes.";
const char ff10[] PROGMEM = "Shelter belts check wind erosion in deserts.";
const char fd10[] PROGMEM = "Rows of trees break wind force, protecting sandy Rajasthan soil.";
const char ff11[] PROGMEM = "Humus determines soil fertility.";
const char fd11[] PROGMEM = "Decayed organic matter: binds soil, improves aeration & water.";
const char ff12[] PROGMEM = "Topography dictates soil profile depth.";
const char fd12[] PROGMEM = "Steep slopes: thin soils. Plains/valleys: thick deep profiles.";
const char ff13[] PROGMEM = "Nationalism in India driven by economic exploitation.";
const char fd13[] PROGMEM = "British policies destroyed handicrafts, uniting Indians vs colonialism.";
const char ff14[] PROGMEM = "Naoroji pioneered the 'Drain of Wealth' theory.";
const char fd14[] PROGMEM = "Mathematically showed Britain extracted wealth without equivalent return.";
const char ff15[] PROGMEM = "INC established December 1885.";
const char fd15[] PROGMEM = "Founded by A.O. Hume as 'safety valve' vs violent uprising.";
const char ff16[] PROGMEM = "W.C. Bonnerjee: inaugural INC President.";
const char fd16[] PROGMEM = "Presided over first session at Bombay with 72 delegates.";
const char ff17[] PROGMEM = "Vernacular Press Act 1878: Lord Lytton's repressive law.";
const char fd17[] PROGMEM = "Indian-language editors censored; English papers exempted.";
const char ff18[] PROGMEM = "Ilbert Bill 1883: huge political controversy.";
const char fd18[] PROGMEM = "Indian judges to try Europeans - white colonists forced amendment.";
const char ff19[] PROGMEM = "Raja Ram Mohan Roy: Father of Indian Renaissance.";
const char fd19[] PROGMEM = "Founded Brahmo Samaj 1828; abolished Sati via Bentinck 1829.";
const char ff20[] PROGMEM = "Jyotiba Phule: founded Satya Shodhak Samaj 1873.";
const char fd20[] PROGMEM = "Dismantled caste system; opened first girls school 1848 in Pune.";
const char ff21[] PROGMEM = "INC and Indian National Association: precursors of INC.";
const char fd21[] PROGMEM = "Banerjee and Naoroji's organizations merged to form unified INC.";
const char ff22[] PROGMEM = "Lytton lowered ICS exam age from 21 to 19.";
const char fd22[] PROGMEM = "Deliberate barrier to prevent Indian candidates from competing.";
const char ff23[] PROGMEM = "Delhi Durbar 1877: extreme indifference during famine.";
const char fd23[] PROGMEM = "Lytton spent lavishly for Queen's coronation while millions starved.";
const char ff24[] PROGMEM = "Mendel is 'Father of Genetics'.";
const char fd24[] PROGMEM = "Worked on Pisum sativum (garden pea) from 1856 to 1863.";
const char ff25[] PROGMEM = "A gene is a segment of DNA controlling hereditary traits.";
const char fd25[] PROGMEM = "Alleles are alternative forms at same locus on homologous chromosomes.";
const char ff26[] PROGMEM = "Homozygous = identical alleles; Heterozygous = different alleles.";
const char fd26[] PROGMEM = "TT or tt = homozygous; Tt = heterozygous.";
const char ff27[] PROGMEM = "Mendel's Law of Dominance: dominant masks recessive.";
const char fd27[] PROGMEM = "In Tt condition, only dominant T is expressed in phenotype.";
const char ff28[] PROGMEM = "Law of Segregation = Law of Purity of Gametes.";
const char fd28[] PROGMEM = "Alleles separate cleanly during meiosis without blending.";
const char ff29[] PROGMEM = "Law of Independent Assortment proven by dihybrid cross.";
const char fd29[] PROGMEM = "Different trait pairs segregate independently, giving 9:3:3:1.";
const char ff30[] PROGMEM = "Sex of human child determined by father's gamete.";
const char fd30[] PROGMEM = "Sperm carries X or Y; all eggs carry X chromosome only.";
const char ff31[] PROGMEM = "Colour blindness and haemophilia are X-linked recessive.";
const char fd31[] PROGMEM = "Males (XY) lack backup dominant allele to mask the defect.";
const char ff32[] PROGMEM = "Haemophilia extremely rare in females.";
const char fd32[] PROGMEM = "Female must inherit two defective X chromosomes to be affected.";
const char ff33[] PROGMEM = "Y-linked traits pass exclusively father to son.";
const char fd33[] PROGMEM = "Holandric inheritance; hypertrichosis only in males.";
const char ff34[] PROGMEM = "Mutations are sudden permanent changes in genetic material.";
const char fd34[] PROGMEM = "Radiation or chemicals can alter DNA, causing Sickle Cell Anaemia.";
const char ff35[] PROGMEM = "Acid produces hydronium ions (H3O+) in water.";
const char fd35[] PROGMEM = "H+ ions combine with water molecules to form hydronium ions.";
const char ff36[] PROGMEM = "Phosphorous acid H3PO3 is dibasic.";
const char fd36[] PROGMEM = "Only 2 H are bonded to O and ionizable; 3rd H bonded to P.";
const char ff37[] PROGMEM = "Acetic acid CH3COOH has basicity of one.";
const char fd37[] PROGMEM = "Only -COOH group hydrogen ionizes; other 3 H stay on carbon.";
const char ff38[] PROGMEM = "Alkali = base soluble in water.";
const char fd38[] PROGMEM = "All alkalis are bases; insoluble bases are NOT alkalis.";
const char ff39[] PROGMEM = "Dilute HCl is stronger than concentrated acetic acid.";
const char fd39[] PROGMEM = "Strength = degree of ionization, not concentration of solution.";
const char ff40[] PROGMEM = "Universal Indicator shows pH by color change.";
const char fd40[] PROGMEM = "Red/orange=acidic, yellow/green=neutral, blue/violet=basic.";
const char ff41[] PROGMEM = "Efflorescence: loss of water of crystallization.";
const char fd41[] PROGMEM = "Na2CO3.10H2O loses water in dry air, crumbles to powder.";
const char ff42[] PROGMEM = "Deliquescence: absorbs atmospheric moisture to form solution.";
const char fd42[] PROGMEM = "NaOH and FeCl3 absorb moisture until they dissolve themselves.";
const char ff43[] PROGMEM = "Hygroscopic: absorbs moisture without dissolving.";
const char fd43[] PROGMEM = "Conc H2SO4, CaO, silica gel: absorb water vapor, stay solid.";
const char ff44[] PROGMEM = "Normal salts: complete replacement of acidic hydrogen.";
const char fd44[] PROGMEM = "NaCl = normal salt. NaHSO4 = acid salt (retains H).";
const char ff45[] PROGMEM = "NO2 is a mixed acid anhydride.";
const char fd45[] PROGMEM = "Reacts with H2O to produce HNO2 and HNO3 simultaneously.";
const char ff46[] PROGMEM = "Soluble salts prepared by titration; insoluble by precipitation.";
const char fd46[] PROGMEM = "Precipitation (double decomposition) creates insoluble salts.";
const char ff47[] PROGMEM = "Neuron is the structural and functional unit of nervous system.";
const char fd47[] PROGMEM = "Specialized to detect, receive and conduct electrical impulses.";
const char ff48[] PROGMEM = "Neuron cell body = Cyton or Perikaryon.";
const char fd48[] PROGMEM = "Contains nucleus and neuroplasm; controls cell metabolism.";
const char ff49[] PROGMEM = "Nissl granules: ribosome-dense structures in cyton.";
const char fd49[] PROGMEM = "Highly active in synthesizing proteins for nerve transmission.";
const char ff50[] PROGMEM = "Dendrites carry impulses toward the cell body.";
const char fd50[] PROGMEM = "Large surface area for receiving chemical signals from axons.";
const char ff51[] PROGMEM = "Axon: long unbranched process carrying signals AWAY from cyton.";
const char fd51[] PROGMEM = "Can extend from mm to over 1 meter; ends in axon terminals.";
const char ff52[] PROGMEM = "Myelin Sheath: electrical insulator around peripheral axons.";
const char fd52[] PROGMEM = "Fats+proteins from Schwann cells; saltatory conduction.";
const char ff53[] PROGMEM = "Nodes of Ranvier: uninsulated spaces along myelinated axon.";
const char fd53[] PROGMEM = "Impulses jump node-to-node: significantly faster conduction.";
const char ff54[] PROGMEM = "Neurolemma: thin outermost membrane of peripheral axons.";
const char fd54[] PROGMEM = "Formed by Schwann cells; vital for damaged nerve regeneration.";
const char ff55[] PROGMEM = "Nerves: Sensory, Motor, and Mixed types.";
const char fd55[] PROGMEM = "Sensory to CNS, Motor to effectors, Mixed has both types.";
const char ff56[] PROGMEM = "Ganglion: cluster of nerve cell bodies outside CNS.";
const char fd56[] PROGMEM = "Relay and processing station for peripheral nervous system.";
const char ff57[] PROGMEM = "Synapse transmission is strictly unidirectional.";
const char fd57[] PROGMEM = "Chemical transmitters only from axon; receptors only on dendrite.";
const char ff58[] PROGMEM = "Acetylcholine and Noradrenaline: major autonomic neurotransmitters.";
const char fd58[] PROGMEM = "Acetylcholine: parasympathetic. Noradrenaline: sympathetic.";
const char ff59[] PROGMEM = "Synaptic Cleft: ~20 nanometers gap between two neurons.";
const char fd59[] PROGMEM = "Tiny gap requires chemical transmission to bridge the signal.";
const char ff60[] PROGMEM = "Reflex Action: rapid, involuntary, protective response.";
const char fd60[] PROGMEM = "Bypasses cerebral cortex for immediate protective reaction.";
const char ff61[] PROGMEM = "Reflex Arc: Receptor>Sensory>Cord>Motor>Effector.";
const char fd61[] PROGMEM = "Minimal complete pathway for a spinal reflex response.";
const char ff62[] PROGMEM = "CNS: Brain + Spinal Cord ONLY.";
const char fd62[] PROGMEM = "Primary integration and command center for all body functions.";
const char ff63[] PROGMEM = "Meninges: 3 protective membranes around brain.";
const char fd63[] PROGMEM = "Outer Dura mater, middle Arachnoid, inner Pia mater.";
const char ff64[] PROGMEM = "CSF circulates through brain ventricles and meninges.";
const char fd64[] PROGMEM = "Keeps brain buoyant, cushions it, and exchanges waste products.";
const char ff65[] PROGMEM = "Cranium: 8-boned skeletal structure protecting brain.";
const char fd65[] PROGMEM = "Rigid outer vault protects soft brain from direct compression.";
const char ff66[] PROGMEM = "Brain outer layer = Grey Matter (cytons and dendrites).";
const char fd66[] PROGMEM = "Grey = unmyelinated; white = myelinated axons (brain interior).";
const char ff67[] PROGMEM = "Cerebral cortex folded into Gyri and Sulci.";
const char fd67[] PROGMEM = "Allows billions of extra neurons in limited skull space.";
const char ff68[] PROGMEM = "Corpus Callosum: largest white matter structure in brain.";
const char fd68[] PROGMEM = "200M+ axons connecting left and right cerebral hemispheres.";
const char ff69[] PROGMEM = "Cerebrum divided into two hemispheres.";
const char fd69[] PROGMEM = "Left controls right side of body; Right controls left side.";
const char ff70[] PROGMEM = "Cerebellum beneath cerebrum: coordinates balance.";
const char fd70[] PROGMEM = "Adjusts motor commands for smooth, precise, balanced movements.";
const char ff71[] PROGMEM = "Medulla Oblongata: most critical brain region for survival.";
const char fd71[] PROGMEM = "Houses cardiac and respiratory centers; damage = immediate death.";
const char ff72[] PROGMEM = "Pons: major neural relay bridge.";
const char fd72[] PROGMEM = "Coordinates cerebellum, medulla, cerebrum, and spinal cord.";
const char ff73[] PROGMEM = "Diencephalon: Thalamus + Hypothalamus.";
const char fd73[] PROGMEM = "Gateway for sensory signals and homeostasis control center.";
const char ff74[] PROGMEM = "Hypothalamus links nervous system to endocrine system.";
const char fd74[] PROGMEM = "Controls pituitary gland: regulates hormones, appetite, sleep.";
const char ff75[] PROGMEM = "Spinal cord: main pathway for reflexes.";
const char fd75[] PROGMEM = "Processes simple reflexes locally without conscious brain command.";
const char ff76[] PROGMEM = "Spinal cord: grey matter INNER, white matter OUTER.";
const char fd76[] PROGMEM = "Exact inverse of the brain's grey/white matter arrangement.";
const char ff77[] PROGMEM = "12 cranial + 31 spinal nerve pairs in humans.";
const char fd77[] PROGMEM = "Cranial from brain directly; spinal from spinal cord.";
const char ff78[] PROGMEM = "Natural reflexes: inborn, permanent, and genetic.";
const char fd78[] PROGMEM = "Identical in all healthy species members; e.g. pupil constriction.";
const char ff79[] PROGMEM = "Conditioned reflexes: learned through experience.";
const char fd79[] PROGMEM = "Typing, cycling, playing instrument, Pavlov's dog experiment.";
const char ff80[] PROGMEM = "ANS: self-governing and involuntary.";
const char fd80[] PROGMEM = "Regulates internal visceral organs without conscious command.";
const char ff81[] PROGMEM = "Sympathetic NS: accelerates functions during stress.";
const char fd81[] PROGMEM = "Emergency: increases BP, dilates bronchi, shifts blood to muscles.";
const char ff82[] PROGMEM = "Parasympathetic NS: conserves and restores body energy.";
const char fd82[] PROGMEM = "Slows heart, stimulates digestion, constricts pupils.";
const char ff83[] PROGMEM = "Alcohol disrupts cerebellar pathways.";
const char fd83[] PROGMEM = "Causes ataxia: slurred speech, balance loss, poor coordination.";
const char ff84[] PROGMEM = "Association neurons reside entirely within CNS.";
const char fd84[] PROGMEM = "Process sensory data and pass appropriate signal to motor neurons.";
const char ff85[] PROGMEM = "Dorsal roots contain only sensory (afferent) fibers.";
const char fd85[] PROGMEM = "Dorsal root damage = complete loss of sensation in that region.";
const char ff86[] PROGMEM = "Ventral roots contain only motor (efferent) fibers.";
const char fd86[] PROGMEM = "Ventral root damage = paralysis; sensation remains intact.";
const char ff87[] PROGMEM = "Central canal of spinal cord continuous with brain ventricles.";
const char fd87[] PROGMEM = "Filled with CSF; keeps inner spinal tissues nourished.";
const char ff88[] PROGMEM = "Sensory = afferent (toward CNS); Motor = efferent (away).";
const char fd88[] PROGMEM = "Key mnemonic: SAME=Sensory Afferent Motor Efferent.";
const char ff89[] PROGMEM = "Stimulus: detectable change in environment.";
const char fd89[] PROGMEM = "Receptors like eyes or skin thermoreceptors detect these changes.";
const char ff90[] PROGMEM = "Effectors: muscles or glands performing body's response.";
const char fd90[] PROGMEM = "Contract or secrete in direct response to motor neuron stimulation.";
const char ff91[] PROGMEM = "Cerebellar damage: no paralysis but loss of coordination.";
const char fd91[] PROGMEM = "Muscles can still move; timing and smooth execution lost.";
const char ff92[] PROGMEM = "Voluntary actions originate exclusively in cerebrum.";
const char fd92[] PROGMEM = "Require conscious awareness and command by cerebral cortex.";
const char ff93[] PROGMEM = "Micturition: urinating process controlled by nervous system.";
const char fd93[] PROGMEM = "Bladder wall contracts; urethral sphincters relax coordinately.";
const char ff94[] PROGMEM = "ANS uses two neurons: preganglionic and postganglionic.";
const char fd94[] PROGMEM = "They meet at an autonomic ganglion between CNS and effector.";
const char ff95[] PROGMEM = "Nissl granules absent in the axon of a neuron.";
const char fd95[] PROGMEM = "Present in cyton and dendrites only for protein synthesis.";
const char ff96[] PROGMEM = "Myelin sheath formed by Schwann cells wrapping axon.";
const char fd96[] PROGMEM = "Multiple layers of Schwann cell membrane create the sheath.";
const char ff97[] PROGMEM = "Grey matter: cell bodies. White matter: myelinated axons.";
const char fd97[] PROGMEM = "The color difference is due to presence or absence of myelin.";
const char ff98[] PROGMEM = "Synaptic vesicles in presynaptic terminal release transmitters.";
const char fd98[] PROGMEM = "They fuse with membrane, releasing chemicals into synaptic cleft.";
const char ff99[] PROGMEM = "Allele is alternative gene form at specific chromosome locus.";
const char fd99[] PROGMEM = "Controls contrasting characters of same inheritable trait.";

const Fact FACT_TABLE[] PROGMEM = {
  {ff0,fd0},{ff1,fd1},{ff2,fd2},{ff3,fd3},{ff4,fd4},
  {ff5,fd5},{ff6,fd6},{ff7,fd7},{ff8,fd8},{ff9,fd9},
  {ff10,fd10},{ff11,fd11},{ff12,fd12},{ff13,fd13},{ff14,fd14},
  {ff15,fd15},{ff16,fd16},{ff17,fd17},{ff18,fd18},{ff19,fd19},
  {ff20,fd20},{ff21,fd21},{ff22,fd22},{ff23,fd23},{ff24,fd24},
  {ff25,fd25},{ff26,fd26},{ff27,fd27},{ff28,fd28},{ff29,fd29},
  {ff30,fd30},{ff31,fd31},{ff32,fd32},{ff33,fd33},{ff34,fd34},
  {ff35,fd35},{ff36,fd36},{ff37,fd37},{ff38,fd38},{ff39,fd39},
  {ff40,fd40},{ff41,fd41},{ff42,fd42},{ff43,fd43},{ff44,fd44},
  {ff45,fd45},{ff46,fd46},{ff47,fd47},{ff48,fd48},{ff49,fd49},
  {ff50,fd50},{ff51,fd51},{ff52,fd52},{ff53,fd53},{ff54,fd54},
  {ff55,fd55},{ff56,fd56},{ff57,fd57},{ff58,fd58},{ff59,fd59},
  {ff60,fd60},{ff61,fd61},{ff62,fd62},{ff63,fd63},{ff64,fd64},
  {ff65,fd65},{ff66,fd66},{ff67,fd67},{ff68,fd68},{ff69,fd69},
  {ff70,fd70},{ff71,fd71},{ff72,fd72},{ff73,fd73},{ff74,fd74},
  {ff75,fd75},{ff76,fd76},{ff77,fd77},{ff78,fd78},{ff79,fd79},
  {ff80,fd80},{ff81,fd81},{ff82,fd82},{ff83,fd83},{ff84,fd84},
  {ff85,fd85},{ff86,fd86},{ff87,fd87},{ff88,fd88},{ff89,fd89},
  {ff90,fd90},{ff91,fd91},{ff92,fd92},{ff93,fd93},{ff94,fd94},
  {ff95,fd95},{ff96,fd96},{ff97,fd97},{ff98,fd98},{ff99,fd99}
};
#define FACT_COUNT 100

// ============================================================
//  UTILITY: Read PROGMEM string safely into a buffer
// ============================================================
void pgmRead(const char* pgmPtr, char* buf, size_t maxLen) {
  strncpy_P(buf, pgmPtr, maxLen - 1);
  buf[maxLen - 1] = '\0';
}

// ============================================================
//  DISPLAY HELPERS
// ============================================================
void oledClear() {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
}

// Print multi-line wrapped text at (x,y) within maxWidth px
// Returns ending Y position
int printWrapped(const char* str, int x, int y, int maxWidth) {
  char word[24];
  char line[26];
  line[0] = '\0';
  int lineX = x;
  int lineY = y;
  int charW = 6; // textSize 1 = 6px wide

  int maxChars = maxWidth / charW;
  int wLen = 0;
  const char* p = str;

  display.setCursor(lineX, lineY);

  while (*p) {
    if (*p == ' ' || *p == '\0' || *(p + 1) == '\0') {
      if (*(p + 1) == '\0' && *p != ' ') {
        word[wLen++] = *p;
      }
      word[wLen] = '\0';

      if ((int)(strlen(line) + wLen) > maxChars) {
        display.setCursor(lineX, lineY);
        display.print(line);
        lineY += 9;
        strcpy(line, word);
      } else {
        if (strlen(line) > 0) strcat(line, " ");
        strcat(line, word);
      }
      wLen = 0;
    } else {
      word[wLen++] = *p;
    }
    p++;
  }
  if (strlen(line) > 0) {
    display.setCursor(lineX, lineY);
    display.print(line);
    lineY += 9;
  }
  return lineY;
}

// Countdown timer bar at bottom of screen
void showCountdown(unsigned long startMs, unsigned long totalMs, const char* label) {
  unsigned long elapsed = millis() - startMs;
  if (elapsed >= totalMs) elapsed = totalMs;
  int barW = (int)(((totalMs - elapsed) * 100UL) / totalMs);
  display.fillRect(0, 56, 128, 8, SSD1306_BLACK);
  display.fillRect(0, 56, barW, 8, SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 57);
  display.setTextColor(SSD1306_BLACK);
  display.print(label);
  display.setTextColor(SSD1306_WHITE);
  display.display();
}

// ============================================================
//  OLED ON / OFF via I2C command
// ============================================================
void oledOn() {
  Wire.beginTransmission(OLED_ADDR);
  Wire.write(0x00); Wire.write(0xAF);
  Wire.endTransmission();
}
void oledOff() {
  display.clearDisplay(); display.display();
  Wire.beginTransmission(OLED_ADDR);
  Wire.write(0x00); Wire.write(0xAE);
  Wire.endTransmission();
}

// ============================================================
//  BOOT SCREEN
// ============================================================
void showBoot() {
  oledClear();
  display.drawRect(0, 0, 128, 64, SSD1306_WHITE);
  display.setTextSize(2);
  display.setCursor(8, 6);
  display.println(F("WEATHER"));
  display.setTextSize(1);
  display.setCursor(18, 30);
  display.println(F("STATION  v4.0"));
  display.setCursor(12, 42);
  display.println(F("by  Soumyajit"));
  display.setCursor(18, 54);
  display.println(F("ESP32 + SSD1306"));
  display.display();
  delay(2500);

  oledClear();
  display.setTextSize(1);
  display.setCursor(22, 4);
  display.println(F("Initializing..."));
  for (int p = 0; p <= 100; p += 4) {
    display.fillRect(14, 28, p, 10, SSD1306_WHITE);
    display.drawRect(12, 26, 104, 14, SSD1306_WHITE);
    display.fillRect(46, 46, 36, 10, SSD1306_BLACK);
    display.setCursor(52, 48);
    display.print(p); display.print(F("%"));
    display.display();
    delay(25);
  }
  delay(300);

  oledClear();
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

// ============================================================
//  WIFI SCAN (Serial)
// ============================================================
void scanNetworks() {
  Serial.println(F("\n=== WiFi Scan ==="));
  int n = WiFi.scanNetworks();
  if (n <= 0) { Serial.println(F("No networks found.")); }
  else {
    Serial.printf("%d networks:\n  #  RSSI  SSID\n", n);
    for (int i = 0; i < n; i++)
      Serial.printf("  %d  %4d  %s\n", i+1, WiFi.RSSI(i), WiFi.SSID(i).c_str());
  }
  Serial.println(F("=================\n"));
  WiFi.scanDelete();
}

// ============================================================
//  WIFI CONNECT  (3 attempts then fail)
// ============================================================
bool connectWiFi() {
  scanNetworks();
  WiFi.mode(WIFI_STA);

  for (int attempt = 1; attempt <= 3; attempt++) {
    Serial.printf("[WiFi] Attempt %d/3\n", attempt);

    oledClear();
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.println(F("Connecting WiFi:"));
    display.println();
    String ssidStr = String(WIFI_SSID);
    if (ssidStr.length() > 20) ssidStr = ssidStr.substring(0, 17) + "...";
    display.println(ssidStr);
    display.printf("Attempt %d/3", attempt);
    display.display();

    WiFi.begin(WIFI_SSID, WIFI_PASS);

    int dots = 0, tries = 0;
    while (tries < 24 && WiFi.status() != WL_CONNECTED) {
      display.fillRect(0, 48, 128, 14, SSD1306_BLACK);
      display.setCursor(0, 50);
      display.print(F("Connecting"));
      for (int d = 0; d < (dots % 4); d++) display.print('.');
      display.display();
      delay(500);
      Serial.print('.');
      tries++; dots++;
    }
    Serial.println();

    if (WiFi.status() == WL_CONNECTED) {
      Serial.printf("[WiFi] Connected! IP:%s RSSI:%d\n",
                    WiFi.localIP().toString().c_str(), WiFi.RSSI());
      oledClear();
      display.setTextSize(1);
      display.setCursor(0, 0);
      display.println(F("WiFi Connected!"));
      display.println();
      display.print(F("IP: ")); display.println(WiFi.localIP());
      display.print(F("RSSI: ")); display.print(WiFi.RSSI());
      display.println(F(" dBm"));
      display.display();
      delay(1800);
      return true;
    }

    WiFi.disconnect(true);
    delay(1000);
  }

  // All 3 attempts failed
  Serial.println(F("[WiFi] All 3 attempts FAILED."));
  return false;
}

// ============================================================
//  NTP SYNC
// ============================================================
bool syncNTP() {
  configTime(GMT_OFFSET_SEC, DST_OFFSET_SEC, "pool.ntp.org", "time.nist.gov");
  struct tm ti;
  for (int i = 0; i < 20; i++) {
    if (getLocalTime(&ti, 500)) {
      char tb[8], db[14];
      strftime(tb, sizeof(tb), "%H:%M", &ti);
      strftime(db, sizeof(db), "%d %b %Y", &ti);
      g_time = tb; g_date = db;
      Serial.printf("[NTP] %s  %s\n", g_time.c_str(), g_date.c_str());
      return true;
    }
    delay(500);
  }
  Serial.println(F("[NTP] Sync failed."));
  return false;
}

// ============================================================
//  JSON EXTRACTOR
// ============================================================
String jsonExtract(const String& json, const String& key) {
  String token = "\""; token += key; token += "\":";
  int ki = json.indexOf(token);
  if (ki < 0) return "";
  int si = ki + token.length();
  while (si < (int)json.length() && json[si] == ' ') si++;
  if (si >= (int)json.length()) return "";
  char fc = json[si];
  if (fc == '"') {
    si++;
    int ei = json.indexOf('"', si);
    return (ei < 0) ? String("") : json.substring(si, ei);
  }
  if (fc == 't') return "1";
  if (fc == 'f') return "0";
  if (fc == 'n') return "";
  int ei = si;
  while (ei < (int)json.length()) {
    char c = json[ei];
    if (isDigit(c) || c == '.' || c == '-') ei++;
    else break;
  }
  return json.substring(si, ei);
}

// ============================================================
//  ICON SELECTION & DRAW
// ============================================================
String chooseIcon(const String& cond, int isDay) {
  if (isDay == 0) return "night";
  String c = cond; c.toLowerCase();
  if (c.indexOf("thunder") >= 0 || c.indexOf("storm")   >= 0) return "thunder";
  if (c.indexOf("rain")    >= 0 || c.indexOf("drizzle") >= 0) return "rain";
  if (c.indexOf("mist")    >= 0 || c.indexOf("fog")     >= 0) return "mist";
  if (c.indexOf("cloud")   >= 0 || c.indexOf("overcast")>= 0) return "cloud";
  if (c.indexOf("sunny")   >= 0 || c.indexOf("clear")   >= 0) return "sun";
  return "cloud";
}

void drawIcon(const String& icon) {
  const unsigned char* bmp = ICO_CLOUD;
  if      (icon == "sun")     bmp = ICO_SUN;
  else if (icon == "rain")    bmp = ICO_RAIN;
  else if (icon == "night")   bmp = ICO_NIGHT;
  else if (icon == "thunder") bmp = ICO_THUNDER;
  else if (icon == "mist")    bmp = ICO_MIST;
  display.drawBitmap(110, 0, bmp, 16, 16, SSD1306_WHITE);
}

// ============================================================
//  WEATHER DISPLAY
// ============================================================
void renderWeather() {
  oledClear();
  drawIcon(g_icon);

  display.setTextSize(2);
  display.setCursor(0, 0);
  display.print(g_time);

  display.setTextSize(1);
  display.setCursor(0, 18);
  display.print(g_date);

  display.drawFastHLine(0, 27, 128, SSD1306_WHITE);

  display.setTextSize(2);
  display.setCursor(0, 30);
  display.print(g_temp, 1);
  display.print((char)247); display.print('C');

  display.setTextSize(1);
  display.setCursor(76, 30);
  display.print(F("Feels"));
  display.setCursor(76, 40);
  display.print(g_feels, 0);
  display.print((char)247); display.print('C');

  display.setCursor(0, 50);
  display.print(F("H:")); display.print(g_humidity);
  display.print(F("% W:")); display.print(g_windKph);
  display.print(F("kph"));

  if (WiFi.status() == WL_CONNECTED) {
    int rssi = WiFi.RSSI();
    display.setCursor(100, 56);
    if      (rssi > -55) display.print(F("[3]"));
    else if (rssi > -70) display.print(F("[2]"));
    else                 display.print(F("[1]"));
  }
  display.display();
}

// ============================================================
//  FETCH WEATHER via HTTPS
// ============================================================
bool fetchWeather() {
  Serial.println(F("[API] Fetching weather..."));
  oledClear();
  display.setTextSize(1);
  display.setCursor(20, 26);
  display.println(F("Fetching data..."));
  display.display();

  syncNTP();

  String url = F("https://api.weatherapi.com/v1/current.json?key=");
  url += API_KEY; url += F("&q="); url += LOCATION; url += F("&aqi=no");

  WiFiClientSecure client;
  client.setInsecure();
  HTTPClient http;
  http.begin(client, url);
  http.setTimeout(15000);
  http.addHeader(F("Accept"), F("application/json"));

  int code = http.GET();
  Serial.printf("[API] HTTP code: %d\n", code);

  if (code != 200) {
    Serial.printf("[API] Error: %d\n", code);
    http.end();
    return false;
  }

  String payload = http.getString();
  http.end();
  if (payload.length() == 0) return false;

  g_temp     = jsonExtract(payload, "temp_c").toFloat();
  g_humidity = jsonExtract(payload, "humidity").toInt();
  g_feels    = jsonExtract(payload, "feelslike_c").toFloat();
  g_windKph  = (int)jsonExtract(payload, "wind_kph").toFloat();
  g_desc     = jsonExtract(payload, "text");
  g_icon     = chooseIcon(g_desc, jsonExtract(payload, "is_day").toInt());
  g_hasData  = true;
  payload    = "";

  Serial.printf("[API] %.1fC H:%d%% W:%dkph %s\n",
                g_temp, g_humidity, g_windKph, g_desc.c_str());
  return true;
}

// ============================================================
//  SOFT CLOCK TICK
// ============================================================
void clockTick() {
  struct tm ti;
  if (getLocalTime(&ti, 300)) {
    char tb[8];
    strftime(tb, sizeof(tb), "%H:%M", &ti);
    g_time = tb;
  }
}

// ============================================================
//  APOLOGY DISPLAY
// ============================================================
void showApology() {
  int idx = random(0, APOLOGY_COUNT);
  char buf[64];
  pgmRead((const char*)pgm_read_ptr(&APOLOGIES[idx]), buf, sizeof(buf));

  oledClear();
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println(F("Oh no!"));
  display.drawFastHLine(0, 10, 128, SSD1306_WHITE);
  printWrapped(buf, 0, 14, 128);
  display.display();
  Serial.printf("[APOLOGY] %s\n", buf);

  delay(5000);
}

// ============================================================
//  NUMBER GUESSING GAME
// ============================================================
void playGuessGame() {
  int myNum = random(1, 101);

  oledClear();
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println(F("Let's play a game!"));
  display.setCursor(0, 14);
  display.println(F("Pick 1-100 in"));
  display.println(F("your mind..."));
  display.display();
  delay(2000);

  // Countdown
  for (int i = 5; i >= 1; i--) {
    oledClear();
    display.setTextSize(1);
    display.setCursor(30, 10);
    display.println(F("Time to think!"));
    display.setTextSize(3);
    display.setCursor(55, 28);
    display.print(i);
    display.display();
    delay(1000);
  }

  // Reveal
  oledClear();
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println(F("My number is:"));
  display.setTextSize(3);
  int numX = (myNum >= 100) ? 26 : (myNum >= 10) ? 38 : 50;
  display.setCursor(numX, 20);
  display.print(myNum);
  display.setTextSize(1);
  display.setCursor(0, 52);
  display.print(F("How close were you? ;)"));
  display.display();
  Serial.printf("[GAME] My number was: %d\n", myNum);
  delay(4000);
}

// ============================================================
//  SLEEP ANIMATION
// ============================================================
void showSleepAnim() {
  const char* zFrames[] = { "Z", "Zz", "Zzz", "ZzZz", "ZzZzZ" };
  for (int rep = 0; rep < 3; rep++) {
    for (int f = 0; f < 5; f++) {
      oledClear();
      display.setTextSize(1);
      display.setCursor(40, 4);
      display.println(F("Going to sleep"));
      display.drawFastHLine(0, 14, 128, SSD1306_WHITE);

      // Moon icon placeholder (simple arc)
      display.drawCircle(64, 36, 14, SSD1306_WHITE);
      display.fillRect(68, 22, 14, 28, SSD1306_BLACK); // crescent

      display.setTextSize(1);
      display.setCursor(90, 28);
      display.println(zFrames[f]);

      display.setCursor(30, 54);
      display.println(F("Sweet dreams..."));
      display.display();
      delay(400);
    }
  }
}

// ============================================================
//  QUOTE LOOP  (32 quotes, 5 sec each)
// ============================================================
void runQuoteLoop() {
  char buf[64];

  // Shuffle order using a simple shuffle array
  uint8_t order[QUOTE_COUNT];
  for (int i = 0; i < QUOTE_COUNT; i++) order[i] = i;
  for (int i = QUOTE_COUNT - 1; i > 0; i--) {
    int j = random(0, i + 1);
    uint8_t tmp = order[i]; order[i] = order[j]; order[j] = tmp;
  }

  for (int qi = 0; qi < QUOTE_COUNT; qi++) {
    pgmRead((const char*)pgm_read_ptr(&QUOTES[order[qi]]), buf, sizeof(buf));

    oledClear();
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.println(F("-- Soumyajit Says --"));
    display.drawFastHLine(0, 10, 128, SSD1306_WHITE);

    printWrapped(buf, 0, 14, 128);

    // Quote progress
    display.setCursor(95, 56);
    display.print(qi + 1);
    display.print('/');
    display.print(QUOTE_COUNT);
    display.display();

    Serial.printf("[QUOTE %d] %s\n", order[qi], buf);

    unsigned long qStart = millis();
    while (millis() - qStart < 5000) {
      showCountdown(qStart, 5000, "");
      delay(200);
    }
  }
}

// ============================================================
//  MCQ MODE
// ============================================================
void runMCQMode() {
  char qBuf[64], eBuf[80];
  char opts[4][36];
  const char* optLabels[] = {"A) ", "B) ", "C) ", "D) "};

  // Show mode announcement
  oledClear();
  display.setTextSize(1);
  display.setCursor(0, 8);
  display.println(F("Wanna study?"));
  display.println(F("I'm picking..."));
  display.display();
  delay(1500);

  oledClear();
  display.setTextSize(1);
  display.setCursor(10, 20);
  display.println(F("Today's mode:"));
  display.setTextSize(2);
  display.setCursor(10, 36);
  display.println(F("MCQ!"));
  display.display();
  delay(2000);

  // Show 5 random questions
  uint8_t shown[5];
  for (int qi = 0; qi < 5; qi++) {
    // Pick a unique random question
    uint8_t idx;
    bool unique;
    do {
      idx = random(0, MCQ_COUNT);
      unique = true;
      for (int k = 0; k < qi; k++) if (shown[k] == idx) { unique = false; break; }
    } while (!unique);
    shown[qi] = idx;

    MCQ item;
    memcpy_P(&item, &MCQ_TABLE[idx], sizeof(MCQ));
    pgmRead(item.question, qBuf, sizeof(qBuf));
    pgmRead(item.explain,  eBuf, sizeof(eBuf));
    pgmRead(item.optA, opts[0], sizeof(opts[0]));
    pgmRead(item.optB, opts[1], sizeof(opts[1]));
    pgmRead(item.optC, opts[2], sizeof(opts[2]));
    pgmRead(item.optD, opts[3], sizeof(opts[3]));

    // --- Show Question for 10 sec ---
    unsigned long qStart = millis();
    while (millis() - qStart < 10000) {
      oledClear();
      display.setTextSize(1);
      display.setCursor(0, 0);
      display.print(F("Q")); display.print(idx + 1);
      display.print(F("/100  (Don't Google!)"));
      display.drawFastHLine(0, 9, 128, SSD1306_WHITE);
      printWrapped(qBuf, 0, 12, 128);
      showCountdown(qStart, 10000, "Q:");
      delay(200);
    }

    // --- Show Options (5 sec each) ---
    for (int o = 0; o < 4; o++) {
      unsigned long oStart = millis();
      while (millis() - oStart < 5000) {
        oledClear();
        display.setTextSize(1);
        display.setCursor(0, 0);
        display.print(F("Q")); display.print(idx + 1);
        display.print(F("  Opt ")); display.print(o + 1); display.print(F("/4"));
        display.drawFastHLine(0, 9, 128, SSD1306_WHITE);

        // Draw all options; highlight current one
        for (int op = 0; op < 4; op++) {
          int optY = 12 + op * 12;
          if (op == o) {
            display.fillRect(0, optY - 1, 128, 11, SSD1306_WHITE);
            display.setTextColor(SSD1306_BLACK);
          } else {
            display.setTextColor(SSD1306_WHITE);
          }
          display.setCursor(0, optY);
          display.print(optLabels[op]);
          // Truncate option text for display
          char truncOpt[22]; strncpy(truncOpt, opts[op], 21); truncOpt[21] = '\0';
          display.print(truncOpt);
          display.setTextColor(SSD1306_WHITE);
        }
        showCountdown(oStart, 5000, "");
        delay(200);
      }
    }

    // --- Show Answer for 5 sec ---
    unsigned long aStart = millis();
    while (millis() - aStart < 5000) {
      oledClear();
      display.setTextSize(1);
      display.setCursor(0, 0);
      display.print(F("ANS: "));
      display.print(optLabels[item.answer]);
      char truncAns[20]; strncpy(truncAns, opts[item.answer], 19); truncAns[19] = '\0';
      display.print(truncAns);
      display.drawFastHLine(0, 10, 128, SSD1306_WHITE);
      display.setCursor(0, 13);
      printWrapped(eBuf, 0, 13, 128);
      showCountdown(aStart, 5000, "Next:");
      delay(200);
    }

    Serial.printf("[MCQ] Q%d answered. Correct: %s\n", idx + 1, opts[item.answer]);
  }

  // Session summary
  oledClear();
  display.setTextSize(1);
  display.setCursor(15, 20);
  display.println(F("Great session!"));
  display.setCursor(5, 35);
  display.println(F("5 questions done!"));
  display.setCursor(0, 50);
  display.println(F("Knowledge = Power!"));
  display.display();
  delay(3000);
}

// ============================================================
//  FACTS MODE
// ============================================================
void runFactsMode() {
  char fBuf[80], dBuf[80];

  oledClear();
  display.setTextSize(1);
  display.setCursor(20, 15);
  display.println(F("Today's mode:"));
  display.setTextSize(2);
  display.setCursor(5, 32);
  display.println(F("Facts!"));
  display.display();
  delay(2000);

  // Show 5 random facts
  uint8_t shown[5];
  for (int fi = 0; fi < 5; fi++) {
    uint8_t idx;
    bool unique;
    do {
      idx = random(0, FACT_COUNT);
      unique = true;
      for (int k = 0; k < fi; k++) if (shown[k] == idx) { unique = false; break; }
    } while (!unique);
    shown[fi] = idx;

    Fact item;
    memcpy_P(&item, &FACT_TABLE[idx], sizeof(Fact));
    pgmRead(item.fact,   fBuf, sizeof(fBuf));
    pgmRead(item.detail, dBuf, sizeof(dBuf));

    // --- Fact heading (8 sec) ---
    unsigned long fStart = millis();
    while (millis() - fStart < 8000) {
      oledClear();
      display.setTextSize(1);
      display.setCursor(0, 0);
      display.print(F("Fact #")); display.print(idx + 1); display.print(F("/100"));
      display.drawFastHLine(0, 9, 128, SSD1306_WHITE);
      display.setCursor(0, 12);
      display.println(F("Did you know?"));
      printWrapped(fBuf, 0, 24, 128);
      showCountdown(fStart, 8000, "Fact:");
      delay(200);
    }

    // --- Detail (5 sec) ---
    unsigned long dStart = millis();
    while (millis() - dStart < 5000) {
      oledClear();
      display.setTextSize(1);
      display.setCursor(0, 0);
      display.print(F("Fact #")); display.print(idx + 1); display.print(F(" Detail:"));
      display.drawFastHLine(0, 9, 128, SSD1306_WHITE);
      printWrapped(dBuf, 0, 12, 128);
      showCountdown(dStart, 5000, "Next:");
      delay(200);
    }

    Serial.printf("[FACT] #%d: %s\n", idx + 1, fBuf);
  }

  oledClear();
  display.setTextSize(1);
  display.setCursor(10, 20);
  display.println(F("Brain loaded!"));
  display.setCursor(0, 36);
  display.println(F("5 facts absorbed."));
  display.display();
  delay(2500);
}

// ============================================================
//  STUDY MODE (50% MCQ, 50% Facts)
// ============================================================
void runStudyMode() {
  if (random(0, 2) == 0) {
    runMCQMode();
  } else {
    runFactsMode();
  }
}

// ============================================================
//  SLEEP MODE  (Game -> Animation -> Quotes)
// ============================================================
void runSleepMode() {
  oledClear();
  display.setTextSize(1);
  display.setCursor(20, 20);
  display.println(F("No internet..."));
  display.setCursor(8, 35);
  display.println(F("Sleep mode ON!"));
  display.display();
  delay(2000);

  playGuessGame();
  showSleepAnim();
  runQuoteLoop();
}

// ============================================================
//  NO-WIFI FLOW
// ============================================================
void runOfflineMode() {
  showApology();

  // 50% Study, 50% Sleep
  if (random(0, 2) == 0) {
    Serial.println(F("[OFFLINE] Study Mode selected."));
    runStudyMode();
  } else {
    Serial.println(F("[OFFLINE] Sleep Mode selected."));
    runSleepMode();
  }
}

// ============================================================
//  SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  delay(800);
  Serial.println(F("\n=== ESP32 Weather + Study Station v4.0 ==="));
  Serial.println(F("=== by Soumyajit ===\n"));

  randomSeed(esp_random());          // hardware RNG on ESP32

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);

  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println(F("OLED INIT FAILED!"));
    while (true) delay(1000);
  }
  display.setTextColor(SSD1306_WHITE);
  display.clearDisplay();
  display.display();

  showBoot();

  g_wifiOk = connectWiFi();

  if (g_wifiOk) {
    syncNTP();
    bool ok = fetchWeather();
    if (ok) {
      renderWeather();
    } else {
      oledClear();
      display.setTextSize(1);
      display.setCursor(0, 20);
      display.println(F("Weather fetch fail."));
      display.println(F("Showing clock..."));
      display.display();
      delay(2000);
    }
  } else {
    runOfflineMode();
    // After offline mode, retry wifi before entering loop
    g_wifiOk = connectWiFi();
    if (g_wifiOk) { syncNTP(); fetchWeather(); renderWeather(); }
  }
}

// ============================================================
//  LOOP
// ============================================================
void loop() {
  static unsigned long lastWeatherUpdate = 0;
  static unsigned long lastClockTick     = 0;

  // ── WiFi Watchdog ─────────────────────────────────────────
  if (WiFi.status() != WL_CONNECTED) {
    if (g_wifiOk) {
      g_wifiOk = false;
      Serial.println(F("[WiFi] Connection lost!"));
    }
    // Offline mode runs; then re-attempt WiFi
    runOfflineMode();
    g_wifiOk = connectWiFi();
    if (g_wifiOk) {
      syncNTP();
      if (fetchWeather()) {
        lastWeatherUpdate = millis();
        renderWeather();
      }
    }
    return;
  }

  g_wifiOk = true;

  // ── 30-minute weather refresh ──────────────────────────────
  if (millis() - lastWeatherUpdate >= WEATHER_INTERVAL_MS) {
    Serial.println(F("[LOOP] 30-min refresh."));
    oledClear();
    display.setTextSize(1);
    display.setCursor(30, 26);
    display.println(F("Updating..."));
    display.display();

    if (fetchWeather()) {
      lastWeatherUpdate = millis();
      renderWeather();
    } else {
      // Failed: retry after 30s not full 30 min
      lastWeatherUpdate = millis() - WEATHER_INTERVAL_MS + 30000UL;
      renderWeather(); // Show old data
    }
    return;
  }

  // ── Soft clock tick every 60 sec ──────────────────────────
  if (millis() - lastClockTick >= CLOCK_TICK_MS) {
    lastClockTick = millis();
    clockTick();
    if (g_hasData) renderWeather();
  }

  delay(1000);
}