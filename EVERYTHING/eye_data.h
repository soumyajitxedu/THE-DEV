// eye_data.h — PROGMEM data: quotes, LUT, constants
#pragma once
#include <avr/pgmspace.h>

// ── Sin LUT 64 entries ──
static const uint8_t SIN64[64] PROGMEM = {
  128,140,152,165,176,188,199,209,218,226,234,240,245,250,253,255,
  255,255,253,250,245,240,234,226,218,209,199,188,176,165,152,140,
  128,115,103, 90, 79, 67, 56, 46, 37, 29, 21, 15, 10,  5,  2,  0,
    0,  0,  2,  5, 10, 15, 21, 29, 37, 46, 56, 67, 79, 90,103,115
};
inline int8_t fsin(uint8_t a){ return (int8_t)((int16_t)pgm_read_byte(&SIN64[a&63])-128); }
inline int8_t fcos(uint8_t a){ return fsin(a+16); }

// ── Expressions ──
#define E_HAPPY      0
#define E_CUTE       1
#define E_SLEEP      2
#define E_HIDE       3
#define E_SHOOT      4
#define E_ANGRY      5
#define E_LOOKLR     6
#define E_WOBBLE     7
#define E_HYPER      8
#define E_GREET      9
#define E_THINK     10
#define E_WRITE     11
#define E_BLINK     12
#define E_SUS       13
#define E_LOVE      14
#define E_SAD       15
#define E_SHOCK     16
#define E_BORED     17
#define E_FIRE      18
#define E_GLITCH    19
#define E_MUSIC     20
#define E_SPIN      21
#define E_DIZZY     22
#define E_CRY       23
#define E_LASER     24
#define E_HEART     25
#define E_ROLL      26
#define E_SMUG      27
#define E_DEAD      28
#define E_SPARKLE   29
#define E_COUNT     30

// ── System states ──
#define S_BOOT  0
#define S_ACTIVE 1
#define S_SLEEP 2
#define S_MUSIC 3

// ── Quote strings ──
static const char _q0[]  PROGMEM = "eww";
static const char _q1[]  PROGMEM = "hi!";
static const char _q2[]  PROGMEM = "watching you..";
static const char _q3[]  PROGMEM = "be kind always";
static const char _q4[]  PROGMEM = "work harder!";
static const char _q5[]  PROGMEM = "do whatever it takes";
static const char _q6[]  PROGMEM = "wanna be yours";
static const char _q7[]  PROGMEM = "you are special";
static const char _q8[]  PROGMEM = "dev: soumyajit";
static const char _q9[]  PROGMEM = "design: soham";
static const char _q10[] PROGMEM = "love mode on";
static const char _q11[] PROGMEM = "keep going!";
static const char _q12[] PROGMEM = "one step at a time";
static const char _q13[] PROGMEM = "you got this!";
static const char _q14[] PROGMEM = "rest is productive";
static const char _q15[] PROGMEM = "dream big";
static const char _q16[] PROGMEM = "eyes on prize";
static const char _q17[] PROGMEM = "main character";
static const char _q18[] PROGMEM = "404 excuses not found";
static const char _q19[] PROGMEM = "never stop";
static const char _q20[] PROGMEM = "let me play music!";
static const char _q21[] PROGMEM = "spinning up..";
static const char _q22[] PROGMEM = "getting dizzy";
static const char _q23[] PROGMEM = "don't cry bb";
static const char _q24[] PROGMEM = "pew pew!!";
static const char _q25[] PROGMEM = "<3 forever";
static const char _q26[] PROGMEM = "rolling rolling";
static const char _q27[] PROGMEM = "smug mode";
static const char _q28[] PROGMEM = "x_x";
static const char _q29[] PROGMEM = "sparkle time!";
static const char _q30[] PROGMEM = "i am alive!";
static const char _q31[] PROGMEM = "vibing hard";
static const char _q32[] PROGMEM = "stay curious";
static const char _q33[] PROGMEM = "blink blink";
static const char _q34[] PROGMEM = "sus..";

static const char* const QUOTES[35] PROGMEM = {
  _q0,_q1,_q2,_q3,_q4,_q5,_q6,_q7,_q8,_q9,
  _q10,_q11,_q12,_q13,_q14,_q15,_q16,_q17,_q18,_q19,
  _q20,_q21,_q22,_q23,_q24,_q25,_q26,_q27,_q28,_q29,
  _q30,_q31,_q32,_q33,_q34
};
#define NUM_Q 35

// mood-matched quote per expression
static const uint8_t EMOOD[E_COUNT] PROGMEM = {
  11,6,14,2,24,4,16,21,31,1,12,15,33,34,10,7,0,17,18,19,
  20,21,22,23,24,25,26,27,28,29
};
