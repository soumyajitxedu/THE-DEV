// EMO EYE v2.0 — Soumyajit & Soham — 25FPS dual robot eyes
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "eye_data.h"

#define SW 128
#define SH 64
Adafruit_SSD1306 D(SW, SH, &Wire, -1);

#define FPS      25
#define FMS      (1000UL/FPS)
#define EDUR     12000UL
#define SLPDUR   28000UL
#define ACTDUR   110000UL
#define QDUR     4500UL
#define MUSICDUR 6000UL

// eye geometry
#define LX 30
#define RX 98
#define EY 24
#define EW 42
#define EH 36
#define ER 12

// state
uint8_t  SYS=S_BOOT, EXP=E_BLINK, PEXP=E_COUNT;
uint32_t tState,tExpr,tFrame,tLast,tQuote,tMusic;
int8_t   pdx,pdy,tpx,tpy,skx,sky;
uint32_t tDrift;
char     qbuf[23];
bool     qshow;
// particles
#define NP 6
int8_t  px[NP],py[NP],pvx[NP],pvy[NP];
uint8_t plife[NP];

// ── util ──
int8_t lrp(int8_t a,int8_t b,uint8_t t){return a+(((int16_t)(b-a)*t)>>8);}
void spawnP(uint8_t i,int8_t x,int8_t y,int8_t vx,int8_t vy,uint8_t l){
  px[i]=x;py[i]=y;pvx[i]=vx;pvy[i]=vy;plife[i]=l;}
void tickP(){for(uint8_t i=0;i<NP;i++)if(plife[i]){px[i]+=pvx[i];py[i]+=pvy[i];pvy[i]++;if(plife[i])plife[i]--;}}
void drawP(){for(uint8_t i=0;i<NP;i++)if(plife[i])D.fillCircle(px[i],py[i],1,1);}
void killP(){for(uint8_t i=0;i<NP;i++)plife[i]=0;}
void tryP(int8_t x,int8_t y,int8_t vx,int8_t vy,uint8_t l){
  for(uint8_t i=0;i<NP;i++)if(!plife[i]){spawnP(i,x,y,vx,vy,l);return;}}

// ── quote ──
void loadQ(uint8_t idx){
  const char*p=(const char*)pgm_read_word(&QUOTES[idx%NUM_Q]);
  strncpy_P(qbuf,p,22);qbuf[22]=0;}
void showQ(uint8_t idx){loadQ(idx);tQuote=tFrame;qshow=true;}
void drawQ(){
  if(!qshow)return;
  uint32_t el=tFrame-tQuote;
  if(el>QDUR){qshow=false;return;}
  uint8_t nc=(uint8_t)min((uint32_t)22,el/75UL);
  uint8_t ln=strlen(qbuf);if(nc>ln)nc=ln;
  D.setTextSize(1);D.setTextColor(1);
  int16_t tx=(SW-nc*6)/2;if(tx<0)tx=0;
  D.setCursor(tx,56);
  for(uint8_t i=0;i<nc;i++)D.write(qbuf[i]);}

// ── 3D shadow eye ──
void oneEye(int16_t cx,int16_t cy,int16_t w,int16_t h,uint8_t r,uint8_t lt,uint8_t lb){
  // shadow (3D depth)
  D.fillRoundRect(cx-w/2+2,cy-h/2+2,w,h,r,0);
  // glow border
  D.drawRoundRect(cx-w/2-1,cy-h/2-1,w+2,h+2,r+1,1);
  // fill
  D.fillRoundRect(cx-w/2,cy-h/2,w,h,r,1);
  // inner ring
  D.drawRoundRect(cx-w/2+2,cy-h/2+2,w-4,h-4,r-2,0);
  // top lid
  if(lt>0){
    D.fillRect(cx-w/2-1,cy-h/2-1,w+2,lt+1,0);
    if(lt<h-2)D.drawFastHLine(cx-w/2+2,cy-h/2+lt,w-4,1);}
  // bot lid
  if(lb>0){
    D.fillRect(cx-w/2-1,cy-h/2+h-lb,w+2,lb+2,0);
    if(lb<h-2)D.drawFastHLine(cx-w/2+2,cy-h/2+h-lb-1,w-4,1);}
}
void eyes(uint8_t tl,uint8_t bl,uint8_t tr,uint8_t br,
          int16_t wl,int16_t hl,int16_t wr,int16_t hr){
  oneEye(LX+pdx+skx,EY+pdy+sky,wl,hl,ER,tl,bl);
  oneEye(RX+pdx+skx,EY+pdy+sky,wr,hr,ER,tr,br);}
void eyeS(uint8_t t,uint8_t b){eyes(t,b,t,b,EW,EH,EW,EH);}

// ── idle drift ──
void drift(){
  if(tFrame>=tDrift){
    tpx=(int8_t)random(-5,6);tpy=(int8_t)random(-2,3);
    tDrift=tFrame+random(200,700);}
  pdx=lrp(pdx,tpx,55);pdy=lrp(pdy,tpy,55);}

uint8_t prog(){
  uint32_t el=tFrame-tExpr;
  return el>=EDUR?255:(uint8_t)((el*255UL)/EDUR);}

uint8_t randE(){
  uint8_t e;do{e=(uint8_t)random(0,E_COUNT);}while(e==PEXP||e==E_MUSIC);
  PEXP=EXP;return e;}

// ── music visualiser ──
uint8_t barH[8];
void initBars(){for(uint8_t i=0;i<8;i++)barH[i]=random(4,28);}
void drawMusic(){
  uint8_t phase=(uint8_t)((tFrame/60)&0xFF);
  D.setTextSize(1);D.setTextColor(1);
  D.setCursor(28,2);D.print(F(">  NOW PLAYING  <"));
  // 8 bars, centred
  for(uint8_t i=0;i<8;i++){
    int8_t target=8+abs(fsin(phase+(i<<3)))/5;
    barH[i]=(uint8_t)lrp((int8_t)barH[i],(int8_t)target,80);
    uint8_t bx=8+i*14;
    uint8_t by=50-barH[i];
    D.fillRect(bx,by,10,barH[i],1);
    D.drawRect(bx-1,by-1,12,barH[i]+2,1);}
  // speaker icon
  D.drawRect(2,20,6,10,1);
  D.fillTriangle(8,18,14,14,14,34,1);
  D.drawPixel(16,20,1);D.drawPixel(17,18,1);D.drawPixel(17,28,1);D.drawPixel(16,26,1);}

// ── render ──
void render(uint8_t pg){
  uint8_t ph=(uint8_t)((tFrame/80)&0xFF);
  uint8_t tl=0,bl=0,tr=0,br=0;
  int16_t wl=EW,hl=EH,wr=EW,hr=EH;
  skx=0;sky=0;

  switch(EXP){
    case E_HAPPY:
      tl=9+(uint8_t)(abs(fsin(ph))/28);bl=5;tr=tl;br=bl;break;

    case E_CUTE:
      wl=wr=46;hl=hr=40;
      pdx+=fsin(ph/2)/40;break;

    case E_SLEEP:{
      tl=bl=tr=br=EH/2;
      uint8_t zf=(ph/18)%4;
      D.setTextSize(1);D.setTextColor(1);
      if(zf>=1){D.setCursor(106,20);D.print(F("z"));}
      if(zf>=2){D.setCursor(112,13);D.print(F("Z"));}
      if(zf>=3){D.setCursor(118,6); D.print(F("Z"));}
      break;}

    case E_HIDE:
      tl=14;bl=14;tr=0;br=0;
      tpx=6;break;

    case E_SHOOT:{
      tl=bl=tr=br=9;
      uint8_t sp=(ph/5)%20;
      if(sp<3){skx=(int8_t)random(-4,5);sky=(int8_t)random(-2,3);}
      if(sp==0)tryP(RX+24,EY,6,(int8_t)random(-1,2),10);
      if(sp==0)tryP(LX-24,EY,-6,(int8_t)random(-1,2),10);
      break;}

    case E_ANGRY:{
      tl=tr=13;bl=br=2;
      skx=(int8_t)random(-2,3);
      D.drawLine(LX-16+skx,EY-20,LX+16+skx,EY-14,1);
      D.drawLine(LX-16+skx,EY-19,LX+16+skx,EY-13,1);
      D.drawLine(RX+16+skx,EY-20,RX-16+skx,EY-14,1);
      D.drawLine(RX+16+skx,EY-19,RX-16+skx,EY-13,1);
      break;}

    case E_LOOKLR:
      tpx=(int8_t)(fsin(ph/3)/14);tpy=0;break;

    case E_WOBBLE:
      skx=fsin(ph*4)/50;sky=fsin(ph*4+16)/50;
      pdx+=fsin(ph*3)/30;pdy+=fsin(ph*3+8)/30;break;

    case E_HYPER:{
      uint8_t bc=ph%20;
      if(bc<3){tl=bl=tr=br=EH/2;}
      int8_t ps=abs(fsin(ph*3))/25;
      wl=wr=EW+ps;hl=hr=EH+ps;
      skx=fsin(ph*5)/60;break;}

    case E_GREET:{
      if(pg>60&&pg<180){tr=EH/2;br=EH/2;}
      uint8_t wy=(uint8_t)(abs(fsin(ph*3))/30);
      D.setTextSize(1);D.setTextColor(1);
      D.setCursor(54,2+wy);D.print(F("Hi!"));
      // hand wave arc
      D.drawCircle(64,5+wy,4,1);
      break;}

    case E_THINK:{
      tpx=-5;tpy=-4;
      uint8_t nd=(ph/16)%4;
      D.setTextSize(1);D.setTextColor(1);
      D.setCursor(46,50);
      for(uint8_t i=0;i<nd;i++)D.print(F(". "));
      // thought bubble
      D.drawCircle(LX+2,EY-20,3,1);
      D.drawCircle(LX+8,EY-25,4,1);
      D.drawCircle(LX+16,EY-28,5,1);
      break;}

    case E_WRITE:{
      tpy=4;tpx=0;
      uint8_t wpx=20+(ph%70);
      D.drawLine(wpx,52,wpx+4,47,1);
      D.fillTriangle(wpx,54,wpx+2,54,wpx+1,52,1);
      for(uint8_t i=0;i<min((int)(ph/8),8);i++)
        D.drawPixel(20+i*8+random(0,4),55,1);
      break;}

    case E_BLINK:{
      uint8_t v=0;
      if     (pg<25) v=(uint8_t)((uint16_t)pg*(EH/2)/25);
      else if(pg<45) v=EH/2;
      else if(pg<80) v=EH/2-(uint8_t)((uint16_t)(pg-45)*(EH/2)/35);
      else if(pg<110)v=(uint8_t)((uint16_t)(pg-80)*(EH/2)/30);
      else if(pg<130)v=EH/2;
      else if(pg<165)v=EH/2-(uint8_t)((uint16_t)(pg-130)*(EH/2)/35);
      tl=tr=v;bl=br=v>4?v-4:0;
      break;}

    case E_SUS:
      tl=bl=tr=br=11;
      tpx=fsin(ph/6)/16;break;

    case E_LOVE:{
      int8_t hx=64,hy=EY-2+fsin(ph)/32;
      D.fillCircle(hx-4,hy-2,3,1);
      D.fillCircle(hx+4,hy-2,3,1);
      D.fillTriangle(hx-7,hy+1,hx+7,hy+1,hx,hy+7,1);
      if((ph%22)==0)tryP(hx+(int8_t)random(-20,21),EY+16,0,-2,14);
      break;}

    case E_SAD:{
      tl=tr=9;
      tpy=3;
      uint8_t ty=EY+EH/2+((ph/3)%16);
      D.fillCircle(LX+6,ty,1,1);
      D.drawPixel(LX+6,ty-2,1);
      // sad brow
      D.drawLine(LX-12,EY-20,LX+12,EY-14,1);
      D.drawLine(RX+12,EY-20,RX-12,EY-14,1);
      break;}

    case E_SHOCK:
      wl=wr=50;hl=hr=44;
      D.fillCircle(LX+pdx+skx,EY+pdy+sky,4,0);
      D.fillCircle(RX+pdx+skx,EY+pdy+sky,4,0);break;

    case E_BORED:
      tl=tr=13;bl=br=2;
      tpx=fsin(ph/9)/22;
      if(pg>170){tpx=-7;tpy=4;
        D.drawRect(LX-32,EY+6,8,6,1);D.drawPixel(LX-29,EY+9,1);}
      break;

    case E_FIRE:{
      tl=tr=3;bl=br=3;wl=wr=44;hl=hr=38;
      for(uint8_t i=0;i<6;i++){
        int8_t fx=LX+(int8_t)random(-14,15);
        int8_t fy=EY-EH/2-3+(int8_t)random(-4,1);
        D.drawPixel(fx,fy,1);D.drawPixel(fx+1,fy-1,1);
        fx=RX+(int8_t)random(-14,15);
        D.drawPixel(fx,fy,1);D.drawPixel(fx+1,fy-1,1);}
      break;}

    case E_GLITCH:
      if(pg<100){
        for(uint8_t i=0;i<14;i++)
          D.drawFastHLine(random(0,128),random(0,64),random(3,22),1);
        skx=(int8_t)random(-5,6);sky=(int8_t)random(-4,5);
      }else{
        uint8_t rv=(pg<190)?EH/2:(uint8_t)max(0,(int)(EH/2-(pg-190)/4));
        tr=br=rv;}
      break;

    case E_MUSIC: // handled in main
      break;

    case E_SPIN:{
      // spinning concentric rings
      uint8_t ang=(uint8_t)((tFrame/30)&0xFF);
      for(uint8_t r=4;r<16;r+=4){
        int8_t ox=fcos(ang+(r*4))/8,oy=fsin(ang+(r*4))/8;
        D.drawCircle(LX+ox,EY+oy,r,1);
        D.drawCircle(RX+ox,EY+oy,r,1);}
      tl=bl=tr=br=0;
      break;}

    case E_DIZZY:{
      skx=fsin(ph*5)/45;sky=fcos(ph*5)/45;
      uint8_t ang=(uint8_t)((tFrame/25)&0xFF);
      // X pupils
      int8_t ox=fcos(ang)/30,oy=fsin(ang)/30;
      tl=bl=tr=br=5;
      // draw X marks after eyes
      D.drawLine(LX-3,EY-3,LX+3,EY+3,0);D.drawLine(LX+3,EY-3,LX-3,EY+3,0);
      D.drawLine(RX-3,EY-3,RX+3,EY+3,0);D.drawLine(RX+3,EY-3,RX-3,EY+3,0);
      break;}

    case E_CRY:{
      tl=tr=7;
      for(uint8_t i=0;i<3;i++){
        uint8_t ty=EY+EH/2+((ph/2+i*8)%22);
        D.fillCircle(LX+4+i*2,ty,1,1);
        D.fillCircle(RX+4+i*2,ty,1,1);}
      break;}

    case E_LASER:{
      tl=tr=12;bl=br=12;
      // laser beam
      uint8_t lt2=(ph/3)%4;
      if(lt2<2){
        D.drawFastHLine(RX+22,EY,SW-RX-22,1);
        D.drawFastHLine(RX+22,EY-1,SW-RX-22,1);}
      break;}

    case E_HEART:{
      // heart eyes
      int8_t hbx=(int8_t)(fsin(ph)/60);
      // left heart
      D.fillCircle(LX-3+hbx,EY-2,4,1);D.fillCircle(LX+3+hbx,EY-2,4,1);
      D.fillTriangle(LX-7+hbx,EY+1,LX+7+hbx,EY+1,LX+hbx,EY+7,1);
      // right heart
      D.fillCircle(RX-3+hbx,EY-2,4,1);D.fillCircle(RX+3+hbx,EY-2,4,1);
      D.fillTriangle(RX-7+hbx,EY+1,RX+7+hbx,EY+1,RX+hbx,EY+7,1);
      // black outlines
      D.fillCircle(LX+hbx,EY,2,0);D.fillCircle(RX+hbx,EY,2,0);
      return;}// skip normal draw

    case E_ROLL:{
      tpx=fsin(ph/2)/12;tpy=fcos(ph/2)/20;
      skx=fsin(ph*3)/55;sky=fcos(ph*3)/55;
      break;}

    case E_SMUG:
      tl=4;bl=4;tr=14;br=0;// right eye more closed
      tpx=3;tpy=-1;break;

    case E_DEAD:
      // X eyes
      tl=bl=tr=br=EH/2;
      // draw X after
      D.drawLine(LX-8,EY-8,LX+8,EY+8,1);D.drawLine(LX+8,EY-8,LX-8,EY+8,1);
      D.drawLine(LX-7,EY-8,LX+9,EY+8,1);
      D.drawLine(RX-8,EY-8,RX+8,EY+8,1);D.drawLine(RX+8,EY-8,RX-8,EY+8,1);
      D.drawLine(RX-7,EY-8,RX+9,EY+8,1);
      break;

    case E_SPARKLE:{
      // sparkle burst every frame
      for(uint8_t i=0;i<4;i++){
        uint8_t ang=(uint8_t)(ph*3+i*16);
        int8_t sx=LX+(int8_t)(fcos(ang)/9);
        int8_t sy=EY+(int8_t)(fsin(ang)/9);
        D.drawPixel(sx,sy,1);
        sx=RX+(int8_t)(fcos(ang)/9);
        D.drawPixel(sx,sy,1);
        // cross sparkle
        D.drawLine(sx-3,sy,sx+3,sy,1);D.drawLine(sx,sy-3,sx,sy+3,1);}
      break;}

    default:break;
  }

  // draw eyes normally (unless heart/dead overrides)
  if(EXP!=E_HEART && EXP!=E_SPIN && EXP!=E_MUSIC){
    eyes(tl,bl,tr,br,wl,hl,wr,hr);
  }

  // post-draw overlays
  if(EXP==E_SHOCK){
    D.fillCircle(LX+pdx+skx,EY+pdy+sky,4,0);
    D.fillCircle(RX+pdx+skx,EY+pdy+sky,4,0);}
  if(EXP==E_DIZZY){
    D.drawLine(LX-4,EY-4,LX+4,EY+4,0);D.drawLine(LX+4,EY-4,LX-4,EY+4,0);
    D.drawLine(RX-4,EY-4,RX+4,EY+4,0);D.drawLine(RX+4,EY-4,RX-4,EY+4,0);}
  if(EXP==E_DEAD){
    D.drawLine(LX-8,EY-8,LX+8,EY+8,1);D.drawLine(LX+8,EY-8,LX-8,EY+8,1);
    D.drawLine(RX-8,EY-8,RX+8,EY+8,1);D.drawLine(RX+8,EY-8,RX-8,EY+8,1);}
}

// ── boot ──
void boot(){
  D.clearDisplay();
  // animated eye logo
  D.fillRoundRect(30,10,28,22,9,1);
  D.fillRoundRect(70,10,28,22,9,1);
  D.drawRoundRect(29,9,30,24,10,1);
  D.drawRoundRect(69,9,30,24,10,1);
  D.setTextSize(1);D.setTextColor(1);
  D.setCursor(14,38);D.print(F("Eye v2.0 TURBO"));
  D.setCursor(10,50);D.print(F("Soumyajit & Soham"));
  D.display();
  delay(2200);
  SYS=S_ACTIVE;tState=tExpr=millis();
  EXP=randE();}

void setup(){
  randomSeed(analogRead(A0));
  Wire.begin();
  if(!D.begin(SSD1306_SWITCHCAPVCC,0x3C))for(;;);
  D.clearDisplay();D.display();
  initBars();
  boot();}

void loop(){
  uint32_t now=millis();
  if(now-tLast<FMS)return;
  tLast=now; tFrame=now;
  D.clearDisplay();

  if(SYS==S_BOOT){boot();return;}

  if(SYS==S_MUSIC){
    drawMusic();drawQ();
    if(now-tMusic>=MUSICDUR){
      SYS=S_ACTIVE;tExpr=now;EXP=randE();
      showQ(pgm_read_byte(&EMOOD[EXP]));}
    D.display();return;}

  if(SYS==S_SLEEP){
    if(now-tState>=SLPDUR){
      SYS=S_ACTIVE;tState=tExpr=now;EXP=randE();
      killP();pdx=pdy=0;}
    else{
      uint8_t ph=(uint8_t)((now/80)&0xFF);
      eyeS(EH/2,EH/2);
      uint8_t zf=(ph/18)%4;
      D.setTextSize(1);D.setTextColor(1);
      if(zf>=1){D.setCursor(108,20);D.print(F("z"));}
      if(zf>=2){D.setCursor(114,13);D.print(F("Z"));}
      if(zf>=3){D.setCursor(120,6); D.print(F("Z"));}
    }
    D.display();return;}

  // S_ACTIVE
  if(now-tState>=ACTDUR){
    SYS=S_SLEEP;tState=now;EXP=E_SLEEP;killP();qshow=false;
    D.display();return;}

  if(now-tExpr>=EDUR){
    EXP=randE();tExpr=now;killP();
    // 5% rare shoot
    if(random(100)<5)EXP=E_SHOOT;
    // 8% music event
    if(random(100)<8){SYS=S_MUSIC;tMusic=now;initBars();
      showQ(20);D.display();return;}
    showQ(pgm_read_byte(&EMOOD[EXP]));}

  drift();
  render(prog());
  tickP();drawP();drawQ();
  D.display();}
