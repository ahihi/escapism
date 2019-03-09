import themidibus.*;

// name of the MIDI input device to use
// a list of available devices is printed on startup
String midi_device = "TouchOSC Bridge";

MidiBus midibus;
PShader sh;
boolean inited = false;
float[] sliders = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
int channel0 = 30;

void setup() {
  //size(960, 540, P3D);
  fullScreen(P3D);
  noCursor();
  noStroke();
  noSmooth();
  textSize(24);

  MidiBus.list();
  midibus = new MidiBus(this, midi_device, -1);

  /*
  pg = createGraphics(width, height, P3D);
  pg.beginDraw();
  pg.fill(255, 255, 255);
  pg.noFill();
  pg.endDraw();
  
  feedback_shader = loadShader("feedback.frag");
  pg2 = createGraphics(width, height, P3D);
  pg2.beginDraw();
  pg2.background(0, 0, 0);
  pg2.fill(255, 255, 255);
  pg2.noStroke();
  pg2.shader(feedback_shader);
  pg2.endDraw();
  */

  sh = loadShader("cool.frag");
  
  inited = true;
}

void draw() {
  float time = getTime();

  shader(sh);
  sh.set("resolution", width, height);
  sh.set("time", time);
  sh.set("sliders", sliders);
  rect(0, 0, width, height);
}

float getTime() {
  return 0.001*millis();
}

void controllerChange(int channel, int number, int value) {
  if(!inited) {
    return;
  }
  
  int offset = number - channel0;
  
  if(0 <= offset && offset < sliders.length) {
    sliders[offset] = float(value) / 127.0;
  } else {
    println("controllerChange " + channel + " " + number + " " + value);
  }
}
