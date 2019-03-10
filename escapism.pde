import themidibus.*;

// name of the MIDI input device to use
// a list of available devices is printed on startup
String midi_device = "TouchOSC Bridge";

MidiBus midibus;
PShader sh;
boolean inited = false;
float[] sliders = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
int cc_0 = 30;

void setup() {
  //size(960, 540, P3D);
  fullScreen(P3D);
  noCursor();
  noStroke();
  noSmooth();
  textSize(24);

  MidiBus.list();
  midibus = new MidiBus(this, midi_device, -1);

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
  
  int offset = number - cc_0;
  
  if(0 <= offset && offset < sliders.length) {
    sliders[offset] = float(value) / 127.0;
  } else {
    println("controllerChange " + channel + " " + number + " " + value);
  }
}
