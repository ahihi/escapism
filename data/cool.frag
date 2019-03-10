#version 410

uniform vec2 resolution;
uniform float time;
uniform float[10] sliders;

out vec4 out_color;

#define TAU 6.283185307179586

#define EYES sliders[0]
#define WOBBLY sliders[1]
#define PERFORATIONS sliders[2]
#define SHIFTY sliders[3]
#define REDYELLOW sliders[4]
#define HUE sliders[5]
#define CRUNCH sliders[6]
#define QUANTIZE sliders[7]

// ever watchful

#define UNIT 1.3

float scale(float l0, float r0, float l1, float r1, float x) {
  return (x - l0) / (r0 - l0) * (r1 - l1) + l1;
}

vec2 shear(float theta, vec2 p) {
  return vec2(p.x - p.y / tan(theta), p.y / sin(theta));
}

vec2 unshear(float theta, vec2 p) {
  float y = p.y * sin(theta);
  float x = p.x + y / tan(theta);
  return vec2(x, y);	
}

float rand(vec2 p){
  return fract(sin(dot(p.xy, vec2(1.3295, 4.12))) * 493022.1);
}

float timestep(float duration, float t) {
  return t == 0.0 ? t : floor(t / duration) * duration;
}

vec2 polar(vec2 p) {
  return vec2(atan(p.y, p.x), length(p));
}

vec2 cartesian(vec2 p) {
  return vec2(p.y * cos(p.x), p.y * sin(p.x))	;
}

vec3 eyes(float t, vec2 coord) {
  vec2 p0 = 2.0*(coord - 0.5 * resolution.xy) / resolution.xx;
	
  float unit = UNIT;
  float d_step0 = 1.1544 * unit;
  float d_step1 = 1.823 * unit;
  float d_step2 = 2.32 * unit;
  float d_step3 = 2.9757 * unit;
  float d_step4 = 1.21 * unit;
  float d_step5 = 0.93354 * unit;
	
  float t_step0 = timestep(d_step0, t);
  vec2 p0_rot = cartesian(polar(p0) + vec2(scale(0.0, 1.0, 0.0, TAU, rand(vec2(0.0, sqrt(t_step0)))), 0.0));
  vec2 p0_step = p0_rot + vec2(
                               scale(0.0, 1.0, -1.0, 1.0, rand(vec2(t_step0, 0.0))),
                               scale(0.0, 1.0, -1.0, 1.0, rand(vec2(0.0, t_step0)))
                               );
  float theta = TAU/4.0 + scale(0.0, 1.0, -1.0, 1.0, rand(vec2(0.0, timestep(d_step1, t))));

  float k_p1 = scale(0.0, 1.0, 2.0, 5.0, rand(vec2(timestep(d_step2, t), 0.0)));
  vec2 p1 = k_p1 * p0_step;
		
  vec2 p2 = shear(theta, p1);
	
  float d_move = 0.4;
  vec2 p_c = floor(p2) + 0.5 + scale(0.0, 1.0, -d_move, d_move, rand(vec2(timestep(d_step3, t), 0.0)));
  vec2 p3 = unshear(theta, p_c);
		
  float radius = scale(0.0, 1.0, 0.3, 0.6, rand(vec2(-42.0, timestep(0.21 * unit, t))));
  float rings = floor(scale(0.0, 1.0, 1.0, 4.0, rand(vec2(0.0, timestep(d_step4, t)))));
  float dist = distance(p3, p1);
  float ring_i = floor(dist/radius * rings);
  float ring_i2 = floor(dist / radius * 2.0 * rings);
  float ring_pos = fract(dist / radius * rings);
  float ring_pos2 = fract(dist / radius * 2.0 * rings);
  float r_pupil = radius / scale(0.0, 1.0, 1.5, 2.0, rand(vec2(timestep(0.322*unit, t), 0.0)));
	
  bool in_pupil = dist < r_pupil;
  bool in_iris = dist < radius;
	
  float bright = 1.0 - 0.75 * ring_i/rings * radius;
  float k_light = scale(0.0, 1.0, 0.76, 1.25, rand(vec2(-42.0, timestep(0.267*unit, t))));
  float light = k_light * bright * scale(0.0, 1.0, 0.5, 1.0, ring_pos);
  vec3 color = vec3(light, light, light);
  if(in_pupil) {
    color = vec3(0.0, 0.0, 0.0);
  } else if(in_iris) {
    vec3 iris0 = vec3(
                      scale(-1.0, 1.0, 0.2, 0.96, sin(timestep(0.2*unit, 0.6654*t))),
                      scale(0.0, 1.0, 0.0, 0.6, rand(floor(p2) * vec2(timestep(0.33*unit, .5*t + 53.0*floor(p2.x+p2.y)), -3211.1))),
                      scale(-1.0, 1.0, 0.1, 0.7, sin(timestep(0.115*unit, 0.533*t)))
                      );

    vec3 iris1 = iris0 * 0.7;
		
    color = mix(iris0, iris1, ring_pos2);
  }
  color *= scale(0.0, 1.0, 0.3, 1.0, rand(floor(p2) + vec2(timestep(1.0*unit, t), 0.1222)));
	
  return color;
}

vec3 ever_watchful(vec2 fragCoord, float t) {
  float t_0 = t;
  float t_1 = t_0 + UNIT*1.7;
  float mod_offset = 0.0;
  float mod0 = fract(t / UNIT);
  float mod_t = scale(-1.0, 1.0, 0.3, 0.7, sin(5.55 * t));
	
  vec3 color = eyes(t_0, fragCoord);
  if(rand(fragCoord.xy + vec2(0.0, timestep(0.125 * UNIT, t))) < mod_t) {
    color = eyes(t_1, fragCoord);	
  }
	
  return color;
}

// wobbly

vec3 c1a = vec3(0.0, 0.0, 0.0);
vec3 c1b = vec3(0.9, 0.0, 0.4);
vec3 c2a = vec3(0.0, 0.5, 0.9);
vec3 c2b = vec3(0.0, 0.0, 0.0);

vec3 wobbly(vec2 fragCoord, float t) {
  vec2 p = 2.0*(0.5 * resolution.xy - fragCoord.xy) / resolution.xx;
  float angle = atan(p.y, p.x);
  float turn = (angle + 0.5*TAU) / TAU;
  float radius = sqrt(p.x*p.x + p.y*p.y);
	
  float sine_kf = 19.0;
  float ka_wave_rate = 0.94;
  float ka_wave = sin(ka_wave_rate*t);
  float sine_ka = 0.35 * ka_wave;
  float sine2_ka = 0.47 * sin(0.87*t);
  float turn_t = turn + -0.0*t + sine_ka*sin(sine_kf*radius) + sine2_ka*sin(8.0 * angle);
  bool turn_bit = mod(10.0*turn_t, 2.0) < 1.0; 
	
  float blend_k = pow((ka_wave + 1.0) * 0.5, 1.0);
  vec3 c;
  if(turn_bit) {
    c = blend_k * c1a + (1.0 -blend_k) * c1b;
  } else {
    c = blend_k * c2a + (1.0 -blend_k) * c2b;
  }
  c *= 1.0 + 1.0*radius;
	
  return c;
}

// perforations

float mod289(float x){return x - floor(x * (1.0 / 289.0)) * 289.0;}
vec3 mod289(vec3 x) {return x - floor(x * (1.0 / 289.0)) * 289.0;}
vec4 mod289(vec4 x){return x - floor(x * (1.0 / 289.0)) * 289.0;}
vec4 permute(vec4 x){return mod289(((x * 34.0) + 1.0) * x);}

float noise(vec3 p){
  vec3 a = floor(p);
  vec3 d = p - a;
  d = d * d * (3.0 - 2.0 * d);

  vec4 b = a.xxyy + vec4(0.0, 1.0, 0.0, 1.0);
  vec4 k1 = permute(b.xyxy);
  vec4 k2 = permute(k1.xyxy + b.zzww);

  vec4 c = k2 + a.zzzz;
  vec4 k3 = permute(c);
  vec4 k4 = permute(c + 1.0);

  vec4 o1 = fract(k3 * (1.0 / 41.0));
  vec4 o2 = fract(k4 * (1.0 / 41.0));

  vec4 o3 = o2 * d.z + o1 * (1.0 - d.z);
  vec2 o4 = o3.yw * d.x + o3.xz * (1.0 - d.x);

  return o4.y * d.y + o4.x * (1.0 - d.y);
}

//
// GLSL textureless classic 3D noise "cnoise",
// with an RSL-style periodic variant "pnoise".
// Author:  Stefan Gustavson (stefan.gustavson@liu.se)
// Version: 2011-10-11
//
// Many thanks to Ian McEwan of Ashima Arts for the
// ideas for permutation and gradient selection.
//
// Copyright (c) 2011 Stefan Gustavson. All rights reserved.
// Distributed under the MIT license. See LICENSE file.
// https://github.com/stegu/webgl-noise
//

vec4 taylorInvSqrt(vec4 r)
{
  return 1.79284291400159 - 0.85373472095314 * r;
}

vec3 fade(vec3 t) {
  return t*t*t*(t*(t*6.0-15.0)+10.0);
}

// Classic Perlin noise
float cnoise(vec3 P)
{
  vec3 Pi0 = floor(P); // Integer part for indexing
  vec3 Pi1 = Pi0 + vec3(1.0); // Integer part + 1
  Pi0 = mod289(Pi0);
  Pi1 = mod289(Pi1);
  vec3 Pf0 = fract(P); // Fractional part for interpolation
  vec3 Pf1 = Pf0 - vec3(1.0); // Fractional part - 1.0
  vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
  vec4 iy = vec4(Pi0.yy, Pi1.yy);
  vec4 iz0 = Pi0.zzzz;
  vec4 iz1 = Pi1.zzzz;

  vec4 ixy = permute(permute(ix) + iy);
  vec4 ixy0 = permute(ixy + iz0);
  vec4 ixy1 = permute(ixy + iz1);

  vec4 gx0 = ixy0 * (1.0 / 7.0);
  vec4 gy0 = fract(floor(gx0) * (1.0 / 7.0)) - 0.5;
  gx0 = fract(gx0);
  vec4 gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
  vec4 sz0 = step(gz0, vec4(0.0));
  gx0 -= sz0 * (step(0.0, gx0) - 0.5);
  gy0 -= sz0 * (step(0.0, gy0) - 0.5);

  vec4 gx1 = ixy1 * (1.0 / 7.0);
  vec4 gy1 = fract(floor(gx1) * (1.0 / 7.0)) - 0.5;
  gx1 = fract(gx1);
  vec4 gz1 = vec4(0.5) - abs(gx1) - abs(gy1);
  vec4 sz1 = step(gz1, vec4(0.0));
  gx1 -= sz1 * (step(0.0, gx1) - 0.5);
  gy1 -= sz1 * (step(0.0, gy1) - 0.5);

  vec3 g000 = vec3(gx0.x,gy0.x,gz0.x);
  vec3 g100 = vec3(gx0.y,gy0.y,gz0.y);
  vec3 g010 = vec3(gx0.z,gy0.z,gz0.z);
  vec3 g110 = vec3(gx0.w,gy0.w,gz0.w);
  vec3 g001 = vec3(gx1.x,gy1.x,gz1.x);
  vec3 g101 = vec3(gx1.y,gy1.y,gz1.y);
  vec3 g011 = vec3(gx1.z,gy1.z,gz1.z);
  vec3 g111 = vec3(gx1.w,gy1.w,gz1.w);

  vec4 norm0 = taylorInvSqrt(vec4(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
  g000 *= norm0.x;
  g010 *= norm0.y;
  g100 *= norm0.z;
  g110 *= norm0.w;
  vec4 norm1 = taylorInvSqrt(vec4(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
  g001 *= norm1.x;
  g011 *= norm1.y;
  g101 *= norm1.z;
  g111 *= norm1.w;

  float n000 = dot(g000, Pf0);
  float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
  float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
  float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
  float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
  float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
  float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
  float n111 = dot(g111, Pf1);

  vec3 fade_xyz = fade(Pf0);
  vec4 n_z = mix(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fade_xyz.z);
  vec2 n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);
  float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x); 
  return 2.2 * n_xyz;
}

// Classic Perlin noise, periodic variant
float pnoise(vec3 P, vec3 rep)
{
  vec3 Pi0 = mod(floor(P), rep); // Integer part, modulo period
  vec3 Pi1 = mod(Pi0 + vec3(1.0), rep); // Integer part + 1, mod period
  Pi0 = mod289(Pi0);
  Pi1 = mod289(Pi1);
  vec3 Pf0 = fract(P); // Fractional part for interpolation
  vec3 Pf1 = Pf0 - vec3(1.0); // Fractional part - 1.0
  vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
  vec4 iy = vec4(Pi0.yy, Pi1.yy);
  vec4 iz0 = Pi0.zzzz;
  vec4 iz1 = Pi1.zzzz;

  vec4 ixy = permute(permute(ix) + iy);
  vec4 ixy0 = permute(ixy + iz0);
  vec4 ixy1 = permute(ixy + iz1);

  vec4 gx0 = ixy0 * (1.0 / 7.0);
  vec4 gy0 = fract(floor(gx0) * (1.0 / 7.0)) - 0.5;
  gx0 = fract(gx0);
  vec4 gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
  vec4 sz0 = step(gz0, vec4(0.0));
  gx0 -= sz0 * (step(0.0, gx0) - 0.5);
  gy0 -= sz0 * (step(0.0, gy0) - 0.5);

  vec4 gx1 = ixy1 * (1.0 / 7.0);
  vec4 gy1 = fract(floor(gx1) * (1.0 / 7.0)) - 0.5;
  gx1 = fract(gx1);
  vec4 gz1 = vec4(0.5) - abs(gx1) - abs(gy1);
  vec4 sz1 = step(gz1, vec4(0.0));
  gx1 -= sz1 * (step(0.0, gx1) - 0.5);
  gy1 -= sz1 * (step(0.0, gy1) - 0.5);

  vec3 g000 = vec3(gx0.x,gy0.x,gz0.x);
  vec3 g100 = vec3(gx0.y,gy0.y,gz0.y);
  vec3 g010 = vec3(gx0.z,gy0.z,gz0.z);
  vec3 g110 = vec3(gx0.w,gy0.w,gz0.w);
  vec3 g001 = vec3(gx1.x,gy1.x,gz1.x);
  vec3 g101 = vec3(gx1.y,gy1.y,gz1.y);
  vec3 g011 = vec3(gx1.z,gy1.z,gz1.z);
  vec3 g111 = vec3(gx1.w,gy1.w,gz1.w);

  vec4 norm0 = taylorInvSqrt(vec4(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
  g000 *= norm0.x;
  g010 *= norm0.y;
  g100 *= norm0.z;
  g110 *= norm0.w;
  vec4 norm1 = taylorInvSqrt(vec4(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
  g001 *= norm1.x;
  g011 *= norm1.y;
  g101 *= norm1.z;
  g111 *= norm1.w;

  float n000 = dot(g000, Pf0);
  float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
  float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
  float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
  float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
  float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
  float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
  float n111 = dot(g111, Pf1);

  vec3 fade_xyz = fade(Pf0);
  vec4 n_z = mix(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fade_xyz.z);
  vec2 n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);
  float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x); 
  return 2.2 * n_xyz;
}

vec2 rotate(float theta, vec2 p) {
  vec2 p_polar = polar(p);
  p_polar.x += theta;
  return cartesian(p_polar);
}

vec2 target(float theta, float delta, vec2 p) {
  return unshear(theta, floor(shear(theta, p) + delta) - delta + 0.5);
}

float perforations0(float theta, float rot, float scale, float r, vec2 p0) {
  vec2 p1 = scale * rotate(rot, p0);
  return distance(p1, target(theta, 0.5, p1)) - r;
}

vec3 blend(float k, vec3 c0, vec3 c1) {
  float k_clamp = clamp(k, 0.0, 1.0);
  return (1.0 - k) * c0 + k * c1;
}

vec3 perforations(vec2 fragCoord, float t) {
  vec2 p0 = 2.0*(fragCoord.xy - 0.5 * resolution.xy) / resolution.xx;
  vec2 p0_1 = vec2(p0.x, p0.y);
  vec2 p0_2 = vec2(p0.x, p0.y);
	
  vec2 p1_polar = polar(p0);
  float wave_cos =
    (0.05 * sin(0.102*t)) *
    cos(-0.7722*t + (0.0 + 10.0 * sin(0.114112*t)) * length(p0));
  p1_polar.y = p1_polar.y * pow(1.0 + wave_cos, 1.5);
  vec2 p1 = cartesian(p1_polar);
	
  float theta = TAU / 3.0;
	
  float rot1 = 0.0011 * TAU * t;
  float rot2 = rot1 + TAU / 12.0 + TAU / 4.0 * sin(0.05213 * t);
	
  float scale1 = 12.0 + 0.0 * sin(0.3212*t);
  float scale2 = 12.0;
	
  float r1 = 0.24;
  float r2 = 0.24;
	
  vec2 p2 = p1 + vec2(0.02 * sin(0.212 * t), 0.01 * cos(0.12 * t));
    
  float i1 = perforations0(theta, rot1, scale1, r1, p2);
  float i2 = perforations0(theta, rot2, scale2, r2, p1);
	
  vec2 bg_p1 = 10.2 * rotate(-0.03 * t, p2);
  float bg_noise1 = scale(-1.0, 1.0, 0.0, 1.0, cnoise(vec3(bg_p1.x, 0.5*t, bg_p1.y)));
    
  vec2 bg_p2 = 77.11 * rotate(0.024 * t, p2);
  float bg_noise2 = scale(-1.0, 1.0, 0.0, 1.0, cnoise(vec3(bg_p2.x, 1.33*t, bg_p2.y)));
    
  float bg_noise = bg_noise1 * pow(bg_noise2, 1.0);
    
  float fg_noise = 1.0 - 0.9 * noise(vec3(231.0*p1, 7.2*t));

  float k123 = 2.0;
  vec3 bg = blend(length(p0), k123*vec3(0.2), k123*vec3(0.1)) * bg_noise;
  vec3 fg = blend(length(p0), k123*vec3(0.4), k123*vec3(0.2)) * fg_noise;
	
  float satan = 0.06;
  float k = scale(-satan, satan, 0.0, 1.0, max(i1, i2));
	
  vec3 color;
  if(k < 0.0) {
    color = fg;
  } else if(k < 1.0) {
    color = blend(k, bg/0.2, fg);
  } else {
    color = blend(k-1.0, 0.9*bg, bg);   
  }
	    
  return color;
}

// shifty

vec3 shifty(vec2 fragCoord, float t) {
  vec2 p0 = 2.0*(0.5 * resolution.xy - fragCoord.xy) / resolution.xx;
  float angle0 = atan(p0.y, p0.x);
  float turn0 = (angle0 + TAU*0.5) / TAU;
  float radius0 = sqrt(p0.x*p0.x + p0.y*p0.y);
  
  float section = floor(pow(radius0*1000.0, 0.6) - t*0.7);
  float turn = turn0 + 0.04 * sin(1.3*(-t + 0.3*section));
  
  float segments = 18.0;
  float segment_angle = 1.0/segments;
  vec3 light0 = vec3(0.9, 0.2, 1.0);
  vec3 color = (mod(turn, 2.0*segment_angle) < segment_angle)
    ? vec3(0.0)
    : mix(0.3*light0, 0.5*light0, pow(noise(vec3(fragCoord*0.6, 11.0*time)), 3.0));
    
  return color;
}

// tools

vec3 rgb2hsl(vec3 rgb) {
  float r = rgb.r;
  float g = rgb.g;
  float b = rgb.b;
  float v, m, vm, r2, g2, b2;
  float h = 0.0;
  float s = 0.0;
  float l = 0.0;
  v = max(max(r, g), b);
  m = min(min(r, g), b);
  l = (m + v) / 2.0;
  if(l > 0.0) {
    vm = v - m;
    s = vm;
    if(s > 0.0) {
      s /= (l <= 0.5) ? (v + m) : (2.0 - v - m);
      r2 = (v - r) / vm;
      g2 = (v - g) / vm;
      b2 = (v - b) / vm;
      if(r == v) {
        h = (g == m ? 5.0 + b2 : 1.0 - g2);
      } else if(g == v) {
        h = (b == m ? 1.0 + r2 : 3.0 - b2);
      } else {
        h = (r == m ? 3.0 + g2 : 5.0 - r2);
      }
    }
  }
  h /= 6.0;
  return vec3(h, s, l);
}

vec3 hsl2rgb(vec3 hsl) {
  float h = hsl.x;
  float s = hsl.y;
  float l = hsl.z;
  float r = l;
  float g = l;
  float b = l;
  float v = (l <= 0.5) ? (l * (1.0 + s)) : (l + s - l*s);
  if(v > 0.0) {
    float m, sv;
    int sextant;
    float fract, vsf, mid1, mid2;
    m = l + l - v;
    sv = (v - m) / v;
    h *= 6.0;
    sextant = int(h);
    fract = h - float(sextant);
    vsf = v * sv * fract;
    mid1 = m + vsf;
    mid2 = v - vsf;
    if(sextant == 0) {
      r = v;
      g = mid1;
      b = m;
    } else if(sextant == 1) {
      r = mid2;
      g = v;
      b = m;
    } else if(sextant == 2) {
      r = m;
      g = v;
      b = mid1;
    } else if(sextant == 3) {
      r = m;
      g = mid2;
      b = v;
    } else if(sextant == 4) {
      r = mid1;
      g = m;
      b = v;
    } else if(sextant == 5) {
      r = v;
      g = m;
      b = mid2;
    }
  }
  return vec3(r, g, b);
}

vec3 hueshift(float dh, vec3 color) {
  vec3 hsl = rgb2hsl(color);
  hsl.x = fract(hsl.x + 1.0 + dh);
  return hsl2rgb(hsl);
}

vec3 crunch(float ds, vec2 xy, vec3 color) {
  float v = noise(vec3(xy, 10.9*time));
  return color * v;
}

vec3 redyellow(float k, vec3 color) {
  float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114));
  vec3 ry = gray < 0.5
    ? mix(vec3(0.0), vec3(1.0, 0.0, 0.0), gray*2.0)
    : mix(vec3(1.0, 0.0, 0.0), vec3(1.0, 1.0, 0.0), (gray-0.5)*2.0);
    
  return mix(color, ry, k);
}

float quantize(vec2 xy, float t) {
  vec2 rel = xy/resolution;
  float n = 100 / (1.0 + abs(rel.x - 0.5) * 9.0);
  float k_q = cnoise(vec3(floor(abs(rel.y-0.5)*2.0*n)*1.234, 0.0, 0.0));
  float duration = scale(0.0, 1.0, 0.7, 0.8, k_q);
  return timestep(duration, t);
}

// main

void main() {
  vec2 xy = gl_FragCoord.xy;
  vec2 xy0 = xy;
  
  float k_shift = CRUNCH * resolution.y / 30.0;
  float x_shift = scale(0.0, 1.0, -k_shift, k_shift, noise(vec3(0.6*xy, 10.0*time)));
  float y_shift = scale(0.0, 1.0, -k_shift, k_shift, noise(vec3(0.9*xy, 19.2*time)));
  xy += vec2(x_shift, y_shift);
  
  float t = mix(time, quantize(xy0, time), pow(QUANTIZE, 2.5));
  
  vec3 c_eyes = EYES * ever_watchful(xy, t);
  vec3 c_wobbly = WOBBLY * wobbly(xy, t);
  vec3 c_perforations = PERFORATIONS * perforations(xy, t);
  vec3 c_shifty = SHIFTY * shifty(xy, t);
  vec3 bg = c_eyes + c_wobbly + c_perforations + c_shifty;

  vec3 bg_ry = redyellow(REDYELLOW, bg);
  vec3 bg_hue = hueshift(HUE, bg_ry);
  
  out_color = vec4(bg_hue, 1.0);
}
