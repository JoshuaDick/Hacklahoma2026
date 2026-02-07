

#include "bluetooth.h"
#


  // =============== Structures ===============
struct MotorPins {
  uint8_t IN1;
  uint8_t IN2;
  uint8_t IN3;
  uint8_t IN4;
};

 // ===========================================



 // ================ CONSTANTS ================

MotorPins motors[4] = { // tl tr bl br switch order
  {32,33,25,26}, // FL
  {23,22,21,19}, // FR
  {27,14,12,13}, // BL
  {16,4,2,15} // BR
};

enum Wheel {
  FRONT_LEFT = 0,
  FRONT_RIGHT = 1,
  BACK_LEFT = 2,
  BACK_RIGHT = 3
};

enum Direction {
  FORWARD,
  BACKWARD,
  IDLE,
  STOP
};

Direction commandToDirection(char c) {
  switch (c) {
    case 'f': return FORWARD;
    case 'b': return BACKWARD;
    case 'i': return IDLE;
    case 's': return STOP;
  }
}



const char* COMMAND_DELIMITER = " ";

 // ===========================================



 // ================ FUNCTIONS ================

void driveWheel(Wheel wheel, Direction dir) {
  MotorPins m = motors[wheel];

  switch (dir) {
    case FORWARD:
      digitalWrite(m.IN1, HIGH);
      digitalWrite(m.IN2, LOW);
      digitalWrite(m.IN3, HIGH);
      digitalWrite(m.IN4, LOW);
      break;

    case BACKWARD:
      digitalWrite(m.IN1, LOW);
      digitalWrite(m.IN2, HIGH);
      digitalWrite(m.IN3, LOW);
      digitalWrite(m.IN4, HIGH);
      break;

    case IDLE:
      break;

    default:
      digitalWrite(m.IN1, LOW);
      digitalWrite(m.IN2, LOW);
      digitalWrite(m.IN3, LOW);
      digitalWrite(m.IN4, LOW);
      break;
    };
  }

  
 // ===========================================



 // ================= RUNTIME =================

void setup() {
      // Initialize Serial
  Serial.begin(115200);
  Serial.println("Initialized serial");

    // Initialize all motor pins
  for (int i = 0; i < 4; i++) {
    pinMode(motors[i].IN1, OUTPUT);
    pinMode(motors[i].IN2, OUTPUT);
    pinMode(motors[i].IN3, OUTPUT);
    pinMode(motors[i].IN4, OUTPUT);

    digitalWrite(motors[i].IN1, LOW);
    digitalWrite(motors[i].IN2, LOW);
    digitalWrite(motors[i].IN3, LOW);
    digitalWrite(motors[i].IN4, LOW);
  }
  Serial.println("Initialized pins");

    // Initialize bluetooth
  initBluetooth();
  Serial.println("Initialized bluetooth");


  Serial.println("\n");
};


void loop() {
    // Wait until command is updated to something
  Serial.println("Command: " + command);

  if (!command.equals("none")) {
    if (command.length() != 4 && ) { // generalized command
      Serial.println("General command: " + command);
    } else { // Wheel specific commands
      driveWheel(FRONT_LEFT, commandToDirection(command.charAt(0)));
      driveWheel(FRONT_RIGHT, commandToDirection(command.charAt(1)));
      driveWheel(BACK_LEFT, commandToDirection(command.charAt(2)));
      driveWheel(BACK_RIGHT, commandToDirection(command.charAt(3)));
    }
    command = "none"; // Reset back to none when processed 
  }
  
  delay(250); // buffer between checks
};

