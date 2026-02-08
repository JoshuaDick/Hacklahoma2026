

// This program controls the h-bridge for each motor of our RC car (1 h-bridge per motor). It recieves commands via
// a bluetooth signal that is connected to a mobile phone. There are preset commands that the user needs to enter to
// remotely change the behavior of the motors, thus altering the way the car drives. When the command is processed, it
// activates the "mode" it commanded and keeps it on, so if the user commands "forward", it will keep going forward
// until another valid command is recieved. Thus, overriding the previous behavior.



#include "bluetooth.h"
#include <ESP32Servo.h>



  // =============== Structures ===============
struct MotorPins {
  uint8_t IN1;
  uint8_t IN2;
  uint8_t IN3;
  uint8_t IN4;
};

 // ===========================================



 // ================ CONSTANTS ================

const uint8_t LEFT_SERVO_PIN = 23;
const uint8_t RIGHT_SERVO_PIN = 22;

Servo leftServo;
Servo rightServo;


MotorPins motors[4] = { // tl tr bl br switch order
  {32,33,25,26}, // FL
  {19,18,5,17}, // FR
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
  }
}


 // ===========================================



 // ================ FUNCTIONS ================




  
 // ===========================================



 // ============ DRIVE FUNCTIONS ==============

void driveWheel(Wheel wheel, Direction dir) {
  MotorPins m = motors[wheel];

  switch (dir) {
    case FORWARD:
      digitalWrite(m.IN1, LOW);
      digitalWrite(m.IN2, HIGH);
      digitalWrite(m.IN3, LOW);
      digitalWrite(m.IN4, HIGH);
      break;

    case BACKWARD:
      digitalWrite(m.IN1, HIGH);
      digitalWrite(m.IN2, LOW);
      digitalWrite(m.IN3, HIGH);
      digitalWrite(m.IN4, LOW);
      break;

    case IDLE:
      digitalWrite(m.IN1, HIGH);
      digitalWrite(m.IN2, HIGH);
      digitalWrite(m.IN3, LOW);
      digitalWrite(m.IN4, LOW);
      break;
  }
}

void driveForward() {
  driveWheel(FRONT_LEFT, FORWARD);
  driveWheel(FRONT_RIGHT, FORWARD);
  driveWheel(BACK_LEFT, FORWARD);
  driveWheel(BACK_RIGHT, FORWARD);
}

void driveBackward() {
  driveWheel(FRONT_LEFT, BACKWARD);
  driveWheel(FRONT_RIGHT, BACKWARD);
  driveWheel(BACK_LEFT, BACKWARD);
  driveWheel(BACK_RIGHT, BACKWARD);
}

void driveIdle() {
  driveWheel(FRONT_LEFT, IDLE);
  driveWheel(FRONT_RIGHT, IDLE);
  driveWheel(BACK_LEFT, IDLE);
  driveWheel(BACK_RIGHT, IDLE);
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

    digitalWrite(motors[i].IN1, HIGH);
    digitalWrite(motors[i].IN2, HIGH);
    digitalWrite(motors[i].IN3, LOW);
    digitalWrite(motors[i].IN4, LOW);
  }
  Serial.println("Initialized pins");

    // Initialize servos
  leftServo.attach(LEFT_SERVO_PIN);
  rightServo.attach(RIGHT_SERVO_PIN);

    // Initialize bluetooth
  initBluetooth();
  Serial.println("Initialized bluetooth");


  Serial.println("\n");
};


void loop() {
    // Wait until command is updated to something
  Serial.println("Command: " + command);

  if (!command.equals("none")) {
    if (command.length() != 4) { // generalized command
      Serial.println("General command: " + command);
      if (command == "forward") {
        driveForward();
      } else if (command == "backward") {
        driveBackward();
      } else if (command == "idle") {
        driveIdle();
      }

    } else { // Wheel specific commands
      Serial.println("Custom command");
      driveWheel(FRONT_LEFT, commandToDirection(command.charAt(0)));
      driveWheel(FRONT_RIGHT, commandToDirection(command.charAt(1)));
      driveWheel(BACK_LEFT, commandToDirection(command.charAt(2)));
      driveWheel(BACK_RIGHT, commandToDirection(command.charAt(3)));
    }
    command = "none"; // Reset back to none when processed 
  }
  
  delay(250); // buffer between checks

  leftServo.write(0);
  rightServo.write(45);
  delay(1000);
  leftServo.write(90);
  rightServo.write(135);
  delay(1000);
};

