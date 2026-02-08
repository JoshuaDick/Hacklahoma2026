

// This program controls the h-bridge for each motor of our RC car (1 h-bridge per motor). It recieves commands via
// a bluetooth signal that is connected to a mobile phone. There are preset commands that the user needs to enter to
// remotely change the behavior of the motors, thus altering the way the car drives. When the command is processed, it
// activates the "mode" it commanded and keeps it on, so if the user commands "forward", it will keep going forward
// until another valid command is recieved. Thus, overriding the previous behavior.



#include "bluetooth.h"
#include <ESP32Servo.h>



  // =============== Structures ===============


 // ===========================================



 // ================ CONSTANTS ================

const uint8_t LEFT_SERVO_PIN = 22; //34
const uint8_t RIGHT_SERVO_PIN = 21; //35
const uint8_t NUM_FLAPS = 3;
Servo leftServo;
Servo rightServo;

const uint8_t TASER_PIN = 23;

const uint8_t motor11 = 27;
const uint8_t motor12 = 14;
const uint8_t motor21 = 16;
const uint8_t motor22 = 4;
const uint8_t motor31 = 12;
const uint8_t motor32 = 13;
const uint8_t motor41 = 2;
const uint8_t motor42 = 15;

 // ===========================================



 // ================ FUNCTIONS ================

void moveWings() {
  for(int i = 0; i < NUM_FLAPS; i++) {
    leftServo.write(90);
    rightServo.write(90);
    delay(1000);
    leftServo.write(0);
    rightServo.write(180);
    delay(500);
  }
}


void runTaser() {
  digitalWrite(TASER_PIN, HIGH);
  delay(1000);
  digitalWrite(TASER_PIN, LOW);
}
  
 // ===========================================



 // ============ DRIVE FUNCTIONS ==============


void driveForward() {
    // FL clockwise
  digitalWrite(motor11, HIGH);
  digitalWrite(motor12, LOW);
    // FR counter
  digitalWrite(motor21, LOW);
  digitalWrite(motor22, HIGH);
    // BL clockwise
  digitalWrite(motor31, HIGH);
  digitalWrite(motor32, LOW);
    // BR counter
  digitalWrite(motor41, LOW);
  digitalWrite(motor42, HIGH);
}

void driveBackward() {
    // FL counter
  digitalWrite(motor11, LOW);
  digitalWrite(motor12, HIGH);
    // FR clockwise
  digitalWrite(motor21, HIGH);
  digitalWrite(motor22, LOW);
    // BL counter
  digitalWrite(motor31, LOW);
  digitalWrite(motor32, HIGH);
    // BR clockwise
  digitalWrite(motor41, HIGH);
  digitalWrite(motor42, LOW);
}

void driveIdle() {
  digitalWrite(motor11, HIGH);
  digitalWrite(motor12, HIGH);
  digitalWrite(motor21, HIGH);
  digitalWrite(motor22, HIGH);
  digitalWrite(motor31, HIGH);
  digitalWrite(motor32, HIGH);
  digitalWrite(motor41, HIGH);
  digitalWrite(motor42, HIGH);
}


void driveLeft() {
  // FL counter
  digitalWrite(motor11, LOW);
  digitalWrite(motor12, HIGH);
  // FR counter
  digitalWrite(motor21, LOW);
  digitalWrite(motor22, HIGH);
  // BL counter
  digitalWrite(motor31, LOW);
  digitalWrite(motor32, HIGH);
  // BR counter
  digitalWrite(motor41, LOW);
  digitalWrite(motor42, HIGH);
}


void driveRight() {
  // FL clock
  digitalWrite(motor11, HIGH);
  digitalWrite(motor12, LOW);
  // FR clock
  digitalWrite(motor21, HIGH);
  digitalWrite(motor22, LOW);
  // BL clock
  digitalWrite(motor31, HIGH);
  digitalWrite(motor32, LOW);
  // BR clock
  digitalWrite(motor41, HIGH);
  digitalWrite(motor42, LOW);
}


 // ===========================================



 // ================= RUNTIME =================

void setup() {
      // Initialize Serial
  Serial.begin(115200);
  Serial.println("Initialized serial");

    // Initialize all motor pins
  pinMode(motor11, OUTPUT);
  pinMode(motor12, OUTPUT);
  pinMode(motor21, OUTPUT);
  pinMode(motor22, OUTPUT);
  pinMode(motor31, OUTPUT);
  pinMode(motor32, OUTPUT);
  pinMode(motor41, OUTPUT);
  pinMode(motor42, OUTPUT);

  driveIdle();
  Serial.println("Initialized pins");

    // Initialize servos
  leftServo.attach(LEFT_SERVO_PIN);
  rightServo.attach(RIGHT_SERVO_PIN);

    // Intialize taser
  pinMode(TASER_PIN, OUTPUT);

    // Initialize bluetooth
  initBluetooth();
  Serial.println("Initialized bluetooth");


  // Serial.println("\n");
};


void loop() {
    // Wait until command is updated to something
  if (!command.equals("none")) {
    Serial.println("Command: " + command);

    if (command == "forward") {
      driveForward();
    } else if (command == "backward") {
      driveBackward();
    } else if (command == "idle") {
      driveIdle();
    } else if (command == "left") {
      driveLeft();
    } else if (command == "right") {
      driveRight();
    } else if (command == "wings") {
      moveWings();
    } else if (command == "taser") {
      runTaser();
    }
    command = "none"; // Reset back to none when processed
  }
  
  delay(100); // buffer between checks
};

