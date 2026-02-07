


  // CONSTANTS and structures
struct MotorPins {
  uint8_t IN1;
  uint8_t IN2;
  uint8_t IN3;
  uint8_t IN4;
};

MotorPins motors[4] = { // tl tr bl br switch order
  {32,33,25,26}, // FL
  {23,22,1,3}, // FR
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
  IDLE
};





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
      digitalWrite(m.IN1, HIGH);
      digitalWrite(m.IN2, HIGH);
      digitalWrite(m.IN3, HIGH);
      digitalWrite(m.IN4, HIGH);
      break;

    default:
      digitalWrite(m.IN1, LOW);
      digitalWrite(m.IN2, LOW);
      digitalWrite(m.IN3, LOW);
      digitalWrite(m.IN4, LOW);
      break;
    };
  }

  





void setup() {
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
};


void loop() {
    // Drive forward 30s
  driveWheel(FRONT_LEFT, IDLE);
  driveWheel(FRONT_RIGHT, IDLE);
  driveWheel(BACK_LEFT, IDLE);
  driveWheel(BACK_RIGHT, IDLE);
  // delay(30000);
  //   // Drive backward 30s
  // driveWheel(FRONT_LEFT, BACKWARD);
  // driveWheel(FRONT_RIGHT, BACKWARD);
  // driveWheel(BACK_LEFT, BACKWARD);
  // driveWheel(BACK_RIGHT, BACKWARD);
  // delay(30000);
  //   // Idle 30s
  // driveWheel(FRONT_LEFT, IDLE);
  // driveWheel(FRONT_RIGHT, IDLE);
  // driveWheel(BACK_LEFT, IDLE);
  // driveWheel(BACK_RIGHT, IDLE);
  // delay(30000);
};

