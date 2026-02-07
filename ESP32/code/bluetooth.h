#include <NimBLEDevice.h>
#include <time.h>

static NimBLEServer* pServer;
static NimBLECharacteristic* pCharacteristic;
struct tm current_time = {0};

String command = "none";

class ServerCallbacks : public NimBLEServerCallbacks {
    void onConnect(NimBLEServer* pServer, NimBLEConnInfo& connInfo) override {
        pServer->updateConnParams(connInfo.getConnHandle(), 24, 48, 0, 180);
    }

    void onDisconnect(NimBLEServer* pServer, NimBLEConnInfo& connInfo, int reason) override {
        NimBLEDevice::startAdvertising();
    }

    void onMTUChange(uint16_t MTU, NimBLEConnInfo& connInfo) override {
        //Serial.printf("MTU updated: %u for connection ID: %u\n", MTU, connInfo.getConnHandle());
    }
};

class TimeCallback : public NimBLECharacteristicCallbacks {
    void onWrite(NimBLECharacteristic* pCharacteristic, NimBLEConnInfo& connInfo) override {

        String value = pCharacteristic->getValue();
        String response = "recieved";
        command = value; // Example input: "f f f f" where each char represents an action the user wants that repective wheel to take

        //do logic
        pCharacteristic->setValue(response.c_str());
        pCharacteristic->notify();
    }
};

void initBluetooth() {
    NimBLEDevice::init("T");
    NimBLEDevice::setPower(ESP_PWR_LVL_P9);

    pServer = NimBLEDevice::createServer();
    pServer->setCallbacks(new ServerCallbacks());
    NimBLEService* pService = pServer->createService("12345679-1234-1234-1234-1234567890ab");

    // Create a writable/readable characteristic with notify
    pCharacteristic = pService->createCharacteristic(
        "97654321-4321-4321-4321-ba0987654321",
        NIMBLE_PROPERTY::READ | NIMBLE_PROPERTY::WRITE | NIMBLE_PROPERTY::WRITE_NR | NIMBLE_PROPERTY::NOTIFY
    );

    pCharacteristic->setValue("90,90");
    pCharacteristic->setCallbacks(new TimeCallback());

    pService->start();

    NimBLEAdvertising* pAdvertising = NimBLEDevice::getAdvertising();
    pAdvertising->setName("JD");
    pAdvertising->addServiceUUID(pService->getUUID());
    pAdvertising->enableScanResponse(true);
    pAdvertising->start();
}
