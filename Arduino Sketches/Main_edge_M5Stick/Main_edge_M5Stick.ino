#include <M5StickCPlus.h>
#include <WiFi.h>
#include <PubSubClient.h>

// WiFi credentials
const char* ssid = "This Could be iPhone 2.0";
const char* password = "12345678";

// MQTT broker settings
const char* mqtt_server = "172.20.10.9"; // MQTT Server or where the broker is
const char* mqtt_topic = "imu_data";

WiFiClient espClient;
PubSubClient client(espClient);

void setup() {
  M5.begin();
  M5.IMU.Init();
  Serial.begin(9600);
  Serial.println("IMU Data");
  
  M5.Lcd.setRotation(3);
  setupWifi();
  client.setServer(mqtt_server,1883);

  M5.Lcd.setRotation(3);  // Rotate the screen. 将屏幕旋转
  M5.Lcd.fillScreen(BLACK);
  M5.Lcd.setTextSize(1);
  M5.Lcd.setCursor(80, 15);  // set the cursor location.  设置光标位置
  M5.Lcd.println("IMU TEST");
  M5.Lcd.setCursor(70, 30);
  M5.Lcd.println("  X       Y       Z");
  M5.Lcd.setCursor(30, 70);
  M5.Lcd.println("  Pitch   Roll    Yaw");
}

void setupWifi() {
    delay(10);
    M5.Lcd.printf("Connecting to %s", ssid);
    WiFi.mode(
        WIFI_STA);  // Set the mode to WiFi station mode.  设置模式为WIFI站模式
    WiFi.begin(ssid, password);  // Start Wifi connection.  开始wifi连接

    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        M5.Lcd.print(".");
    }
    M5.Lcd.printf("\nSuccess\n");
}



void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  float accX, accY, accZ;
  float gyroX, gyroY, gyroZ;
  float pitch, roll, yaw;

  // Read I2C data
  M5.IMU.getAccelData(&accX, &accY, &accZ);
  M5.IMU.getGyroData(&gyroX, &gyroY, &gyroZ);
  M5.IMU.getAhrsData(&pitch, &roll, &yaw);

  // Print the data to the serial monitor
  /*Serial.print("Accel: X=");
  Serial.print(accX);
  Serial.print(", Y=");
  Serial.print(accY);
  Serial.print(", Z=");
  Serial.print(accZ);
  Serial.print(" | Gyro: X=");
  Serial.print(gyroX);
  Serial.print(", Y=");
  Serial.print(gyroY);
  Serial.print(", Z=");
  Serial.println(gyroZ);*/
  M5.Lcd.setCursor(30, 40);
  M5.Lcd.printf("Gyro %6.2f  %6.2f  %6.2f      ", gyroX, gyroY, gyroZ);
  M5.Lcd.setCursor(170, 40);
  M5.Lcd.print("o/s");
  M5.Lcd.setCursor(30, 50);
  M5.Lcd.printf("Acce  %5.2f   %5.2f   %5.2f   ", accX, accY, accZ);
  M5.Lcd.setCursor(170, 50);
  M5.Lcd.print("G");
  M5.Lcd.setCursor(30, 80);
  M5.Lcd.printf(" %5.2f   %5.2f   %5.2f   ", pitch, roll, yaw);

  char payload[100];
  snprintf(payload, sizeof(payload), "Accel: X=%f, Y=%f, Z=%f Gyro: X=%f, Y=%f, Z=%f", accX, accY, accZ, gyroX, gyroY, gyroZ);

  // Serial.print("Payload data: ");
  // Serial.println(payload);
  // Publish the message and check if it was successful
  if (client.publish(mqtt_topic, payload)) {
    Serial.println("Message sent successfully");
  } else {
    Serial.println("Failed to send message");
  }
  delay(3.5);
}

void reconnect() {
  // Loop until we're reconnected
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    // Attempt to connect
    if (client.connect("M5StickClient")) {
      Serial.println("connected");
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      // Wait 5 seconds before retrying
      delay(5000);
    }
  }
}
