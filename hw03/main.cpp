#include "mbed.h"
#include "mbed_rpc.h"
#include "uLCD_4DGL.h"
#include "MQTTNetwork.h"
#include "MQTTmbed.h"
#include "MQTTClient.h"
#include "accelerometer_handler.h"
#include "config.h"
#include "magic_wand_model_data.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "stm32l475e_iot01_accelero.h"
//#​include <math.h>

WiFiInterface *wifi;
InterruptIn btn2(USER_BUTTON);
volatile int message_num = 0;
volatile int arrivedcount = 0;
volatile bool closed = false;
const char* topic = "Mbed";
DigitalOut led1(LED1);  // gesture_UI
DigitalOut led2(LED2);  // tiltangle
DigitalOut led3(LED3);

Thread mqtt_thread(osPriorityHigh);
EventQueue mqtt_queue;

constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
uLCD_4DGL uLCD(D1, D0, D2);

BufferedSerial pc(USBTX, USBRX);
EventQueue queue(32 * EVENTS_EVENT_SIZE);
EventQueue queue1(32 * EVENTS_EVENT_SIZE);
EventQueue queue2(32 * EVENTS_EVENT_SIZE);
EventQueue queue3(32 * EVENTS_EVENT_SIZE);
void gesture_UI(Arguments *in, Reply *out);
RPCFunction gestureui(&gesture_UI, "gesture_UI");
void tiltangle(Arguments *in, Reply *out);
RPCFunction rpcLED(&tiltangle, "tiltangle");

int set_confirm = 1;
int gesture_index;
int select_angle;
int mode = 0;
int success_ang; // bigger than angle
//uLCD
void print(int gesture_index) {
    uLCD.cls();
    uLCD.background_color(WHITE);
    uLCD.color(BLUE);
    uLCD.text_width(4); //4X size text
    uLCD.text_height(4);
    uLCD.textbackground_color(WHITE);

    if (gesture_index == 0) 
      uLCD.printf("\nring:30\n");
    else if (gesture_index == 1) 
      uLCD.printf("\nslope:45\n");
    else if (gesture_index == 2) 
      uLCD.printf("\nline:60\n");
    else uLCD.printf("\nline\n");
}

void print1(int success_ang) {
    uLCD.cls();
    uLCD.background_color(WHITE);
    uLCD.color(BLUE);
    uLCD.text_width(2); //4X size text
    uLCD.text_height(2);
    uLCD.textbackground_color(WHITE);
    if (success_ang == 1)
     uLCD.printf("\nbigger than selected threshold angle\n");
    else
    uLCD.printf("\nsmaller than selected threshold angle\n");
}
void messageArrived(MQTT::MessageData& md) {
    MQTT::Message &message = md.message;
    char msg[300];
    sprintf(msg, "Message arrived: QoS%d, retained %d, dup %d, packetID %d\r\n", message.qos, message.retained, message.dup, message.id);
    printf(msg);
    ThisThread::sleep_for(1000ms);

    char payload[300];
    sprintf(payload, "Payload %.*s\r\n", message.payloadlen, (char*)message.payload);
    printf(payload);
    ++arrivedcount;
}

void publish_message(MQTT::Client<MQTTNetwork, Countdown>* client) {
    message_num++;
    MQTT::Message message;
    set_confirm = 0;
    //printf("tiltangle=%d\n\r", tangle);
    char buff[100];
    if(mode = 0)
    sprintf(buff, "tiltangle=%d\n\r #%d", select_angle, message_num);
    else 
    sprintf(buff, "tiltangle=%d\n\r #%d", select_angle, message_num);
    message.qos = MQTT::QOS0;
    message.retained = false;
    message.dup = false;
    message.payload = (void*) buff;
    message.payloadlen = strlen(buff) + 1;
    int rc = client->publish(topic, message);
    printf("rc:  %d\r\n", rc);
    printf("Publish message: %s\r\n", buff);
}

void close_mqtt() {
    closed = true;
}

/*void confirm_ges() {
  int tangle;
  set_confirm = 0;
  if (gesture_index == 0) tangle = 30;
  else if (gesture_index == 1) tangle = 45;
  else if (gesture_index == 2) tangle = 60;
  else tangle = 30;
  printf("tiltangle=%d\n\r", tangle);
  mqtt_queue.event(&publish_message, &client);
}*/

int PredictGesture(float* output) {
  static int continuous_count = 0;
  static int last_predict = -1;

  int this_predict = -1;
  for (int i = 0; i < label_num; i++) {
    if (output[i] > 0.8) this_predict = i;
  }

  if (this_predict == -1) {
    continuous_count = 0;
    last_predict = label_num;
    return label_num;
  }

  if (last_predict == this_predict) {
    continuous_count += 1;
  } else {
    continuous_count = 0;
  }
  last_predict = this_predict;
  if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {
    return label_num;
  }
  continuous_count = 0;
  last_predict = -1;
  return this_predict;
}

void detectges() {
  bool should_clear_buffer = false;
  bool got_data = false;
  //int gesture_index;
  led1 = !led1;
  mode = 0;
  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    //return -1;
  }

  static tflite::MicroOpResolver<6> micro_op_resolver;
  micro_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                               tflite::ops::micro::Register_MAX_POOL_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                               tflite::ops::micro::Register_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                               tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                               tflite::ops::micro::Register_SOFTMAX());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                               tflite::ops::micro::Register_RESHAPE(), 1);
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  tflite::MicroInterpreter* interpreter = &static_interpreter;
  interpreter->AllocateTensors();

  TfLiteTensor* model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != config.seq_length) ||
      (model_input->dims->data[2] != kChannelNumber) ||
      (model_input->type != kTfLiteFloat32)) {
    error_reporter->Report("Bad input tensor parameters in model");
    //return -1;
  }

  int input_length = model_input->bytes / sizeof(float);

  TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
  if (setup_status != kTfLiteOk) {
    error_reporter->Report("Set up failed\n");
    //return -1;
  }
  error_reporter->Report("Set up successful...\n");

  while (set_confirm) {
    got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                                 input_length, should_clear_buffer);

    if (!got_data) {
      should_clear_buffer = false;
      continue;
    }

    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed on index: %d\n", begin_index);
      continue;
    }

    gesture_index = PredictGesture(interpreter->output(0)->data.f);

    should_clear_buffer = gesture_index < label_num;

    if (gesture_index < label_num) {
      error_reporter->Report(config.output_message[gesture_index]);
      print(gesture_index);
      if (gesture_index == 0) select_angle = 30;
      else if (gesture_index == 1) select_angle = 45;
      else if (gesture_index == 2) select_angle = 60;
      else select_angle = 0;
    }
    
  }
}

void detectang() {
  led2 = !led2;
  mode = 1;
  int i;
  int16_t pDataXYZ[3] = {0};  // for initialize
  float cosangle;
  float long1;
  float long2;
  float cos_select;
  if (select_angle = 30) cos_select = 0.866;
  else if (select_angle = 45) cos_select = 0.707;
  else if (select_angle = 60) cos_select = 0.5;
  else cos_select = 0.5;

  BSP_ACCELERO_AccGetXYZ(pDataXYZ);
  printf("initial Accelerometer values: (%d, %d, %d)", pDataXYZ[0], pDataXYZ[1], pDataXYZ[2]);
  ThisThread::sleep_for(1000ms);
  int16_t pDataXYZ1[3] = {0};
  /*for (i = 0; i < 10; i++) {
    BSP_ACCELERO_AccGetXYZ(pDataXYZ1);
    printf("detect Accelerometer values: (%d, %d, %d)\r\n", pDataXYZ1[0], pDataXYZ1[1], pDataXYZ1[2]);
    long1 = sqrt(pDataXYZ[0] * pDataXYZ[0] + pDataXYZ[1] * pDataXYZ[1] + pDataXYZ[2] * pDataXYZ[2]);
    long2 = sqrt(pDataXYZ1[0] * pDataXYZ1[0] + pDataXYZ1[1] * pDataXYZ1[1] + pDataXYZ1[2] * pDataXYZ1[2]);
    cosangle = (pDataXYZ[0] * pDataXYZ1[0] + pDataXYZ[1] * pDataXYZ1[1] + pDataXYZ[2] * pDataXYZ1[2]) / (long1 * long2);
    if (cosangle < cos_select) success_ang = 1;
    else success_ang = 0; 
    printf("success_ang = %d\r\n", success_ang);
    print1(success_ang);
    ThisThread::sleep_for(1000ms);
  } */
  while(1) {
    BSP_ACCELERO_AccGetXYZ(pDataXYZ1);
    printf("detect Accelerometer values: (%d, %d, %d)\r\n", pDataXYZ1[0], pDataXYZ1[1], pDataXYZ1[2]);
    long1 = sqrt(pDataXYZ[0] * pDataXYZ[0] + pDataXYZ[1] * pDataXYZ[1] + pDataXYZ[2] * pDataXYZ[2]);
    long2 = sqrt(pDataXYZ1[0] * pDataXYZ1[0] + pDataXYZ1[1] * pDataXYZ1[1] + pDataXYZ1[2] * pDataXYZ1[2]);
    cosangle = (pDataXYZ[0] * pDataXYZ1[0] + pDataXYZ[1] * pDataXYZ1[1] + pDataXYZ[2] * pDataXYZ1[2]) / (long1 * long2);
    if (cosangle < cos_select) success_ang = 1;
    else success_ang = 0; 
    printf("success_ang = %d\r\n", success_ang);
    print1(success_ang);
    i++;
    ThisThread::sleep_for(1000ms);
    if (i>10){break};
  }
}

//double x, y;
Thread t1;
Thread t2;
Thread t3; // for buttom confirm

int main() {

    t3.start(callback(&queue3, &EventQueue::dispatch_forever));
    //btn2.rise(queue3.event(confirm_ges));
    //mqtt_queue.event(&publish_message, &client);
    //btn2.rise(mqtt_queue.event(&publish_message, &client));
    char buf[256], outbuf[256];
    FILE *devin = fdopen(&pc, "r");
    FILE *devout = fdopen(&pc, "w");
    BSP_ACCELERO_Init();

    wifi = WiFiInterface::get_default_instance();
    if (!wifi) {
      printf("ERROR: No WiFiInterface found.\r\n");
      //return -1;
    }
    printf("\nConnecting to %s...\r\n", MBED_CONF_APP_WIFI_SSID);
    int ret = wifi->connect(MBED_CONF_APP_WIFI_SSID, MBED_CONF_APP_WIFI_PASSWORD, NSAPI_SECURITY_WPA_WPA2);
    if (ret != 0) {
      printf("\nConnection error: %d\r\n", ret);
      //return -1;
    }
    NetworkInterface* net = wifi;
    MQTTNetwork mqttNetwork(net);
    MQTT::Client<MQTTNetwork, Countdown> client(mqttNetwork);
    const char* host = "192.168.43.219";
    printf("Connecting to TCP network...\r\n");
    SocketAddress sockAddr;
    sockAddr.set_ip_address(host);
    sockAddr.set_port(1883);
    printf("address is %s/%d\r\n", (sockAddr.get_ip_address() ? sockAddr.get_ip_address() : "None"),  (sockAddr.get_port() ? sockAddr.get_port() : 0) ); //check setting
    int rc = mqttNetwork.connect(sockAddr);//(host, 1883);
    if (rc != 0) {
      printf("Connection error.");
      //return -1;
    }
    printf("Successfully connected!\r\n");
    MQTTPacket_connectData data = MQTTPacket_connectData_initializer;
    data.MQTTVersion = 3;
    data.clientID.cstring = "Mbed";
    if ((rc = client.connect(data)) != 0){
            printf("Fail to connect MQTT\r\n");
    }
    if (client.subscribe(topic, MQTT::QOS0, messageArrived) != 0){
            printf("Fail to subscribe\r\n");
    }    
    mqtt_thread.start(callback(&mqtt_queue, &EventQueue::dispatch_forever));
    /*if (set_confirm == 0) {
      mqtt_queue.event(&publish_message, &client);
    }*/
    btn2.rise(mqtt_queue.event(&publish_message, &client));
    
    while(1) {
      memset(buf, 0, 256);
      for (int i = 0; ; i++) {
          char recv = fgetc(devin);
          if (recv == '\n') {
              printf("\r\n");
              break;
          }
          buf[i] = fputc(recv, devout);
      }
    RPC::call(buf, outbuf);
    printf("%s\r\n", outbuf);
    }
    int num = 0;
    while (num != 5) {
            client.yield(100);
            ++num;
    }

    while (1) {
            if (closed) break;
            client.yield(500);
            ThisThread::sleep_for(500ms);
    }
    printf("Ready to close MQTT Network......\n");

    if ((rc = client.unsubscribe(topic)) != 0) {
            printf("Failed: rc from unsubscribe was %d\n", rc);
    }
    if ((rc = client.disconnect()) != 0) {
    printf("Failed: rc from disconnect was %d\n", rc);
    }

    mqttNetwork.disconnect();
    printf("Successfully closed!\n");

    return 0;
}

void gesture_UI(Arguments *in, Reply *out) {
    t1.start(callback(&queue1, &EventQueue::dispatch_forever));
    queue1.call(detectges);

}
void tiltangle(Arguments *in, Reply *out) {
    t2.start(callback(&queue2, &EventQueue::dispatch_forever));
    queue2.call(detectang);
}