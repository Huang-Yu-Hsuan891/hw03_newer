#include "mbed.h"
#include "mbed_rpc.h"
#include "uLCD_4DGL.h"

uLCD_4DGL uLCD(D1, D0, D2);
BufferedSerial pc(USBTX, USBRX);
EventQueue queue1(32 * EVENTS_EVENT_SIZE);
//EventQueue queue2(32 * EVENTS_EVENT_SIZE);
DigitalOut myled(LED1);

void gesture_UI(Arguments *in, Reply *out);
RPCFunction gestureui(&gesture_UI, "gesture_UI");

//void tiltangle(Arguments *in, Reply *out);
//RPCFunction rpcLED(&tiltangle, "tiltangle");
void detectges() {
    //myled = 1;
    printf("test\r\n");
}

//double x, y;
Thread t1;
//Thread t2;

int main() {

    
    //t2.start(callback(&queue2, &EventQueue::dispatch_forever));

    char buf[256], outbuf[256];
    FILE *devin = fdopen(&pc, "r");
    FILE *devout = fdopen(&pc, "w");
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

}

void gesture_UI(Arguments *in, Reply *out) {
    t1.start(callback(&queue1, &EventQueue::dispatch_forever));
    queue1.call(&detectges);
    //char outbuf[256];
    //char buffer[200];
}
//void tiltangle(Arguments *in, Reply *out);