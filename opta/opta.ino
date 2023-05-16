/* Edge Impulse ingestion SDK
   Copyright (c) 2022 EdgeImpulse Inc.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

*/

/* Includes ---------------------------------------------------------------- */
#include <Opta_Predictive_Maintenance_Expo_inferencing.h>

/* Private variables ------------------------------------------------------- */
static bool debug_nn = true;  // Set this to true to see e.g. features generated from the raw signal
int offAmperage;

float readAmperage() {
  int readingsSum = 0;
  int sensorValueA0;
  float AmperageA0;

  sensorValueA0 = analogRead(A0);
  
  // 1.2 is the factor used to map the measured voltage to the corresponding current
  AmperageA0 = (sensorValueA0 * (3.0 / 4095.0) / 0.3) * 1.2; 

  return AmperageA0;
}

/**
  @brief      Arduino setup function
*/
void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  // comment out the below line to cancel the wait for USB connection (needed for native USB)
  //  while (!Serial)
  //    ;
  pinMode(D0, OUTPUT);
  Serial.println("Edge Impulse Inferencing Demo");

  // 65535 is the max value with 16 bits resolution set by analogReadResolution(16)
  // 4095 is the max value with 12 bits resolution set by analogReadResolution(12)
  analogReadResolution(12);

  if (EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME != 1) {
    ei_printf("ERR: EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME should be equal to 1 (the 1 sensor axes)\n");
    return;
  }
}

/**
   @brief Return the sign of the number

   @param number
   @return int 1 if positive (or 0) -1 if negative
*/
float ei_get_sign(float number) {
  return (number >= 0.0) ? 1.0 : -1.0;
}

/**
  @brief      Get data and run inferencing

  @param[in]  debug  Get debug info if true
*/
void loop() {
  // ei_printf("\nStarting inferencing in 2 seconds...\n");

  delay(2000);

  // ei_printf("Sampling...\n");

  // Allocate a buffer here for the values we'll read from the IMU
  float buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE] = { 0 };

  for (size_t ix = 0; ix < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE; ix += 1) {
    // Determine the next tick (and then sleep later)
    uint64_t next_tick = micros() + (EI_CLASSIFIER_INTERVAL_MS * 1000);

    buffer[ix] = readAmperage() * 1000; // read current
    delayMicroseconds(next_tick - micros());
  }

  // Turn the raw buffer in a signal which we can the classify
  signal_t signal;
  int err = numpy::signal_from_buffer(buffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
  if (err != 0) {
    ei_printf("Failed to create signal from buffer (%d)\n", err);
    return;
  }

  // Run the classifier
  ei_impulse_result_t result = { 0 };

  err = run_classifier(&signal, &result, debug_nn);
  if (err != EI_IMPULSE_OK) {
    ei_printf("ERR: Failed to run classifier (%d)\n", err);
    return;
  }
  // print the predictions
  ei_printf("Predictions ");
  ei_printf("(DSP: %d ms., Classification: %d ms., Anomaly: %d ms.)",
            result.timing.dsp, result.timing.classification, result.timing.anomaly);
  ei_printf(": \n");
  for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
    ei_printf("    %s: %.5f\n", result.classification[ix].label, result.classification[ix].value);
  }
#if EI_CLASSIFIER_HAS_ANOMALY == 1
  ei_printf("    anomaly score: %.3f\n", result.anomaly);
  if (result.anomaly > 0.3)
    digitalWrite(D0, HIGH);
  else digitalWrite(D0, LOW);


#endif
}
#if !defined(EI_CLASSIFIER_SENSOR)
#error "Invalid model for current sensor"
#endif
