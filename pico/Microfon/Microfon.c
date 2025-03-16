#include <stdio.h>
#include "pico/stdlib.h"
#include "hardware/dma.h"
#include "pico/cyw43_arch.h"
#include "hardware/gpio.h"
#include "hardware/adc.h"

#define MARIME_BUFFER 1000
uint16_t buffer_adc[MARIME_BUFFER];

int main()
{
    stdio_init_all();

    // Initialise the Wi-Fi chip
    if (cyw43_arch_init()) {
        printf("Wi-Fi init failed\n");
        return -1;
    }

    // Example to turn on the Pico W LED
    cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 1);

    adc_init();
    adc_select_input(2);

    int index = 0;
    bool ok = 0;
    while (true) {
        cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, ok);
        ok = 1 - ok;
        buffer_adc[index] = adc_read();
        sleep_us(40);
        index++;
        if(index == MARIME_BUFFER){
            index = 0;
            for(int i = 0; i < MARIME_BUFFER; i++){
                printf("%d ", buffer_adc[i]);
            }
            printf("\n");
            //sleep_ms(250);
        }
    }
}
