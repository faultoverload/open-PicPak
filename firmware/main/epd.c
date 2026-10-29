#include "epd.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/spi_master.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "nvs.h"
#include <string.h>

static const char *TAG = "epd";

/* Pin mapping (verified from PicPak RE) */
#define PIN_SCLK 6
#define PIN_MOSI 3
#define PIN_CS   9
#define PIN_DC   8
#define PIN_RST  10
#define PIN_BUSY 20

/* BUSY: LOW = busy, HIGH = idle (GxEPD2 busy_level=LOW for this panel) */
#define BUSY_IDLE_LEVEL 1

static spi_device_handle_t s_spi;
static bool s_init_done;

static void epd_cmd(uint8_t c)
{
    gpio_set_level(PIN_DC, 0);
    spi_transaction_t t = { .length = 8, .tx_buffer = &c };
    spi_device_polling_transmit(s_spi, &t);
}

static void epd_data1(uint8_t d)
{
    gpio_set_level(PIN_DC, 1);
    spi_transaction_t t = { .length = 8, .tx_buffer = &d };
    spi_device_polling_transmit(s_spi, &t);
}

static void epd_data(const uint8_t *buf, size_t len)
{
    gpio_set_level(PIN_DC, 1);
    spi_transaction_t t = { .length = 8 * len, .tx_buffer = buf };
    spi_device_polling_transmit(s_spi, &t);
}

static void wait_busy(const char *tag, int timeout_ms)
{
    int waited = 0;
    while (gpio_get_level(PIN_BUSY) != BUSY_IDLE_LEVEL) {
        vTaskDelay(pdMS_TO_TICKS(10));
        waited += 10;
        if (waited >= timeout_ms) {
            ESP_LOGW(TAG, "%s: BUSY timeout after %d ms (level=%d)",
                     tag, timeout_ms, gpio_get_level(PIN_BUSY));
            return;
        }
    }
    ESP_LOGI(TAG, "%s: idle after %d ms", tag, waited);
}

static void epd_reset(void)
{
    gpio_set_level(PIN_RST, 1); vTaskDelay(pdMS_TO_TICKS(20));
    /* 20 ms low: a 5 ms pulse does not wake the controller from its deep sleep (0x07) --
     * BUSY then stays low and every step times out. Only a fresh boot, whose own reset holds
     * the line low far longer, used to recover it. */
    gpio_set_level(PIN_RST, 0); vTaskDelay(pdMS_TO_TICKS(20));
    gpio_set_level(PIN_RST, 1); vTaskDelay(pdMS_TO_TICKS(20));
    wait_busy("reset", 1000);
}

/* RAM window area (cmd 0x83) */
static void epd_set_ram_area(uint16_t x, uint16_t y, uint16_t w, uint16_t h)
{
    uint16_t xe = x + w - 1, ye = y + h - 1;
    uint8_t d[9] = { x >> 8, x & 0xFF, xe >> 8, xe & 0xFF,
                     y >> 8, y & 0xFF, ye >> 8, ye & 0xFF, 0x00 /* full mode */ };
    epd_cmd(0x83);
    epd_data(d, sizeof(d));
}

/* Full panel init (A/B wave, default-off): the complete register set for the panel's native
 * (OTP) waveform, incl. the registers the short sequence below leaves at reset defaults --
 * PFS, TSE (internal temperature sensor -> OTP waveform selection), PWS -- plus the vendor
 * register page, closed again by the second 0xFF write. Row format { cmd, n, data... };
 * 0xFF is a valid command here, so iterate by size, never by sentinel. */
static const uint8_t EPD_INIT_FULL[] = {
    0x00, 2, 0x07, 0x29,                                         /* PSR */
    0x01, 2, 0x07, 0x00,                                         /* PWRR */
    0x03, 3, 0x10, 0x54, 0x44,                                   /* PFS */
    0x06, 3, 0xC0, 0xC0, 0xC0,                                   /* BTST */
    0x30, 1, 0x08,                                               /* PLL */
    0x41, 1, 0x00,                                               /* TSE: internal sensor */
    0x50, 1, 0x37,                                               /* CDI */
    0x61, 4, EPD_W >> 8, EPD_W & 0xFF, EPD_H >> 8, EPD_H & 0xFF, /* TRES */
    0x65, 4, 0x00, 0x00, 0x00, 0x00,                             /* GSST */
    0xE3, 1, 0x22,                                               /* PWS */
    0xE7, 1, 0x1C,
    0xE9, 1, 0x01,
    0xFF, 1, 0xA5,                                               /* vendor page open */
    0xEF, 8, 0x01, 0x32, 0x08, 0x32, 0x0A, 0x32, 0x0F, 0x19,
    0xFD, 1, 0x01,
    0xE8, 1, 0x00,
    0xDF, 1, 0x3C,
    0xDC, 1, 0x00,
    0xDD, 1, 0x01,
    0xDE, 1, 0x14,
    0xFF, 1, 0xE3,                                               /* vendor page close */
};

/* Policy = data: NVS "epd"/"init" == "full" selects EPD_INIT_FULL (NVSSET epd init full);
 * anything else (incl. absent) keeps the short factory sequence. */
static bool full_init_selected(void)
{
    nvs_handle_t h; char v[8] = {0}; size_t n = sizeof(v); bool on = false;
    if (nvs_open("epd", NVS_READONLY, &h) == ESP_OK) {
        if (nvs_get_str(h, "init", v, &n) == ESP_OK) on = (strcmp(v, "full") == 0);
        nvs_close(h);
    }
    return on;
}

static void epd_send_seq(const uint8_t *seq, size_t len)
{
    for (size_t i = 0; i + 1 < len; ) {
        uint8_t cmd = seq[i++], n = seq[i++];
        epd_cmd(cmd);
        if (n) epd_data(&seq[i], n);
        i += n;
    }
}

/* Panel init sequence for the 0x060401 panel. The register map (PSR/PWRR/BTST/PLL/CDI/TRES/
 * GSST/...) is the public UC81xx-class command set. The specific VALUES are the MANUFACTURER'S
 * FACTORY CALIBRATION for this panel -- empirically tuned by the vendor for these exact display
 * characteristics; they are not arbitrary, and reproducing them is what makes the panel work
 * correctly. The panel is selected by its own hardware ID, so this is a property of the panel
 * hardware (the configuration it requires to operate), not a firmware-version secret; carried
 * here as interoperability information.
 * Deliberately NOT set (the panel keeps its OTP/reset defaults): 0x03 POFS, 0x41 TSE,
 * 0x60 TCON, 0x82 VDCS (VCOM = OTP default), 0xE3 PWS, 0xE0. */
static void epd_init_panel(void)
{
    epd_reset();
    if (full_init_selected()) {
        epd_send_seq(EPD_INIT_FULL, sizeof(EPD_INIT_FULL));
        epd_cmd(0x04);
        wait_busy("power_on", 2000);
        ESP_LOGI(TAG, "panel init (FULL seq) + power on done");
        return;
    }
    epd_cmd(0x00); { uint8_t d[] = {0x07, 0x29}; epd_data(d, 2); }                          /* PSR */
    epd_cmd(0x01); { uint8_t d[] = {0x07, 0x00}; epd_data(d, 2); }                          /* PWRR */
    epd_cmd(0x06); { uint8_t d[] = {0x0F, 0x8B, 0x9C, 0x96}; epd_data(d, 4); }              /* BTST */
    epd_cmd(0x30); epd_data1(0x08);                                                         /* PLL */
    epd_cmd(0x50); epd_data1(0x37);                                                         /* CDI */
    epd_cmd(0x61); { uint8_t d[] = {EPD_W >> 8, EPD_W & 0xFF, EPD_H >> 8, EPD_H & 0xFF}; epd_data(d, 4); } /* TRES 400x300 = 01,90,01,2C */
    epd_cmd(0x65); { uint8_t d[] = {0x00, 0x00, 0x00, 0x00}; epd_data(d, 4); }              /* GSST */
    epd_cmd(0xE7); epd_data1(0x96);
    epd_cmd(0xE9); epd_data1(0x01);
    epd_cmd(0xFF); epd_data1(0xA5);                                                         /* Vendor-specific (undocumented) */
    epd_cmd(0x04);                                                                          /* PowerOn (stock keeps it in the refresh helper; retained here, tested) */
    wait_busy("power_on", 2000);
    ESP_LOGI(TAG, "panel init (0x060401 factory seq) + power on done");
}

void epd_init(void)
{
    /* Idempotent: the GPIO + SPI bus are set up ONCE (spi_bus_initialize aborts via
     * ESP_ERROR_CHECK on a second call -> panic). A repeated epd_init (keep-awake loop,
     * or any multi-refresh-without-reboot) only re-inits the PANEL, which is needed anyway
     * to wake it from the deep sleep that epd_sleep() left it in. */
    if (s_init_done) { epd_init_panel(); return; }

    gpio_config_t out = {
        .pin_bit_mask = (1ULL << PIN_DC) | (1ULL << PIN_RST),
        .mode = GPIO_MODE_OUTPUT,
    };
    gpio_config(&out);
    gpio_config_t in = { .pin_bit_mask = (1ULL << PIN_BUSY), .mode = GPIO_MODE_INPUT };
    gpio_config(&in);

    spi_bus_config_t bus = {
        .mosi_io_num = PIN_MOSI,
        .sclk_io_num = PIN_SCLK,
        .miso_io_num = -1,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = EPD_FRAME_BYTES + 16,
    };
    ESP_ERROR_CHECK(spi_bus_initialize(SPI2_HOST, &bus, SPI_DMA_CH_AUTO));
    spi_device_interface_config_t dev = {
        .clock_speed_hz = 4 * 1000 * 1000,   /* 4 MHz; lower on instability */
        .mode = 0,
        .spics_io_num = PIN_CS,
        .queue_size = 1,
    };
    ESP_ERROR_CHECK(spi_bus_add_device(SPI2_HOST, &dev, &s_spi));

    epd_init_panel();
    s_init_done = true;
}

void epd_write_full(const uint8_t *buf)
{
    if (!s_init_done) epd_init();
    epd_set_ram_area(0, 0, EPD_W, EPD_H);
    epd_cmd(0x10);
    /* Plain linear block transfer -- the framebuffer arrives panel-ready. The panel's gate
     * scan (factory PSR 0x07,0x29) is Y-mirrored vs a top-to-bottom image, so the data is
     * pre-mirrored at the source instead of per-frame here: the baked setup screens are
     * generated vertically flipped, and the server renders the image flipped too. No
     * per-frame transform on the device. */
    epd_data(buf, EPD_FRAME_BYTES);
    ESP_LOGI(TAG, "frame written (%d B)", EPD_FRAME_BYTES);
}

/* Fill the whole panel RAM with one colour code (0=K 1=W 2=Y 3=R) without a 30 kB buffer. */
void epd_write_solid(uint8_t code)
{
    if (!s_init_done) epd_init();
    uint8_t chunk[1000];
    memset(chunk, (code & 3) * 0x55, sizeof(chunk));
    epd_set_ram_area(0, 0, EPD_W, EPD_H);
    epd_cmd(0x10);
    for (int off = 0; off < EPD_FRAME_BYTES; off += sizeof(chunk)) epd_data(chunk, sizeof(chunk));
}

/* De-ghost: drive every pixel through all four inks n times, ending on white, then sleep the
 * panel. Clears image retention left by a frame that stayed static for a long time. */
void epd_clear_cycles(int n)
{
    static const uint8_t seq[] = { 0, 1, 2, 3 };   /* K W Y R */
    for (int c = 0; c < n; c++)
        for (size_t i = 0; i < sizeof(seq); i++) {
            epd_init(); epd_write_solid(seq[i]); epd_refresh();
        }
    epd_init(); epd_write_solid(1); epd_refresh();
    epd_sleep();
}

void epd_refresh(void)
{
    epd_cmd(0x50); epd_data1(0x37);   /* CDI full */
    epd_cmd(0x12); epd_data1(0x00);   /* Display Refresh */
    vTaskDelay(pdMS_TO_TICKS(1));
    wait_busy("refresh", 35000);
    ESP_LOGI(TAG, "refresh done");
}

void epd_sleep(void)
{
    epd_cmd(0x02); epd_data1(0x00);   /* PowerOff */
    wait_busy("power_off", 2000);
    epd_cmd(0x07); epd_data1(0xA5);   /* Deep Sleep */
    ESP_LOGI(TAG, "panel deep sleep");
}
