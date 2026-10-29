/* Host unit test for the button-cooldown pure logic (button_core.h).
 * Compile + run on the host (no ESP hardware). */
#include "button_core.h"
#include <stdio.h>

static int fails = 0;
#define CHECK(cond) do { if (!(cond)) { printf("  FAIL: %s\n", #cond); fails++; } } while (0)
#define CD 300u

int main(void)
{
    btn_decision_t d;

    /* 1. First press ever (no previous button refresh) -> refresh. */
    d = btn_on_press(1000, false, 0, CD, true, 4000, 3600);
    CHECK(!d.skip);

    /* 2. Press inside the cooldown -> skip, and sleep until the already-scheduled timer wake
     *    (the hourly cadence is kept, not restarted by the ignored press). */
    d = btn_on_press(1100, true, 1000, CD, true, 4600, 3600);
    CHECK(d.skip);
    CHECK(d.sleep_s == 3500);

    /* 3. Exactly at / after the cooldown -> refresh again. */
    d = btn_on_press(1300, true, 1000, CD, true, 4600, 3600); CHECK(!d.skip);
    d = btn_on_press(1299, true, 1000, CD, true, 4600, 3600); CHECK(d.skip);

    /* 4. Kids mashing: 1000 presses one second apart after a refresh -> only one refresh per
     *    cooldown window (the protective property). */
    {
        uint64_t last = 0; bool have = false; int refreshes = 0;
        for (uint64_t t = 10; t < 10 + 1000; t++) {
            d = btn_on_press(t, have, last, CD, true, 100000, 3600);
            if (!d.skip) { refreshes++; last = t; have = true; }
        }
        CHECK(refreshes == 4);   /* t=10, 310, 610, 910 */
    }

    /* 5. Skip with no / past scheduled timer wake -> fall back to the wake interval, never 0. */
    d = btn_on_press(1100, true, 1000, CD, false, 0, 3600);    CHECK(d.skip && d.sleep_s == 3600);
    d = btn_on_press(1100, true, 1000, CD, true, 1100, 3600);  CHECK(d.skip && d.sleep_s == 3600);
    d = btn_on_press(1100, true, 1000, CD, true, 1050, 3600);  CHECK(d.skip && d.sleep_s == 3600);

    /* 6. Clock went backwards (RTC reset between runs) -> not trusted, refresh. */
    d = btn_on_press(500, true, 1000, CD, true, 4000, 3600);
    CHECK(!d.skip);

    if (fails == 0) { printf("ALL BUTTON CORE TESTS PASSED\n"); return 0; }
    printf("%d CHECK(S) FAILED\n", fails);
    return 1;
}
