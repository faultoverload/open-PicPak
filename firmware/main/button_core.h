#ifndef BUTTON_CORE_H
#define BUTTON_CORE_H
/* Button-press cooldown pure decision logic — no hardware/NVS, host-unit-testable.
 * A press only ever means "pull a new picture"; after a button refresh, further presses
 * within the cooldown do nothing (the device goes straight back to sleep). */
#include <stdint.h>
#include <stdbool.h>

typedef struct {
    bool     skip;      /* inside the cooldown: no WiFi/EPD, go back to sleep */
    uint32_t sleep_s;   /* when skipping: seconds to sleep */
} btn_decision_t;

/* now_s / last_s / next_timer_s are seconds on the same RTC clock. have_last: a previous
 * button refresh is on record. have_next: the timer wake scheduled before this press. */
static inline btn_decision_t btn_on_press(uint64_t now_s, bool have_last, uint64_t last_s,
                                          uint32_t cooldown_s, bool have_next,
                                          uint64_t next_timer_s, uint32_t wake_s)
{
    btn_decision_t d = { false, 0 };
    /* now < last = the clock restarted -> the record is stale, don't trust it. */
    if (!have_last || now_s < last_s || now_s - last_s >= cooldown_s) return d;
    d.skip = true;
    /* Keep the regular cadence: sleep until the timer wake that was already scheduled. */
    d.sleep_s = (have_next && next_timer_s > now_s) ? (uint32_t)(next_timer_s - now_s) : wake_s;
    if (d.sleep_s == 0) d.sleep_s = wake_s ? wake_s : 1;
    return d;
}

#endif /* BUTTON_CORE_H */
