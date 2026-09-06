#!/bin/sh
# thermal_report.sh — one-line thermal-pressure status for perf logs.
#
# A throttled run silently changes tok/s (busy spin-wait threads power-couple
# laptops into clamping CPU frequency), so benchmark/gate scripts record this
# next to their config. `pmset -g therm` is the sudo-free macOS readout of
# recorded thermal/performance warnings; absence of a note means no warning
# has been recorded, not that the SoC is cool right now — pair it with a
# wall-clock sanity check when it matters.
out="$(pmset -g therm 2>/dev/null)"
if [ $? -ne 0 ] || [ -z "$out" ]; then
    echo "thermal: unknown (pmset unavailable)"
    exit 0
fi
if printf '%s\n' "$out" | grep -q "No thermal warning level"; then
    echo "thermal: ok (no macOS thermal/perf warnings recorded)"
else
    echo "thermal: WARNING — pressure recorded:"
    printf '%s\n' "$out" | sed 's/^/    /'
fi
exit 0
