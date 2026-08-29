#!/usr/bin/env zsh
# h3_euler.sh — M4 flow-matching Euler driver over the packed denoiser.
#
# Contract (pinned from comfy/ldm/minimax + ModelSamplingAV):
#   timestep sigma 1000 -> 0 over N steps (dt = 1/N in sigma space);
#   t_v = 1 - sigma_v (video/text rows), sigma_a = shift_map(sigma_v, 12->3)
#   with t_a = 1 - sigma_a (audio rows);
#   the model returns -v in x0 space, so the Euler update against the
#   flow ODE (noise -> data as sigma falls) is  x <- x + dt * out.
#
# State: video latent file only in this spike (audio tracked the same way
# once its velocity path is exercised); H3_X_IN/H3_X_OUT hand the video
# segment to h3_forward, which also replays the text/audio conditioning
# per step (text_states dump is read every step).
#
# Usage: h3_euler.sh <denoiser.gguf> <text_states.bin> <n_steps> <vt> <lat_h> <lat_w> <audio_t>
set -euo pipefail
GGUF="$1"; TE="$2"; N="${3:-4}"; VT="${4:-1}"; LH="${5:-32}"; LW="${6:-32}"; AT="${7:-2}"

X=/tmp/h3_x_current.bin
python3 - "$VT" "$LH" "$LW" <<'PY'
import struct, sys
vt, lh, lw = map(int, sys.argv[1:4])
ln = 24 * vt * lh * lw
# deterministic pure noise in [-1, 1) — x at sigma=1
state = 12345
def rnd():
    global state
    state = (1103515245 * state + 12345) & 0x7FFFFFFF
    return state / 0x40000000 - 1.0
with open("/tmp/h3_x_current.bin", "wb") as f:
    f.write(struct.pack("<%df" % ln, *(rnd() for _ in range(ln))))
print(f"x0 noise: {ln} floats", file=sys.stderr)
PY

i=0
while [ "$i" -lt "$N" ]; do
  SIGMA=$(python3 -c "print(1000.0 * (1.0 - ($i + 0.5) / $N))")
  export H3_SIGMA_V=$(python3 -c "print($SIGMA / 1000.0)")
  export H3_X_IN="$X"
  export H3_X_OUT=/tmp/h3_x_next_v.bin
  ./h3_forward "$GGUF" "$TE" "$VT" "$LH" "$LW" "$AT" > /tmp/h3_step.out 2>>/tmp/h3_euler.log
  # velocity v = -out; x_{i+1} = x_i + dt * (-out)?? model returns -v so
  # x + dt*(-v)... flow ODE: dx/dsigma = -v => x_{i+1} = x_i - dt*out
  python3 - "$X" /tmp/h3_x_next_v.bin "$N" <<'PY'
import struct, sys
xp, vp, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
x = struct.unpack("<%df" % (24 * 1024 * 1024 // 4), b"") if False else None
xf = open(xp, "rb").read()
vf = open(vp, "rb").read()
m = min(len(xf), len(vf))
cnt = m // 4
xs = struct.unpack("<%df" % cnt, xf[:m])
vs = struct.unpack("<%df" % cnt, vf[:m])
dt = 1.0 / n
out = struct.pack("<%df" % cnt, *(xs[k] - dt * vs[k] for k in range(cnt)))
open(xp, "wb").write(out)
print(f"step done dt={dt}", file=sys.stderr)
PY
  i=$((i + 1))
  echo "=== step $i/$N (sigma=$SIGMA) ==="
done
cp "$X" /tmp/h3_x_final.bin
echo "final latent: /tmp/h3_x_final.bin"
