# Stratum native runtime build.
#
# Binaries:
#   stratum       — universal GGUF runtime (Llama-arch + qwen35-arch)
#   stratum_p35.c — historical Phase-7 source kept as archive only (not
#                   buildable: its dependency stratum_prefetch.h is gone)
#
# Usage:
#   make            # builds the universal binary (+ metallib)
#   make clean

CC      := clang
# -ffast-math: FP contraction is toolchain-dependent. Greedy argmax is pinned
# by the gate scripts, but logit bit patterns may shift across Xcode updates —
# re-run run_all_gates.sh after toolchain upgrades (AGENTS.md, "Determinism &
# exactness contract").
CFLAGS  := -O3 -ffast-math -DACCELERATE_NEW_LAPACK -Wno-deprecated-declarations \
           -fblocks -mcpu=apple-m1
LDFLAGS := -framework Accelerate -lpthread

.PHONY: all clean check tests verify deps

all: stratum $(METAL_LIB)

# MemX compressed-memory runtime — OPTIONAL external dependency
# (github.com/shiaho777/memx, MIT). Backs staging buffers and KV/SSM state.
# It is DOWNLOADED, never vendored: fetch_memx.sh clones/pulls the upstream
# repo into $(MEMX_HOME) (gitignored), and upstream updates flow in via
# `make deps`. The engine builds and runs without it:  make USE_MEMX=0
USE_MEMX ?= 1
MEMX_HOME ?= memx
MEMX_SRC   := $(MEMX_HOME)/libmemx3.m
MEMX_INC   := $(MEMX_HOME)/include/memx_runtime.h
MEMX_DYLIB := $(MEMX_HOME)/build/libmemx_runtime.dylib
ifeq ($(USE_MEMX),1)
  CFLAGS  += -DSTRATUM_USE_MEMX -I$(MEMX_HOME)/include
  LDFLAGS += -L$(MEMX_HOME)/build -lmemx_runtime -Wl,-rpath,$(abspath $(MEMX_HOME))/build
  MEMX_OBJ := $(MEMX_DYLIB)
else
  MEMX_OBJ :=
endif

# Fetch the dependency source (clone or pull upstream).
$(MEMX_SRC) $(MEMX_INC):
	@./fetch_memx.sh $(MEMX_HOME) || { \
	    echo "memx: fetch failed (offline?). Build without it:  make USE_MEMX=0" >&2; \
	    exit 1; \
	}

$(MEMX_DYLIB): $(MEMX_SRC) $(MEMX_INC)
	@mkdir -p $(dir $@)
	$(CC) -dynamiclib -O2 -I$(MEMX_HOME)/include -framework Metal -framework Foundation -lz \
	      -install_name @rpath/libmemx_runtime.dylib -o $@ $(MEMX_SRC)

# Metal GPU support (set USE_METAL=0 to disable)
USE_METAL ?= 1
ifeq ($(USE_METAL),1)
  CFLAGS  += -DSTRATUM_USE_METAL
  LDFLAGS += -framework Metal -framework Foundation -fobjc-arc
  METAL_OBJ := stratum_metal.o
  METAL_LIB := stratum_q4k.metallib
else
  METAL_OBJ :=
  METAL_LIB :=
endif

# Architecture implementations are auto-collected: drop a new
# stratum_arch_*.inc.c into this directory and rebuild — stratum.c
# never needs to be edited to register an architecture.
ARCH_INCS := $(wildcard stratum_arch_*.inc.c)

stratum_archs.gen.h: $(ARCH_INCS)
	@rm -f $@; for f in $(ARCH_INCS); do echo "#include \"$$f\""; done > $@

stratum: stratum.c stratum_archs.gen.h $(METAL_OBJ) $(MEMX_OBJ) \
         stratum_gguf.h stratum_arch.h stratum_linear.h stratum_engine.h stratum_soft_io.h \
         stratum_metal.h \
         stratum_memx.h \
         stratum_sro.h \
         stratum_q2k.h stratum_q2k_neon.h stratum_q3k.h stratum_q3k_neon.h \
         stratum_q4k.h stratum_q4k_neon.h \
         stratum_q5k.h stratum_q5k_neon.h \
         stratum_q6k.h stratum_q6k_neon.h \
         stratum_q8_0.h \
         stratum_bf16.h \
         stratum_lin_attn_v3.h stratum_lin_attn_v3_bf16.h stratum_st.h
	$(CC) $(CFLAGS) -o $@ stratum.c $(METAL_OBJ) $(LDFLAGS)

stratum_metal.o: stratum_metal.m stratum_metal.h
	$(CC) -ObjC -O3 -fobjc-arc -c stratum_metal.m -o stratum_metal.o

stratum_q4k.metallib: stratum_q4k.metal stratum_q5k.metal stratum_q6k.metal stratum_q2k.metal
	xcrun -sdk macosx metal -c stratum_q4k.metal -o stratum_q4k.air
	xcrun -sdk macosx metal -c stratum_q5k.metal -o stratum_q5k.air
	xcrun -sdk macosx metal -c stratum_q6k.metal -o stratum_q6k.air
	xcrun -sdk macosx metal -c stratum_q2k.metal -o stratum_q2k.air
	xcrun -sdk macosx metallib stratum_q4k.air stratum_q5k.air stratum_q6k.air stratum_q2k.air \
	    -o stratum_q4k.metallib
	rm -f stratum_q4k.air stratum_q5k.air stratum_q6k.air stratum_q2k.air

# stratum_p35.c is kept in the tree as a historical Phase-7 archive only:
# it depends on stratum_prefetch.h, which no longer exists, so it is NOT
# buildable and has no build target. See stratum_p35.c header comment.

quant_test: quant_test.c stratum_q2k.h stratum_q2k_neon.h stratum_q3k.h stratum_q3k_neon.h stratum_q4k.h \
            stratum_q4k_neon.h stratum_q5k.h stratum_q5k_neon.h \
            stratum_q6k.h stratum_q6k_neon.h stratum_q8_0.h stratum_bf16.h
	$(CC) -O3 -ffast-math -fblocks -mcpu=apple-m1 -o $@ quant_test.c

metal_test: metal_test.c stratum_metal.o stratum_q4k.metallib
	$(CC) -O3 -ffast-math -fobjc-arc -framework Metal -framework Foundation \
	    -o $@ metal_test.c stratum_metal.o

spec_sample_test: spec_sample_test.c
	$(CC) -O2 -o $@ spec_sample_test.c

# Quick correctness sanity:
#   - Run all quant kernel cross-validation
#   - Run TinyLlama generation and check the output starts as expected
# Backend consistency (CPU vs GPU-NC vs GPU2) is separate — needs a small
# model and the metallib:  make verify MODEL=</path/to/small.gguf>
check: stratum quant_test spec_sample_test
	@echo "=== quant kernel cross-validation ==="
	./quant_test
	@echo ""
	@echo "=== Leviathan-Chen rejection-sampling exactness ==="
	./spec_sample_test
	@echo ""
	@echo "=== smoke: llama arch (NEON) — pass MODEL=<path-to-gguf> ==="
	@test -n "$(MODEL)" || { echo "make check MODEL=<model.gguf>  (no default model)"; exit 1; }
	./stratum "$(MODEL)" 2 1 450 7483 310 3444 338 2>&1 | tail -3

verify: stratum
	@test -n "$(MODEL)" || { echo "make verify MODEL=<small.gguf>  (no default model)"; exit 1; }
	./verify_backends.sh "$(MODEL)"

tests: quant_test spec_sample_test
	./quant_test
	./spec_sample_test

# Fetch + build the optional MemX dependency (clone/pull upstream, then
# compile the dylib). Re-run to pull upstream updates.
deps: $(MEMX_DYLIB)
	@echo "memx dependency ready: $(MEMX_DYLIB)"

clean:
	rm -f stratum quant_test metal_test spec_sample_test stratum_metal.o \
	      stratum_q4k.metallib stratum_q4k.air stratum_archs.gen.h
	rm -rf $(MEMX_HOME)/build
