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
CFLAGS  := -O3 -ffast-math -DACCELERATE_NEW_LAPACK -Wno-deprecated-declarations \
           -fblocks -mcpu=apple-m1
LDFLAGS := -framework Accelerate -lpthread

# MemX compressed-memory runtime — OPTIONAL, off by default. The engine's
# mmap + page-cache path is self-contained; MemX only changes where some
# anonymous buffers live. Enable explicitly with:
#   make USE_MEMX=1 MEMX_HOME=/path/to/memx
USE_MEMX ?= 0
ifeq ($(USE_MEMX),1)
  MEMX_HOME ?= $(HOME)/Desktop/memx
  CFLAGS  += -DSTRATUM_USE_MEMX -I$(MEMX_HOME)/include
  LDFLAGS += -L$(MEMX_HOME)/build -lmemx_runtime -Wl,-rpath,$(MEMX_HOME)/build
endif

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

.PHONY: all clean check tests verify

all: stratum $(METAL_LIB)

# Architecture implementations are auto-collected: drop a new
# stratum_arch_*.inc.c into this directory and rebuild — stratum.c
# never needs to be edited to register an architecture.
ARCH_INCS := $(wildcard stratum_arch_*.inc.c)

stratum_archs.gen.h: $(ARCH_INCS)
	@rm -f $@; for f in $(ARCH_INCS); do echo "#include \"$$f\""; done > $@

stratum: stratum.c stratum_archs.gen.h $(METAL_OBJ) \
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
ifeq ($(USE_MEMX),1)
	@install_name_tool -change build/libmemx_runtime.dylib $(MEMX_HOME)/build/libmemx_runtime.dylib $@ 2>/dev/null || true
endif

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

clean:
	rm -f stratum quant_test metal_test spec_sample_test stratum_metal.o \
	      stratum_q4k.metallib stratum_q4k.air stratum_archs.gen.h
