CC := gcc
CFLAGS := -O3 -Wall -fPIC -I.
LDFLAGS := -shared
BUILD_DIR := .
TARGET_LIBS := libsvd.so libmetrics.so

# sources (update names if yours differ)
SVD_SRC := svd.c
METRICS_SRC := metrics.c

.PHONY: all clean install

all: $(TARGET_LIBS)

libsvd.so: $(SVD_SRC)
	$(CC) $(CFLAGS) -o $@ $(LDFLAGS) $^

libmetrics.so: $(METRICS_SRC)
	$(CC) $(CFLAGS) -o $@ $(LDFLAGS) $^

install: all
	@mkdir -p ../hybrid_c_python/C_backend
	@mkdir -p ../c_libs
	cp -f libsvd.so ../hybrid_c_python/C_backend/
	cp -f libmetrics.so ../hybrid_c_python/C_backend/
	cp -f libsvd.so ../c_libs/
	cp -f libmetrics.so ../c_libs/
	@echo "Copied libraries to hybrid_c_python/C_backend and c_libs"

clean:
	rm -f libsvd.so libmetrics.so

