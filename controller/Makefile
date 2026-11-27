# Native build of the VCU controller for the sim, and a cross-compile for the
# Teensy 4.1 (Cortex-M7) to prove the same sources build for the car.
#   make            build/libvcu_control.so   (sim)
#   make teensy     build/vcu_control_m7.o    (needs `nix develop .#firmware`)
WARN := -Wall -Wextra -Werror -Wdouble-promotion
CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 $(WARN) -fPIC -fno-exceptions -fno-rtti
ARM_CXX ?= arm-none-eabi-g++
# Teensy 4.1: single-precision FPU only, so -Wdouble-promotion catches slow doubles
ARM_FLAGS := -O2 -std=c++17 $(WARN) -mcpu=cortex-m7 -mthumb -mfloat-abi=hard \
             -mfpu=fpv5-d16 -fno-exceptions -fno-rtti -ffunction-sections -fdata-sections

build/libvcu_control.so: vcu_control.cpp vcu_control.h
	mkdir -p build
	$(CXX) $(CXXFLAGS) -shared -o $@ vcu_control.cpp

teensy: build/vcu_control_m7.o
	arm-none-eabi-size $<

build/vcu_control_m7.o: vcu_control.cpp vcu_control.h
	mkdir -p build
	$(ARM_CXX) $(ARM_FLAGS) -c -o $@ vcu_control.cpp

clean:
	rm -rf build

.PHONY: teensy clean
