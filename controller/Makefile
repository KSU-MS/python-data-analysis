# Native build of the VCU controller for the sim. On the car the same sources
# compile into the firmware.
CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Werror -fPIC -fno-exceptions -fno-rtti

build/libvcu_control.so: vcu_control.cpp vcu_control.h
	mkdir -p build
	$(CXX) $(CXXFLAGS) -shared -o $@ vcu_control.cpp

clean:
	rm -rf build

.PHONY: clean
