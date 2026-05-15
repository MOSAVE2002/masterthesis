CXX      = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra
TARGET   = ha_solver

$(TARGET): ha_solver.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) ha_solver.cpp

clean:
	rm -f $(TARGET)

.PHONY: clean
