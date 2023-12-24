CXX = g++
CXXFLAGS = -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes`
TARGET = llama_model`python3-config --extension-suffix`
SOURCE = bind_api.cpp

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) run.c -o $(TARGET) $(SOURCE)

run: 
	g++ -O3 -o run run.c

clean:
	rm -f $(TARGET)