include make.inc

#-------------------------------------------------------------------------------
# Executables

all: example_gemmtr

example_gemmtr: example_gemmtr.o
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

#-------------------------------------------------------------------------------
# Rules

.PHONY: all

.cpp.o:
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f *.o
	rm -f example_gemmtr
