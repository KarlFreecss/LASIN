# swipl: load_foreign_library(foreign('img_process.so')).

# Source file
SRC = test_c.c

# Object file
OBJ = $(SRC:.c = .o)

# Compiler
CC = gcc
CXX = g++

# Swi-prolog path
SWI_PL_HOME = /home/daiwz/.local/lib/swipl-7.3.15

#LIBS
#CFLAGS

CFLAGS = -Wall -fPIC -Winvalid-pch -std=c++11 -g
LIBS = -Wall -Winvalid-pch -fPIC

#CFLAGS_SWI = -shared -fno-strict-aliasing -pthread -fPIC -I$(SWI_PL_HOME)/include 
COFLAGS=-DO_DEBUG -gdwarf-2 -g3
CFLAGS_SWI = $(CFLAGS) -shared -fno-strict-aliasing -pthread -I$(SWI_PL_HOME)/include
LIBS_SWI = $(LIBS) -shared -lm -L$(SWI_PL_HOME)/lib/x86_64-linux -rdynamic -O2 -pthread -Wl,-rpath=$(SWI_PL_HOME)/lib/x86_64-linux -lswipl 
LIBS_PL = $(LIBS) -lm -L$(SWI_PL_HOME)/lib/x86_64-linux -rdynamic -O2 -pthread -Wl,-rpath=$(SWI_PL_HOME)/lib/x86_64-linux -lswipl 

all:example_ptr test_boost test_prolog.so test test_list

example_ptr:example_ptr.o
	$(CXX) $(LIBS) example_ptr.o -o example_ptr

example_ptr.o:
	$(CXX) $(CFLAGS) -c example_ptr.cpp

test_boost:test_boost.o test_mat.o
	$(CXX) $(LIBS) test_boost.o test_mat.o -o test_boost

test_boost.o:
	$(CXX) $(CFLAGS) -c test_boost.cpp

test_mat.o:
	$(CXX) $(CFLAGS) -c test_mat.cpp

test_prolog.so:test_prolog.o test_mat.o
	$(CXX) $(LIBS_SWI) test_prolog.o test_mat.o -o test_prolog.so

test_prolog.o:
	$(CXX) $(CFLAGS_SWI) -c test_prolog.cpp

test:test.o test_mat.o test_prolog.so
	$(CXX) $(LIBS_PL) test.o test_mat.o -o test

test.o:
	$(CXX) $(CFLAGS_SWI) -c test.cpp

test_list:test_list.o
	$(CXX) $(LIBS_PL) test_list.o -o test_list

test_list.o:
	$(CXX) $(CFLAGS_SWI) -c test_list.cpp


clean:
	rm -rf *.o *.so *.a test_boost example_ptr test_prolog.so test test_list

distclean:
	rm -rf *.o *.so *.a *.gch test_boost example_ptr test_prolog.so test
