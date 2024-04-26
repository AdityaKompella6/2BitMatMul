# Makefile

# Compiler and flags
CC = gcc
CFLAGS = -Wall -O2

# Source files
SRCS = main.cpp serial.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Target executable
TARGET = main

# Rules
main.o: main.cpp
	$(CC) $(CFLAGS) -c $<

serial.o: serial.cpp
	$(CC) $(CFLAGS) -c $<

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

# Default target
all: $(TARGET)

# Clean rule
clean:
	rm -f $(OBJS) $(TARGET)