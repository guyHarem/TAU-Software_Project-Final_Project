# Makefile for compiling and linking the symnmf program

# Compiler and flags
CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors -g
LDFLAGS = -lm

# Source files
SRCS = symnmf.c

# Executable, object files and headers
EXECUTABLE = symnmf
OBJ_FILE = $(SRCS:.c=.o)
HEADERS = symnmf.h

# Default target
all: symnmf

symnmf: symnmf.o
	@echo "Linking $(EXECUTABLE) executable"
	@$(CC) $(CFLAGS) symnmf.o -o symnmf $(LDFLAGS)

# Compile source file to object file
symnmf.o: symnmf.c 
	@echo "Compiling $(OBJ_FILE) to $(OBJ_FILE)"
	@$(CC) $(CFLAGS) -c symnmf.c 

clean:
	@echo "Cleaning up"
	@rm -f *.o symnmf

# Phony targets
.PHONY: all clean