SRCDIR = .
BINDIR = bin
OBJDIR = obj
MKDIR  = mkdir -p
RM     = rm -rf

CXX      = mpic++
LDLIBS  = -lhdf5 -lm -lstdc++

ifndef $CFLAGS
	CFLAGS  = -Wall -Wextra -I. -mtune=generic -march=native -g
#	CFLAGS += -O0 -fsanitize=address
    CFLAGS += -O3
endif

TARGET = main
TGT = $(BINDIR)/$(TARGET)

$(OBJDIR):
	$(MKDIR) $(OBJDIR)

$(BINDIR):
	$(MKDIR) $(BINDIR)

$(OBJDIR)/poisson.o: $(SRCDIR)/poisson.cc $(SRCDIR)/poisson.h
	$(CXX) $(CFLAGS) -c $< -o $@

$(OBJDIR)/main.o: $(SRCDIR)/main.cc $(SRCDIR)/poisson.h
	$(CXX) $(CFLAGS) -c $< -o $@

OBJ = $(OBJDIR)/poisson.o $(OBJDIR)/main.o

$(TGT): $(OBJ)
	$(CXX) $(CFLAGS) $^ -o $@ $(LDLIBS) 

.DEFAULT_GOAL := all
.PHONY: all clean

all: $(OBJDIR) $(BINDIR) $(TGT) 

clean:
	$(RM) $(OBJDIR) $(BINDIR)