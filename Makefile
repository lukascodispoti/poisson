DEBUG = 0

BINDIR = bin
OBJDIR = obj
MKDIR  = mkdir -p
RM     = rm -rf


CC      = mpic++
CFLAGS  = -Wall -Wextra -I. -mtune=generic -march=native
LDLIBS  = -lhdf5 -lm -lstdc++

ifeq ($(DEBUG), 1)
	CFLAGS += -g -O0 -fsanitize=address
else
	CFLAGS += -O3
endif

TARGET = poisson
TGT = $(BINDIR)/$(TARGET)

$(OBJDIR):
	$(MKDIR) -p $(OBJDIR)

$(BINDIR):
	$(MKDIR) -p $(BINDIR)

$(OBJDIR)/%.o: $(TARGET).cc
	$(CC) $(CFLAGS) -c $< -o $@

$(TGT): $(OBJDIR)/$(TARGET).o
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

.DEFAULT_GOAL := all
.PHONY: all
all: $(OBJDIR) $(BINDIR) $(TGT)

clean:
	$(RM) $(OBJDIR) $(BINDIR)