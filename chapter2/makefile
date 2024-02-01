APPS=checkThreadBlockIndex

all: ${APPS}

%: %.cu           # % matches anything, creating a file having a same name with %.cu with no suffix, which is exactly "hello"
	nvcc -O2 -o $@ $<

clean:
	rm -f ${APPS}

