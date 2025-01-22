all: shared.out texture.out

shared.out:
	nvcc shared.cu -o shared.out `pkg-config --cflags --libs opencv`

texture.out:
	nvcc texture.cu -o texture.out `pkg-config --cflags --libs opencv`

clean:
	rm  ./shared.out ./texture.out 