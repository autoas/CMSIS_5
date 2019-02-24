#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "arm_math.h"
#include "arm_nnfunctions.h"

q7_t* load(const char* file)
{
	size_t sz;
	q7_t* in;
	FILE* fp = fopen(file,"rb");
	assert(fp);
	fseek(fp, 0, SEEK_END);
	sz = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	in = malloc(sz);
	fread(in, 1, sz, fp);
	fclose(fp);
	return in;
}

void save(const char* file, q7_t* out, size_t sz)
{
	FILE* fp = fopen(file,"wb");
	fwrite(out, 1, sz, fp);
	fclose(fp);
}

q7_t conv1_out[28*28*32];
q7_t pool1_out[14*14*32];
q7_t conv2_out[14*14*64];
q7_t pool2_out[7*7*64];
q7_t fc1_out[1024];
q7_t fc2_out[10];
q7_t y_out[10];
int main(int argc, char* argv[])
{
	q7_t *input,*W_conv1,*b_conv1,*W_conv2,*b_conv2,*W_fc1,*b_fc1,*W_fc2,*b_fc2;
	printf("loading input&weights...\n");
	input = load("tmp/input_7.raw");
	W_conv1 = load("tmp/W_conv1_9.raw");
	b_conv1 = load("tmp/b_conv1_10.raw");

	#define CONV1_IM_DIM 28
	#define CONV1_IM_CH  1
	#define CONV1_OUT_CH 32
	#define CONV1_KER_DIM 5
	#define CONV1_PADDING ((CONV1_KER_DIM-1)/2)
	#define CONV1_STRIDE  1
	#define CONV1_OUT_DIM 28
	#define CONV1_BIAS_LSHIFT 6  // 7+9-10
	#define CONV1_OUT_RSHIFT  10 // Q1.6
	arm_convolve_HWC_q7_basic(input, CONV1_IM_DIM, CONV1_IM_CH, W_conv1, CONV1_OUT_CH, CONV1_KER_DIM, CONV1_PADDING,
						  CONV1_STRIDE, b_conv1, CONV1_BIAS_LSHIFT, CONV1_OUT_RSHIFT, conv1_out, CONV1_OUT_DIM,
						  NULL, NULL);
	save("tmp/conv1_out.raw", conv1_out, sizeof(conv1_out));

	arm_relu_q7(conv1_out, CONV1_OUT_DIM * CONV1_OUT_DIM * CONV1_OUT_CH);
	save("tmp/relu1_out.raw", conv1_out, sizeof(conv1_out));

	#define POOL1_KER_DIM 2
	#define POOL1_PADDING ((POOL1_KER_DIM-1)/2)
	#define POOL1_STRIDE  2
	#define POOL1_OUT_DIM 14
	arm_maxpool_q7_HWC(conv1_out, CONV1_OUT_DIM, CONV1_OUT_CH, POOL1_KER_DIM,
						  POOL1_PADDING, POOL1_STRIDE, POOL1_OUT_DIM, NULL, pool1_out);
	save("tmp/pool1_out.raw", pool1_out, sizeof(pool1_out));

	W_conv2 = load("tmp/W_conv2_9.raw");
	b_conv2 = load("tmp/b_conv2_10.raw");
	#define CONV2_IM_DIM 14
	#define CONV2_IM_CH  32
	#define CONV2_OUT_DIM 14
	#define CONV2_OUT_CH 64
	#define CONV2_STRIDE 1
	#define CONV2_KER_DIM 5
	#define CONV2_PADDING ((CONV2_KER_DIM-1)/2)
	#define CONV2_BIAS_LSHIFT 5  // 6+9-10
	#define CONV2_OUT_RSHIFT  10 // Q2.5
	arm_convolve_HWC_q7_fast(pool1_out, CONV2_IM_DIM, CONV2_IM_CH, W_conv2, CONV2_OUT_CH, CONV2_KER_DIM,
						  CONV2_PADDING, CONV2_STRIDE, b_conv2, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, conv2_out,
						  CONV2_OUT_DIM, NULL, NULL);
	save("tmp/conv2_out.raw", conv2_out, sizeof(conv2_out));

	arm_relu_q7(conv2_out, CONV2_OUT_DIM * CONV2_OUT_DIM * CONV2_OUT_CH);
	save("tmp/relu2_out.raw", conv2_out, sizeof(conv2_out));

	#define POOL2_KER_DIM 2
	#define POOL2_PADDING ((POOL2_KER_DIM-1)/2)
	#define POOL2_STRIDE  2
	#define POOL2_OUT_DIM 7
	arm_maxpool_q7_HWC(conv2_out, CONV2_OUT_DIM, CONV2_OUT_CH, POOL2_KER_DIM,
						  POOL2_PADDING, POOL2_STRIDE, POOL2_OUT_DIM, NULL, pool2_out);
	save("tmp/pool2_out.raw", pool2_out, sizeof(pool2_out));

	W_fc1 = load("tmp/W_fc1_9_opt.raw");
	b_fc1 = load("tmp/b_fc1_10.raw");
	#define IP1_DIM 3136
	#define IP1_OUT 1024
	#define IP1_BIAS_LSHIFT 4  // 5+9-10
	#define IP1_OUT_RSHIFT  10 // Q3.4
	arm_fully_connected_q7_opt(pool2_out, W_fc1, IP1_DIM, IP1_OUT, IP1_BIAS_LSHIFT, IP1_OUT_RSHIFT, b_fc1,
						  fc1_out, NULL);
	save("tmp/fc1_out.raw", fc1_out, sizeof(fc1_out));
	arm_relu_q7(fc1_out, IP1_OUT);
	save("tmp/relu3_out.raw", fc1_out, sizeof(fc1_out));

	W_fc2 = load("tmp/W_fc2_9_opt.raw");
	b_fc2 = load("tmp/b_fc2_10.raw");
	#define IP2_DIM 1024
	#define IP2_OUT 10
	#define IP2_BIAS_LSHIFT 3  // 4+9-10
	#define IP2_OUT_RSHIFT  9  // Q3.4
	arm_fully_connected_q7_opt(fc1_out, W_fc2, IP2_DIM, IP2_OUT, IP2_BIAS_LSHIFT, IP2_OUT_RSHIFT, b_fc2,
						  fc2_out, NULL);
	save("tmp/fc2_out.raw", fc2_out, sizeof(fc2_out));

	arm_softmax_q7(fc2_out, IP2_OUT, y_out);
	save("tmp/y_out.raw", y_out, sizeof(y_out));

	printf("inference is done!\n");
	return 0;
}
