#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "arm_math.h"
#include "arm_nnfunctions.h"
#include "weights.h"

#define USE_NNOM

#ifdef USE_NNOM
#include "nnom.h"
#endif
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
#ifdef USE_SHELL
int mnist_main(int argc, char* argv[])
#else
int main(int argc, char* argv[])
#endif
{
	q7_t *input,*W_conv1,*b_conv1,*W_conv2,*b_conv2,*W_fc1,*b_fc1,*W_fc2,*b_fc2;
	printf("loading input&weights...\n");
	input = load("tmp/input.raw");
	W_conv1 = load("tmp/W_conv1.raw");
	b_conv1 = load("tmp/b_conv1.raw");

	#define CONV1_IM_DIM 28
	#define CONV1_IM_CH  1
	#define CONV1_OUT_CH 32
	#define CONV1_KER_DIM 5
	#define CONV1_PADDING ((CONV1_KER_DIM-1)/2)
	#define CONV1_STRIDE  1
	#define CONV1_OUT_DIM 28
	#define CONV1_BIAS_LSHIFT (INPUT_Q+W_CONV1_Q-B_CONV1_Q)
	#define CONV1_OUT_RSHIFT  (INPUT_Q+W_CONV1_Q-CONV1_OUT_Q)
#ifndef USE_NNOM
	arm_convolve_HWC_q7_basic(input, CONV1_IM_DIM, CONV1_IM_CH, W_conv1, CONV1_OUT_CH, CONV1_KER_DIM, CONV1_PADDING,
						  CONV1_STRIDE, b_conv1, CONV1_BIAS_LSHIFT, CONV1_OUT_RSHIFT, conv1_out, CONV1_OUT_DIM,
						  NULL, NULL);
	save("tmp/conv1_out.raw", conv1_out, sizeof(conv1_out));

	arm_relu_q7(conv1_out, CONV1_OUT_DIM * CONV1_OUT_DIM * CONV1_OUT_CH);
	save("tmp/relu1_out.raw", conv1_out, sizeof(conv1_out));
#endif
	#define POOL1_KER_DIM 2
	#define POOL1_PADDING ((POOL1_KER_DIM-1)/2)
	#define POOL1_STRIDE  2
	#define POOL1_OUT_DIM 14
#ifndef USE_NNOM
	arm_maxpool_q7_HWC(conv1_out, CONV1_OUT_DIM, CONV1_OUT_CH, POOL1_KER_DIM,
						  POOL1_PADDING, POOL1_STRIDE, POOL1_OUT_DIM, NULL, pool1_out);
	save("tmp/pool1_out.raw", pool1_out, sizeof(pool1_out));
#endif
	W_conv2 = load("tmp/W_conv2.raw");
	b_conv2 = load("tmp/b_conv2.raw");
	#define CONV2_IM_DIM 14
	#define CONV2_IM_CH  32
	#define CONV2_OUT_DIM 14
	#define CONV2_OUT_CH 64
	#define CONV2_STRIDE 1
	#define CONV2_KER_DIM 5
	#define CONV2_PADDING ((CONV2_KER_DIM-1)/2)
	#define CONV2_BIAS_LSHIFT (CONV1_OUT_Q+W_CONV2_Q-B_CONV2_Q)
	#define CONV2_OUT_RSHIFT  (CONV1_OUT_Q+W_CONV2_Q-CONV2_OUT_Q)
#ifndef USE_NNOM
	arm_convolve_HWC_q7_fast(pool1_out, CONV2_IM_DIM, CONV2_IM_CH, W_conv2, CONV2_OUT_CH, CONV2_KER_DIM,
						  CONV2_PADDING, CONV2_STRIDE, b_conv2, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, conv2_out,
						  CONV2_OUT_DIM, NULL, NULL);
	save("tmp/conv2_out.raw", conv2_out, sizeof(conv2_out));

	arm_relu_q7(conv2_out, CONV2_OUT_DIM * CONV2_OUT_DIM * CONV2_OUT_CH);
	save("tmp/relu2_out.raw", conv2_out, sizeof(conv2_out));
#endif
	#define POOL2_KER_DIM 2
	#define POOL2_PADDING ((POOL2_KER_DIM-1)/2)
	#define POOL2_STRIDE  2
	#define POOL2_OUT_DIM 7
#ifndef USE_NNOM
	arm_maxpool_q7_HWC(conv2_out, CONV2_OUT_DIM, CONV2_OUT_CH, POOL2_KER_DIM,
						  POOL2_PADDING, POOL2_STRIDE, POOL2_OUT_DIM, NULL, pool2_out);
	save("tmp/pool2_out.raw", pool2_out, sizeof(pool2_out));
#endif

	W_fc1 = load("tmp/W_fc1_opt.raw");
	b_fc1 = load("tmp/b_fc1.raw");
	#define IP1_DIM 3136
	#define IP1_OUT 1024
	#define IP1_BIAS_LSHIFT (CONV2_OUT_Q+W_FC1_Q-B_FC1_Q)
	#define IP1_OUT_RSHIFT  (CONV2_OUT_Q+W_FC1_Q-FC1_OUT_Q)
#ifndef USE_NNOM
	arm_fully_connected_q7_opt(pool2_out, W_fc1, IP1_DIM, IP1_OUT, IP1_BIAS_LSHIFT, IP1_OUT_RSHIFT, b_fc1,
						  fc1_out, NULL);
	save("tmp/fc1_out.raw", fc1_out, sizeof(fc1_out));
	arm_relu_q7(fc1_out, IP1_OUT);
	save("tmp/relu3_out.raw", fc1_out, sizeof(fc1_out));
#endif
	W_fc2 = load("tmp/W_fc2_opt.raw");
	b_fc2 = load("tmp/b_fc2.raw");
	#define IP2_DIM 1024
	#define IP2_OUT 10
	#define IP2_BIAS_LSHIFT (FC1_OUT_Q+W_FC2_Q-B_FC2_Q)
	#define IP2_OUT_RSHIFT  (FC1_OUT_Q+W_FC2_Q-FC2_OUT_Q)
#ifndef USE_NNOM
	arm_fully_connected_q7_opt(fc1_out, W_fc2, IP2_DIM, IP2_OUT, IP2_BIAS_LSHIFT, IP2_OUT_RSHIFT, b_fc2,
						  fc2_out, NULL);
	save("tmp/fc2_out.raw", fc2_out, sizeof(fc2_out));

	arm_softmax_q7(fc2_out, IP2_OUT, y_out);
	save("tmp/y_out.raw", y_out, sizeof(y_out));

	printf("inference is done!\n");
#else
	nnom_model_t model;
	static nnom_weight_t c1_w = {0, CONV1_OUT_RSHIFT};  c1_w.p_value = W_conv1;
	static nnom_bias_t   c1_b = {0, CONV1_BIAS_LSHIFT}; c1_b.p_value = b_conv1;
	static nnom_weight_t c2_w = {0, CONV2_OUT_RSHIFT};  c2_w.p_value = W_conv2;
	static nnom_bias_t   c2_b = {0, CONV2_BIAS_LSHIFT}; c2_b.p_value = b_conv2;
	static nnom_weight_t ip1_w = {0, IP1_OUT_RSHIFT};   ip1_w.p_value = W_fc1;
	static nnom_bias_t   ip1_b = {0, IP1_BIAS_LSHIFT};  ip1_b.p_value = b_fc1;
	static nnom_weight_t ip2_w = {0, IP2_OUT_RSHIFT};   ip2_w.p_value = W_fc2;
	static  nnom_bias_t   ip2_b = {0, IP2_BIAS_LSHIFT}; ip2_w.p_value = b_fc2;
	new_model(&model);
	model.add(&model, Input(shape(CONV1_IM_DIM, CONV1_IM_DIM, CONV1_IM_CH), qformat(0, 7), input));
	model.add(&model, Conv2D(CONV1_OUT_CH, kernel(CONV1_KER_DIM, CONV1_KER_DIM), stride(CONV1_STRIDE, CONV1_STRIDE), PADDING_SAME, &c1_w, &c1_b));
	model.add(&model, ReLU());
	model.add(&model, MaxPool(kernel(POOL1_KER_DIM, POOL1_KER_DIM), stride(POOL1_STRIDE, POOL1_STRIDE), PADDING_VALID));
	model.add(&model, Conv2D(CONV2_OUT_CH, kernel(CONV2_KER_DIM, CONV2_KER_DIM), stride(CONV2_STRIDE, CONV2_STRIDE), PADDING_SAME, &c2_w, &c2_b));
	model.add(&model, ReLU());
	model.add(&model, MaxPool(kernel(POOL2_KER_DIM, POOL2_KER_DIM), stride(POOL2_STRIDE, POOL2_STRIDE), PADDING_VALID));
	model.add(&model, Dense(IP1_OUT, &ip1_w, &ip1_b));
	model.add(&model, ReLU());
	model.add(&model, Dense(IP2_OUT, &ip2_w, &ip2_b));
	model.add(&model, Softmax());
	model.add(&model, Output(shape(IP2_OUT, 1, 1), qformat(0, 7), y_out));
	sequencial_compile(&model);

	model_run(&model);

	save("tmp/y_out.raw", y_out, sizeof(y_out));

	printf("inference is done!\n");
#endif
	free(input); free(W_conv1); free(b_conv1); free(W_conv2);
	free(b_conv2); free(W_fc1); free(b_fc1); free(W_fc2); free(b_fc2);
	return 0;
}

#ifdef USE_SHELL
#include "shell.h"

static SHELL_CONST ShellCmdT mnistCmd  = {
		mnist_main,
		0,1,
		"mnist",
		"mnist img",
		"sample mnist digit detection\n",
		{NULL,NULL}
};
SHELL_CMD_EXPORT(mnistCmd)
#endif
