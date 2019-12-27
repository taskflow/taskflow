#ifndef FFT_H
#define FFT_H

/* our real numbers */
typedef double REAL;

/* Complex numbers and operations */
typedef struct {
     REAL re, im;
} COMPLEX;

#define c_re(c)  ((c).re)
#define c_im(c)  ((c).im)

void compute_w_coefficients(int n, int a, int b, COMPLEX * W);
void compute_w_coefficients_seq(int n, int a, int b, COMPLEX * W);
int factor(int n);
void unshuffle(int a, int b, COMPLEX * in, COMPLEX * out, int r, int m);
void unshuffle_seq(int a, int b, COMPLEX * in, COMPLEX * out, int r, int m);
void fft_twiddle_gen1(COMPLEX * in, COMPLEX * out, COMPLEX * W, int r, int m, int nW, int nWdnti, int nWdntm);
void fft_twiddle_gen(int i, int i1, COMPLEX * in, COMPLEX * out, COMPLEX * W, int nW, int nWdn, int r, int m);
void fft_twiddle_gen_seq(int i, int i1, COMPLEX * in, COMPLEX * out, COMPLEX * W, int nW, int nWdn, int r, int m);
void fft_base_2(COMPLEX * in, COMPLEX * out);
void fft_twiddle_2(int a, int b, COMPLEX * in, COMPLEX * out, COMPLEX * W, int nW, int nWdn, int m);
void fft_twiddle_2_seq(int a, int b, COMPLEX * in, COMPLEX * out, COMPLEX * W, int nW, int nWdn, int m);
void fft_unshuffle_2(int a, int b, COMPLEX * in, COMPLEX * out, int m);
void fft_unshuffle_2_seq(int a, int b, COMPLEX * in, COMPLEX * out, int m);
void fft_base_4(COMPLEX * in, COMPLEX * out);
void fft_twiddle_4(int a, int b, COMPLEX * in, COMPLEX * out, COMPLEX * W, int nW, int nWdn, int m);
void fft_twiddle_4_seq(int a, int b, COMPLEX * in, COMPLEX * out, COMPLEX * W, int nW, int nWdn, int m);
void fft_unshuffle_4(int a, int b, COMPLEX * in, COMPLEX * out, int m);
void fft_unshuffle_4_seq(int a, int b, COMPLEX * in, COMPLEX * out, int m);
void fft_base_8(COMPLEX * in, COMPLEX * out);
void fft_twiddle_8(int a, int b, COMPLEX * in, COMPLEX * out, COMPLEX * W, int nW, int nWdn, int m);
void fft_twiddle_8_seq(int a, int b, COMPLEX * in, COMPLEX * out, COMPLEX * W, int nW, int nWdn, int m);
void fft_unshuffle_8(int a, int b, COMPLEX * in, COMPLEX * out, int m);
void fft_unshuffle_8_seq(int a, int b, COMPLEX * in, COMPLEX * out, int m);
void fft_base_16(COMPLEX * in, COMPLEX * out);
void fft_twiddle_16(int a, int b, COMPLEX * in, COMPLEX * out, COMPLEX * W, int nW, int nWdn, int m);
void fft_twiddle_16_seq(int a, int b, COMPLEX * in, COMPLEX * out, COMPLEX * W, int nW, int nWdn, int m);
void fft_unshuffle_16(int a, int b, COMPLEX * in, COMPLEX * out, int m);
void fft_unshuffle_16_seq(int a, int b, COMPLEX * in, COMPLEX * out, int m);
void fft_base_32(COMPLEX * in, COMPLEX * out);
void fft_twiddle_32(int a, int b, COMPLEX * in, COMPLEX * out, COMPLEX * W, int nW, int nWdn, int m);
void fft_twiddle_32_seq(int a, int b, COMPLEX * in, COMPLEX * out, COMPLEX * W, int nW, int nWdn, int m);
void fft_unshuffle_32(int a, int b, COMPLEX * in, COMPLEX * out, int m);
void fft_unshuffle_32_seq(int a, int b, COMPLEX * in, COMPLEX * out, int m);
void fft_aux(int n, COMPLEX * in, COMPLEX * out, int *factors, COMPLEX * W, int nW);
void fft_aux_seq(int n, COMPLEX * in, COMPLEX * out, int *factors, COMPLEX * W, int nW);
void fft(int n, COMPLEX * in, COMPLEX * out);
void fft_seq(int n, COMPLEX * in, COMPLEX * out);
int test_correctness(int n, COMPLEX *out1, COMPLEX *out2);

void fft_tf(int n, COMPLEX * in, COMPLEX * out);

#endif

