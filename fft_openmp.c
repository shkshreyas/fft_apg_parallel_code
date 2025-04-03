#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <string.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>

/* Define OpenMP compatibility layer for systems without OpenMP */
#ifdef _OPENMP
  #include <omp.h>
#else
  /* OpenMP function replacements */
  #define omp_get_max_threads() 1
  #define omp_get_num_threads() 1
  #define omp_set_num_threads(x) ((void)0)
  #define omp_get_thread_num() 0
  #define omp_get_wtime() ((double)clock() / CLOCKS_PER_SEC)
  #define omp_in_parallel() 0
  /* OpenMP pragma replacements */
  #define _Pragma(x)
  #define omp_pragma(x)
#endif

typedef enum {
    FFT_COOLEY_TUKEY,
    FFT_SPLIT_RADIX,
    FFT_STOCKHAM,
    FFT_BLUESTEIN,
    FFT_MIXED_RADIX
} FFT_Algorithm;

typedef struct {
    int tile_size_l1;
    int tile_size_l2;
    int vector_length;
} Cache_Parameters;

typedef struct {
    FFT_Algorithm algorithm;
    int num_threads;
    int adaptive_threading;
    int use_simd;
    int numa_aware;
    Cache_Parameters cache_params;
    int auto_tune;
} FFT_Config;

typedef double complex cmplx;

void fft_reconfigurable(cmplx* input, cmplx* output, int n, FFT_Config* config);
void fft_cooley_tukey(cmplx* input, cmplx* output, int n, int stride, int offset, FFT_Config* config);
void fft_split_radix(cmplx* input, cmplx* output, int n, int stride, int offset, FFT_Config* config);
void fft_stockham(cmplx* input, cmplx* output, int n, FFT_Config* config);
void fft_bluestein(cmplx* input, cmplx* output, int n, FFT_Config* config);
void fft_mixed_radix(cmplx* input, cmplx* output, int n, FFT_Config* config);
void auto_tune_fft(cmplx* input, cmplx* output, int n, FFT_Config* config);
void bit_reverse_copy(cmplx* input, cmplx* output, int n);
int is_power_of_two(int n);
int next_power_of_two(int n);
void optimize_cache_parameters(int n, FFT_Config* config);

int main(int argc, char* argv[]) {
    int n = 1 << 12; // Default size 2^12 = 4096
    int num_runs = 3;
    FFT_Algorithm algorithm = FFT_STOCKHAM; // Default algorithm
    
    // Parse command line arguments
    if (argc > 1) {
        int power = atoi(argv[1]);
        if (power >= 4 && power <= 20) { // Limit size between 2^4 and 2^20
            n = 1 << power;
        } else {
            printf("Invalid size power (must be between 4 and 20), using default: 2^12\n");
        }
    }
    
    if (argc > 2) {
        int alg = atoi(argv[2]);
        if (alg >= 0 && alg <= 4) {
            algorithm = (FFT_Algorithm)alg;
            // Non-power-of-two sizes require Bluestein or Mixed-Radix
            if (!is_power_of_two(n) && (algorithm == FFT_COOLEY_TUKEY || 
                algorithm == FFT_SPLIT_RADIX || algorithm == FFT_STOCKHAM)) {
                printf("Selected algorithm requires power-of-two size, using FFT_BLUESTEIN instead\n");
                algorithm = FFT_BLUESTEIN;
            }
        }
    }
    
    printf("Starting FFT calculation with size %d (2^%d)\n", n, (int)log2(n));
    printf("Algorithm: %d (%s)\n", algorithm, 
           algorithm == FFT_COOLEY_TUKEY ? "Cooley-Tukey" :
           algorithm == FFT_SPLIT_RADIX ? "Split-Radix" :
           algorithm == FFT_STOCKHAM ? "Stockham" :
           algorithm == FFT_BLUESTEIN ? "Bluestein" : "Mixed-Radix");
    
    FFT_Config config = {
        .algorithm = algorithm,
        .num_threads = 1, // Force single thread since OpenMP is not available
        .adaptive_threading = 0,
        .use_simd = 1,
        .numa_aware = 0,
        .cache_params = {
            .tile_size_l1 = 64,
            .tile_size_l2 = 512,
            .vector_length = 4
        },
        .auto_tune = 0 // Disabled auto-tuning
    };
    
    printf("Allocating memory...\n");
    cmplx* input = (cmplx*)malloc(n * sizeof(cmplx));
    cmplx* output = (cmplx*)malloc(n * sizeof(cmplx));
    
    if (!input || !output) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    printf("Initializing input data...\n");
    for (int i = 0; i < n; i++) {
        input[i] = cos(2 * M_PI * i / n) + 0.5 * cos(2 * M_PI * i * 3 / n) + 
                  0.25 * cos(2 * M_PI * i * 7 / n) + 0.125 * sin(2 * M_PI * i * 15 / n) + 
                  0.0 * I;
    }
    
    if (config.auto_tune) {
        printf("Auto-tuning FFT parameters...\n");
        auto_tune_fft(input, output, n, &config);
    }
    
    optimize_cache_parameters(n, &config);
    
    double total_time = 0.0;
    
    for (int run = 0; run < num_runs; run++) {
        printf("Starting run %d...\n", run);
        double start_time = omp_get_wtime();
        
        fft_reconfigurable(input, output, n, &config);
        
        double end_time = omp_get_wtime();
        double run_time = end_time - start_time;
        
        if (run > 0) {
            total_time += run_time;
            printf("Run %d: %.6f seconds\n", run, run_time);
        } else {
            printf("Warm-up run: %.6f seconds\n", run_time);
        }
    }
    
    double avg_time = total_time / (num_runs - 1);
    double gflops = 5.0 * n * log2(n) / (avg_time * 1e9);
    
    printf("\nResults:\n");
    printf("FFT Size: %d points\n", n);
    printf("Algorithm: %d (%s)\n", algorithm,
           algorithm == FFT_COOLEY_TUKEY ? "Cooley-Tukey" :
           algorithm == FFT_SPLIT_RADIX ? "Split-Radix" :
           algorithm == FFT_STOCKHAM ? "Stockham" :
           algorithm == FFT_BLUESTEIN ? "Bluestein" : "Mixed-Radix");
    printf("Threads: %d\n", config.num_threads);
    printf("Average Time: %.6f seconds\n", avg_time);
    printf("Performance: %.2f GFLOPS\n", gflops);
    
    printf("First few output values:\n");
    for (int i = 0; i < 10 && i < n; i++) {
        printf("%d: %.6f + %.6fi\n", i, creal(output[i]), cimag(output[i]));
    }
    
    free(input);
    free(output);
    
    printf("FFT calculation completed successfully\n");
    return 0;
}

void fft_reconfigurable(cmplx* input, cmplx* output, int n, FFT_Config* config) {
    printf("Starting FFT with algorithm: %d, size: %d\n", config->algorithm, n);
    omp_set_num_threads(config->num_threads);
    
    switch (config->algorithm) {
        case FFT_COOLEY_TUKEY:
            if (is_power_of_two(n)) {
                printf("Using Cooley-Tukey algorithm\n");
                memcpy(output, input, n * sizeof(cmplx));
                fft_cooley_tukey(output, input, n, 1, 0, config);
                memcpy(output, input, n * sizeof(cmplx)); // Copy back results
            } else {
                fprintf(stderr, "Error: Cooley-Tukey requires power-of-two size\n");
            }
            break;
            
        case FFT_STOCKHAM:
            printf("Using Stockham algorithm\n");
            fft_stockham(input, output, n, config);
            break;
            
        default:
            fprintf(stderr, "Using default Cooley-Tukey algorithm\n");
            if (is_power_of_two(n)) {
                memcpy(output, input, n * sizeof(cmplx));
                fft_cooley_tukey(output, input, n, 1, 0, config);
                memcpy(output, input, n * sizeof(cmplx)); // Copy back results
            } else {
                fprintf(stderr, "Error: Cooley-Tukey requires power-of-two size\n");
            }
            break;
    }
    printf("FFT calculation completed\n");
}

void fft_cooley_tukey(cmplx* input, cmplx* output, int n, int stride, int offset, FFT_Config* config) {
    if (n == 1) {
        if (stride == 1) {
            output[offset] = input[offset];
        } else {
            output[offset] = input[offset * stride];
        }
        return;
    }

    int m = n / 2;
    
    // Simplified version without OpenMP tasks to avoid crashes
    fft_cooley_tukey(input, output, m, stride * 2, offset, config);
    fft_cooley_tukey(input, output, m, stride * 2, offset + stride, config);
    
    // Combine results
    for (int k = 0; k < m; k++) {
        int idx1 = offset + k;
        int idx2 = offset + k + m;
        cmplx t = output[idx2] * cexp(-2.0 * I * M_PI * k / n);
        cmplx u = output[idx1];
        output[idx1] = u + t;
        output[idx2] = u - t;
    }
}

/* Function to check if a number is a power of two */
int is_power_of_two(int n) {
    return (n > 0) && ((n & (n - 1)) == 0);
}

/* Function to find the next power of two */
int next_power_of_two(int n) {
    if (n <= 0) return 1;
    if (is_power_of_two(n)) return n;
    
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

/* Bit-reverse copy function for FFT */
void bit_reverse_copy(cmplx* input, cmplx* output, int n) {
    int bits = (int)log2(n);
    
    #ifdef _OPENMP
    #pragma omp parallel for if(n >= 1024)
    #endif
    for (int i = 0; i < n; i++) {
        int rev = 0;
        for (int j = 0; j < bits; j++) {
            rev = (rev << 1) | ((i >> j) & 1);
        }
        output[rev] = input[i];
    }
}

/* Split-Radix FFT implementation */
void fft_split_radix(cmplx* input, cmplx* output, int n, int stride, int offset, FFT_Config* config) {
    if (n == 1) {
        if (stride == 1) {
            output[offset] = input[offset];
        } else {
            output[offset] = input[offset * stride];
        }
        return;
    }
    
    int m = n / 2;
    int m4 = n / 4;
    
    #ifdef _OPENMP
    #pragma omp task if(n >= 1024 && omp_get_num_threads() > 1)
    #endif
    fft_split_radix(input, output, m, stride * 2, offset, config);
    
    #ifdef _OPENMP
    #pragma omp task if(n >= 1024 && omp_get_num_threads() > 1)
    #endif
    fft_split_radix(input, output, m4, stride * 4, offset + stride, config);
    
    #ifdef _OPENMP
    #pragma omp task if(n >= 1024 && omp_get_num_threads() > 1)
    #endif
    fft_split_radix(input, output, m4, stride * 4, offset + stride * 3, config);
    
    #ifdef _OPENMP
    #pragma omp taskwait
    #endif
    
    #ifdef _OPENMP
    #pragma omp parallel for if(n >= 1024)
    #endif
    for (int k = 0; k < m4; k++) {
        int idx0 = offset + k;
        int idx1 = offset + k + m;
        int idx2 = offset + k + m/2;
        int idx3 = offset + k + m + m/2;
        
        cmplx t1 = output[idx2] * cexp(-2.0 * I * M_PI * k / n);
        cmplx t2 = output[idx3] * cexp(-2.0 * I * M_PI * 3 * k / n);
        
        cmplx u = output[idx0];
        cmplx v = output[idx1];
        
        output[idx0] = u + t1 + t2;
        output[idx1] = u - t1 - t2;
        output[idx2] = u + t1 * I - t2 * I;
        output[idx3] = u - t1 * I + t2 * I;
    }
}

/* Simplified Stockham FFT implementation for demonstration */
void fft_stockham(cmplx* input, cmplx* output, int n, FFT_Config* config) {
    if (!is_power_of_two(n)) {
        fprintf(stderr, "Error: Stockham algorithm requires power-of-two size\n");
        return;
    }
    
    printf("Starting Stockham FFT with size %d\n", n);
    
    cmplx* buffer1 = (cmplx*)malloc(n * sizeof(cmplx));
    cmplx* buffer2 = (cmplx*)malloc(n * sizeof(cmplx));
    
    if (!buffer1 || !buffer2) {
        fprintf(stderr, "Memory allocation failed in Stockham FFT\n");
        free(buffer1);
        free(buffer2);
        return;
    }
    
    // Copy input to buffer1
    memcpy(buffer1, input, n * sizeof(cmplx));
    
    cmplx* src = buffer1;
    cmplx* dst = buffer2;
    
    int log2n = (int)log2(n);
    printf("Log2(n) = %d\n", log2n);
    
    for (int stage = 0; stage < log2n; stage++) {
        int m = 1 << stage;
        int m2 = m * 2;
        
        printf("Stage %d: m=%d\n", stage, m);
        
        for (int k = 0; k < n/2; k++) {
            int i = k % m;
            int j = k / m;
            
            int idx1 = j * m2 + i;
            int idx2 = j * m2 + i + m;
            
            if (idx1 >= n || idx2 >= n) {
                printf("Error: Index out of bounds: idx1=%d, idx2=%d, n=%d\n", idx1, idx2, n);
                continue;
            }
            
            cmplx t = src[idx2] * cexp(-2.0 * I * M_PI * i / m2);
            dst[k] = src[idx1] + t;
            dst[k + n/2] = src[idx1] - t;
        }
        
        // Swap buffers
        cmplx* temp = src;
        src = dst;
        dst = temp;
    }
    
    // Copy result to output
    if (src != buffer1) {
        memcpy(output, src, n * sizeof(cmplx));
    } else {
        memcpy(output, buffer1, n * sizeof(cmplx));
    }
    
    free(buffer1);
    free(buffer2);
    
    printf("Stockham FFT completed successfully\n");
}

/* Bluestein FFT implementation (for arbitrary sizes) */
void fft_bluestein(cmplx* input, cmplx* output, int n, FFT_Config* config) {
    int m = next_power_of_two(2 * n - 1);
    
    cmplx* a = (cmplx*)calloc(m, sizeof(cmplx));
    cmplx* b = (cmplx*)calloc(m, sizeof(cmplx));
    cmplx* c = (cmplx*)calloc(m, sizeof(cmplx));
    cmplx* temp = (cmplx*)malloc(m * sizeof(cmplx));
    
    if (!a || !b || !c || !temp) {
        fprintf(stderr, "Memory allocation failed in Bluestein FFT\n");
        free(a);
        free(b);
        free(c);
        free(temp);
        return;
    }
    
    /* Precompute chirp factors */
    #ifdef _OPENMP
    #pragma omp parallel for if(n >= 1024)
    #endif
    for (int k = 0; k < n; k++) {
        double phase = M_PI * k * k / n;
        cmplx chirp = cexp(-I * phase);
        a[k] = input[k] * chirp;
        b[k] = chirp;
        b[m-k-1] = chirp; /* Symmetric for convolution */
    }
    
    /* Perform convolution using power-of-two FFT */
    /* Forward FFT of a */
    memcpy(temp, a, m * sizeof(cmplx));
    fft_stockham(temp, c, m, config);
    
    /* Forward FFT of b */
    memcpy(temp, b, m * sizeof(cmplx));
    fft_stockham(temp, a, m, config);
    
    /* Multiply in frequency domain */
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int k = 0; k < m; k++) {
        c[k] *= a[k];
    }
    
    /* Inverse FFT */
    fft_stockham(c, temp, m, config);
    
    /* Scale and apply chirp factors */
    #ifdef _OPENMP
    #pragma omp parallel for if(n >= 1024)
    #endif
    for (int k = 0; k < n; k++) {
        double phase = M_PI * k * k / n;
        cmplx chirp = cexp(-I * phase);
        output[k] = temp[k] * chirp / m;
    }
    
    free(a);
    free(b);
    free(c);
    free(temp);
}

/* Mixed-Radix FFT implementation */
void fft_mixed_radix(cmplx* input, cmplx* output, int n, FFT_Config* config) {
    /* For simplicity, we'll implement a basic mixed-radix algorithm for n = 3^k * 2^m */
    if (is_power_of_two(n)) {
        /* If n is a power of 2, use Stockham algorithm */
        fft_stockham(input, output, n, config);
        return;
    }
    
    /* Check if n is divisible by 3 */
    if (n % 3 == 0) {
        int m = n / 3;
        
        cmplx* temp = (cmplx*)malloc(n * sizeof(cmplx));
        if (!temp) {
            fprintf(stderr, "Memory allocation failed in Mixed-Radix FFT\n");
            return;
        }
        
        /* Divide into 3 sub-problems */
        #ifdef _OPENMP
        #pragma omp parallel sections if(n >= 1024)
        {
            #pragma omp section
            {
        #endif
                cmplx* sub1 = (cmplx*)malloc(m * sizeof(cmplx));
                for (int i = 0; i < m; i++) sub1[i] = input[i*3];
                fft_mixed_radix(sub1, temp, m, config);
                free(sub1);
        #ifdef _OPENMP
            }
            
            #pragma omp section
            {
        #endif
                cmplx* sub2 = (cmplx*)malloc(m * sizeof(cmplx));
                for (int i = 0; i < m; i++) sub2[i] = input[i*3 + 1];
                fft_mixed_radix(sub2, temp + m, m, config);
                free(sub2);
        #ifdef _OPENMP
            }
            
            #pragma omp section
            {
        #endif
                cmplx* sub3 = (cmplx*)malloc(m * sizeof(cmplx));
                for (int i = 0; i < m; i++) sub3[i] = input[i*3 + 2];
                fft_mixed_radix(sub3, temp + 2*m, m, config);
                free(sub3);
        #ifdef _OPENMP
            }
        }
        #endif
        
        /* Combine results */
        cmplx w1 = cexp(-2.0 * I * M_PI / 3);
        cmplx w2 = w1 * w1;
        
        #ifdef _OPENMP
    #pragma omp parallel for if(n >= 1024)
    #endif
        for (int k = 0; k < m; k++) {
            cmplx t1 = temp[k];
            cmplx t2 = temp[k + m];
            cmplx t3 = temp[k + 2*m];
            
            output[k] = t1 + t2 + t3;
            output[k + m] = t1 + t2 * w1 + t3 * w2;
            output[k + 2*m] = t1 + t2 * w2 + t3 * w1;
        }
        
        free(temp);
    } else {
        /* Fallback to Bluestein for arbitrary sizes */
        fft_bluestein(input, output, n, config);
    }
}

/* Auto-tuning function for FFT parameters */
void auto_tune_fft(cmplx* input, cmplx* output, int n, FFT_Config* config) {
    const int num_algorithms = 5;
    FFT_Algorithm algorithms[5] = {
        FFT_COOLEY_TUKEY,
        FFT_SPLIT_RADIX,
        FFT_STOCKHAM,
        FFT_BLUESTEIN,
        FFT_MIXED_RADIX
    };
    
    double best_time = INFINITY;
    FFT_Algorithm best_algorithm = config->algorithm;
    int best_num_threads = config->num_threads;
    
    /* Test different algorithms */
    for (int alg = 0; alg < num_algorithms; alg++) {
        /* Skip algorithms that don't support the size */
        if ((algorithms[alg] == FFT_COOLEY_TUKEY || algorithms[alg] == FFT_SPLIT_RADIX || 
             algorithms[alg] == FFT_STOCKHAM) && !is_power_of_two(n)) {
            continue;
        }
        
        /* Test different thread counts */
        for (int threads = 1; threads <= omp_get_max_threads(); threads *= 2) {
            config->algorithm = algorithms[alg];
            config->num_threads = threads;
            omp_set_num_threads(threads);
            
            /* Run a test */
            double start_time = omp_get_wtime();
            
            cmplx* test_output = (cmplx*)malloc(n * sizeof(cmplx));
            if (!test_output) continue;
            
            /* Call the appropriate FFT function */
            switch (algorithms[alg]) {
                case FFT_COOLEY_TUKEY:
                    memcpy(test_output, input, n * sizeof(cmplx));
                    fft_cooley_tukey(test_output, output, n, 1, 0, config);
                    break;
                case FFT_SPLIT_RADIX:
                    memcpy(test_output, input, n * sizeof(cmplx));
                    fft_split_radix(test_output, output, n, 1, 0, config);
                    break;
                case FFT_STOCKHAM:
                    fft_stockham(input, test_output, n, config);
                    break;
                case FFT_BLUESTEIN:
                    fft_bluestein(input, test_output, n, config);
                    break;
                case FFT_MIXED_RADIX:
                    fft_mixed_radix(input, test_output, n, config);
                    break;
            }
            
            double end_time = omp_get_wtime();
            double run_time = end_time - start_time;
            
            free(test_output);
            
            printf("Algorithm: %d, Threads: %d, Time: %.6f seconds\n", 
                   algorithms[alg], threads, run_time);
            
            if (run_time < best_time) {
                best_time = run_time;
                best_algorithm = algorithms[alg];
                best_num_threads = threads;
            }
        }
    }
    
    /* Set the best configuration */
    config->algorithm = best_algorithm;
    config->num_threads = best_num_threads;
    omp_set_num_threads(best_num_threads);
    
    printf("Auto-tuning complete. Best algorithm: %d, Best thread count: %d\n", 
           best_algorithm, best_num_threads);
}

/* Function to optimize cache parameters based on problem size */
void optimize_cache_parameters(int n, FFT_Config* config) {
    int cache_line_size = 64; /* Typical cache line size in bytes */
    int l1_cache_size = 32 * 1024; /* Typical L1 cache size in bytes */
    int l2_cache_size = 256 * 1024; /* Typical L2 cache size in bytes */
    
    /* Calculate optimal tile sizes based on cache sizes */
    int element_size = sizeof(cmplx);
    int l1_elements = l1_cache_size / element_size;
    int l2_elements = l2_cache_size / element_size;
    
    /* Set tile sizes to be cache-friendly */
    config->cache_params.tile_size_l1 = (int)sqrt(l1_elements / 2);
    config->cache_params.tile_size_l2 = (int)sqrt(l2_elements / 2);
    
    /* Ensure tile sizes are powers of 2 for better FFT performance */
    config->cache_params.tile_size_l1 = next_power_of_two(config->cache_params.tile_size_l1);
    config->cache_params.tile_size_l2 = next_power_of_two(config->cache_params.tile_size_l2);
    
    /* Adjust vector length based on SIMD capabilities */
    if (config->use_simd) {
        /* Assuming AVX2 with 256-bit vectors (4 doubles) */
        config->cache_params.vector_length = 4;
    } else {
        config->cache_params.vector_length = 1;
    }
    
    printf("Cache parameters optimized: L1 tile = %d, L2 tile = %d, Vector length = %d\n",
           config->cache_params.tile_size_l1,
           config->cache_params.tile_size_l2,
           config->cache_params.vector_length);
}
