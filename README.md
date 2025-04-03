# FFT Implementation in C

This repository contains a Fast Fourier Transform (FFT) implementation in C with support for multiple algorithms. The code is designed to work with or without OpenMP support.

## Compilation

To compile the code, use the following command:

```bash
gcc -o fft_openmp fft_openmp.c -lm
```

If you have OpenMP support, you can enable it with:

```bash
gcc -o fft_openmp fft_openmp.c -lm -fopenmp
```

## Usage

The program can be run with command-line arguments to specify the FFT size and algorithm:

```bash
./fft_openmp [size_power] [algorithm]
```

Where:
- `size_power` is the power of 2 for the FFT size (e.g., 10 for 1024 points, 12 for 4096 points)
- `algorithm` is the FFT algorithm to use:
  - 0: Cooley-Tukey (may have stability issues)
  - 1: Split-Radix (may have stability issues)
  - 2: Stockham (most stable, recommended)
  - 3: Bluestein (for non-power-of-two sizes)
  - 4: Mixed-Radix (for arbitrary sizes)

### Examples

1. Run with default settings (4096 points, Stockham algorithm):
   ```bash
   ./fft_openmp
   ```

2. Run with 1024 points using Stockham algorithm:
   ```bash
   ./fft_openmp 10 2
   ```

3. Run with 2048 points using Bluestein algorithm:
   ```bash
   ./fft_openmp 11 3
   ```

## Output

The program outputs:
- Execution time for each run
- Average execution time
- Performance in GFLOPS
- First few values of the FFT output

## Implementation Details

The code includes multiple FFT algorithms:

1. **Cooley-Tukey**: Classic recursive FFT algorithm (requires power-of-two sizes)
2. **Split-Radix**: More efficient variant of Cooley-Tukey (requires power-of-two sizes)
3. **Stockham**: In-place FFT algorithm with better cache performance (requires power-of-two sizes)
4. **Bluestein**: Supports arbitrary-sized FFTs using convolution
5. **Mixed-Radix**: Supports sizes that are products of small primes

The Stockham algorithm is recommended for most use cases as it provides good performance and stability.