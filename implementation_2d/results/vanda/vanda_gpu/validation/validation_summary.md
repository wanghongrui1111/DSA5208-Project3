# Validation summary

## Grid refinement (serial, tol 1e-12)

| n | h | iterations | max error | L2 error | observed order |
|---|---|---|---|---|---|
| 31 | 0.0312 | 117 | 1.5506e-02 | 6.9678e-03 |  |
| 63 | 0.0156 | 237 | 3.8696e-03 | 1.7400e-03 | 2.003 |
| 127 | 0.00781 | 478 | 9.6748e-04 | 4.3488e-04 | 2.000 |
| 255 | 0.00391 | 963 | 2.4186e-04 | 1.0871e-04 | 2.000 |
| 511 | 0.00195 | 1935 | 6.0466e-05 | 2.7178e-05 | 2.000 |
| 1023 | 0.000977 | 3890 | 1.5116e-05 | 6.7944e-06 | 2.000 |

## Parallel runs against the serial reference (n = 257)

| implementation | halo | CG variant | runs | serial iterations | iteration range | max rel. error difference | max true residual | result |
|---|---|---|---|---|---|---|---|---|
| mpi_cuda | blocking | cgcg | 12 | 873 | 873-873 | 4.20e-10 | 9.89e-11 | PASS |
| mpi_cuda | blocking | classic | 12 | 873 | 873-873 | 0.00e+00 | 9.89e-11 | PASS |
| mpi_cuda | blocking | pipe | 12 | 774 | 774-774 | 3.49e-07 | 1.43e-08 | PASS |
| mpi_cuda | overlap | cgcg | 12 | 873 | 873-873 | 4.20e-10 | 9.89e-11 | PASS |
| mpi_cuda | overlap | classic | 12 | 873 | 873-873 | 0.00e+00 | 9.90e-11 | PASS |
| mpi_cuda | overlap | pipe | 12 | 774 | 774-774 | 3.49e-07 | 1.48e-08 | PASS |

## Reproducibility (n = 257, same run with 1/2/4 threads, repeated)

| implementation | halo | reduction | runs | distinct final residuals (bits) | result |
|---|---|---|---|---|---|
| mpi_cuda | overlap | fast | 3 | 1 | (not guaranteed) |
| mpi_cuda | overlap | repro | 3 | 1 | PASS |

**Overall: PASS**
