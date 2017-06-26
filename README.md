# PALM-NMF

Implementation of the NMF algorithm using the PALM framework with smoothness and sparsity constraints

## Example

```
V = rand(50,1000);
params = struct;
params.r = 5;
params.max_iter = 200;

[W,H,objective_function,iteration_times] = palm_nmf(V,params);
```

