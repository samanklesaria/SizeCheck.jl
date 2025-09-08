# SizeCheck: Runtime Shape Validation for Size-Annotated Julia Code

This library provides `sizecheck`, a macro that automatically adds runtime shape checking to Julia functions based on size-annotated variable names.

When writing Julia code, it's common to use naming conventions that indicate
tensor shapes, as in this [Medium
post](https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd).
For example, if a tensor `weights` has shape `N × K`, you might name the
variable `weights_NK`. This macro adds validation checks that tensors match
their annotated shapes at runtime.

```@autodocs
Modules = [SizeCheck]
```
