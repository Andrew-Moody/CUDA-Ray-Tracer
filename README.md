# CUDA Ray Tracer

### A GPU ray tracing renderer written in CUDA and C++ following the [_Ray Tracing in One Weekend_](https://raytracing.github.io/books/RayTracingInOneWeekend.html) tutorial by Peter Shirley supplemented with the NVIDIA Technical Blog [_Accelerated Ray Tracing in One Weekend in CUDA_](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/) by Roger Allen

![FinalRender](/Renders/FinalRender.png)

Rendering is done with CPU and GPU to allow performance comparison and ease debugging. Most of the code is shared by the CPU and GPU versions. For ease of use on windows, I added the ability to save the rendered image as a PNG with the [_stb library_](https://github.com/nothings/stb). Download stb_image.h and stb_image_write.h and place them in an additional includes directory

![HighRefraction](/Renders/HighRefraction.png)

### Performance Optimization
I was able to improve the render time for the cover image from 63 seconds to 33 seconds (less than 3 seconds for the second image) on a GTX 1060. Some of the more impactful steps taken where:
#### Reducing the amount of global memory that must be accessed to check a hit.
-	Removing the need to check if a sphere is visible when checking for a hit either by sorting the sphere list by visibility or removing the option to generate non-visible spheres. (non-visible spheres was an option I added for generality but is not needed for this demo)
-	Separating the center and (squared) radius into its own class to make the data more contiguous. Could go a step further and try to vector load a float4 but I am not sure about potential tradeoffs with register usage.
#### Increasing resident threads by reducing register usage.
-	Switching random number generation from the stateful cuRAND library to the “stateless” PCG hash function reducing global memory access and register use.
-	Pre normalizing camera rays to simplify hit checking.
-	Using an index instead of a pointer when finding the closest sphere.
-	Eliminating accidental use of double in place of float, especially for c math functions.
-	Making use of launch bounds to cap register use when no further reduction could be found. Fortunately, this did not result in spillover, perhaps due to reductions already made.
-	Enabling the -use_fast_math compilation option reduced registers in this case.
### Recommendations for further improvement
Memory dependency is the main bottleneck at this point. The majority of computation time takes place in the hit detection code where every thread needs to read the same data and where this data is too large to fit a copy in shared memory. A potential approach might be to rearrange the kernel to use multiple threads per pixel and have threads stride over the collider list to perform hit checks. This might help scaling with more scene objects in a brute force way.
Typically, applications involving collision detection would use a hierarchical data structure to efficiently check for hits in less than linear time. While outside the scope of original tutorial it is a topic covered by the series. It would be interesting to see how this approach could be adapted for a parallel environment. A potential drawback would be that access to various parts of the data structure would be random depending on the origin and direction of the rays making it difficult to ensure orderly access but there may be clever ways around this.