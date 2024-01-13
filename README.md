# CUDA Ray Tracer

### A GPU ray tracing renderer written in CUDA and C++ following the [_Ray Tracing in One Weekend_](https://raytracing.github.io/books/RayTracingInOneWeekend.html) tutorial by Peter Shirley supplemented with the NVIDIA Technical Blog [_Accelerated Ray Tracing in One Weekend in CUDA_](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/) by Roger Allen

![FinalRender](/Renders/FinalRender.png)

Rendering is done with CPU and GPU to allow performance comparison and ease debugging. Most of the code is shared by the CPU and GPU versions. For ease of use on windows, I added the ability to save the rendered image as a PNG with the [_stb library_](https://github.com/nothings/stb). Download stb_image.h and stb_image_write.h and place them in an additional includes directory

![HighRefraction](/Renders/HighRefraction.png)