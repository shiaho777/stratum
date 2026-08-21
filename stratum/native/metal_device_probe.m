// metal_device_probe.m — minimal "does Metal work here?" probe (issue #22).
// Creates the default device, compiles a trivial kernel from source, runs it,
// and verifies the result. Exit 0 = Metal fully usable; distinct codes/exact
// output lines let CI logs answer *why* not.
#import <Metal/Metal.h>
#import <stdio.h>

int main(void) {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { printf("PROBE: NO_DEVICE\n"); return 1; }
    printf("PROBE: DEVICE %s\n", dev.name.UTF8String);
    printf("PROBE: maxBufferLength %llu\n", (unsigned long long)dev.maxBufferLength);

    NSString* src = @"#include <metal_stdlib>\nusing namespace metal;\n"
                     "kernel void add_one(device float* b [[buffer(0)]], uint i [[thread_position_in_grid]]) { b[i] += 1.0f; }";
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { printf("PROBE: COMPILE_FAIL %s\n", err.localizedDescription.UTF8String); return 2; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"add_one"];
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!ps) { printf("PROBE: PIPELINE_FAIL %s\n", err.localizedDescription.UTF8String); return 3; }

    id<MTLCommandQueue> q = [dev newCommandQueue];
    float data[4] = {1, 2, 3, 4};
    id<MTLBuffer> buf = [dev newBufferWithBytes:data length:sizeof(data) options:0];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:ps];
    [e setBuffer:buf offset:0 atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    const float* r = (const float*)buf.contents;
    printf("PROBE: RESULT %g %g %g %g\n", r[0], r[1], r[2], r[3]);
    return (r[0] == 2 && r[1] == 3 && r[2] == 4 && r[3] == 5) ? 0 : 4;
}
