// MetalFX frame interpolation's motion vectors (metalfx.rs): one texture at the render size,
// what moved on its own (`motion.wgsl`, FSR's convention) where it has a vector, else the
// camera's, from the depth and the reprojection to the previous frame. FSR's prepare pass
// (`interpolation.wgsl`'s `camera_motion_vector`) without its 3x3 dilation, which MetalFX
// does itself. In UV, previous minus current, y down: `metalfx.rs` scales it to pixels.

struct Params {
    // This frame's unjittered clip space to the previous frame's.
    reproject: mat4x4<f32>,
    render_size: vec2<f32>,
    // This frame's sub-pixel jitter, in render pixels.
    jitter: vec2<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var depth: texture_depth_2d;
@group(0) @binding(2) var object_motion: texture_2d<f32>;

@vertex fn vs_motion(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment fn fs_motion(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(at.xy);
    let object = textureLoad(object_motion, p, 0);
    if object.z > 0.5 {
        return vec4(object.xy, 0.0, 0.0);
    }
    let d = textureLoad(depth, p, 0);
    let uv = (vec2<f32>(p) + 0.5) / params.render_size - params.jitter / params.render_size;
    let ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let previous = params.reproject * vec4(ndc, d, 1.0);
    let previous_uv = vec2(0.5, -0.5) * previous.xy / previous.w + 0.5;
    return vec4(previous_uv - uv, 0.0, 0.0);
}
