//! Frame generation with Apple's MetalFX frame interpolation (`MTLFXFrameInterpolator`,
//! macOS 26 and later): the frame halfway between each two real ones, made by the system's
//! interpolator instead of FSR 3.1's optical flow and interpolation. On an M4 Pro it takes
//! about 3.9 ms at 2560x1440, where FSR 3.1's passes took about 8.
//!
//! MetalFX is loaded when first asked for, not linked: the game still starts on macOS 11 to
//! 25, which lack it (or its interpolator), and frame generation there stays FSR 3.1's.
//!
//! The interpolator runs in a Metal command buffer of its own, committed to wgpu's queue
//! between the frame's two submissions (the game's `FrameGen::generate`): the queue runs them in order,
//! and Metal tracks the textures' hazards across them (wgpu-hal leaves its textures tracked).
//! Its inputs are the frame's world without the hand and HUD (`color`), the last frame's
//! (`previous`, kept here), the depth and one texture of motion vectors (`motion.wgsl`,
//! the camera's and what moved on its own); the hand and HUD are drawn over its output as
//! over the FSR 3.1 one.

use objc2::{
    Message, msg_send,
    rc::Retained,
    runtime::{AnyClass, AnyObject, Bool, ProtocolObject},
};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandQueue, MTLDevice, MTLPixelFormat, MTLStorageMode, MTLTexture,
    MTLTextureDescriptor, MTLTextureType, MTLTextureUsage,
};
use std::sync::OnceLock;
use wgpu::hal::api::Metal;

const MOTION: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;

/// Loads MetalFX once; whether its frame interpolator is there (macOS 26 and later).
fn framework() -> bool {
    static LOADED: OnceLock<bool> = OnceLock::new();
    *LOADED.get_or_init(|| {
        // SAFETY: a constant path; the handle is kept for the process (never closed).
        let handle = unsafe {
            libc::dlopen(
                c"/System/Library/Frameworks/MetalFX.framework/MetalFX".as_ptr(),
                libc::RTLD_LAZY | libc::RTLD_LOCAL,
            )
        };
        !handle.is_null() && AnyClass::get(c"MTLFXFrameInterpolatorDescriptor").is_some()
    })
}

fn descriptor_class() -> Option<&'static AnyClass> {
    framework()
        .then(|| AnyClass::get(c"MTLFXFrameInterpolatorDescriptor"))
        .flatten()
}

/// Whether MetalFX frame interpolation runs on `device`: Metal, macOS 26 or later, and a
/// GPU the interpolator supports (`supportsDevice:`).
pub fn supported(device: &wgpu::Device) -> bool {
    let Some(class) = descriptor_class() else {
        return false;
    };
    // SAFETY: `as_hal` only reads the device; `supportsDevice:` is a class method taking
    // an `id<MTLDevice>` and returning BOOL.
    unsafe {
        device.as_hal::<Metal>().is_some_and(|hal| {
            let raw: &ProtocolObject<dyn MTLDevice> = hal.raw_device();
            let ok: Bool = msg_send![class, supportsDevice: raw];
            ok.as_bool()
        })
    }
}

fn pixel_format(format: wgpu::TextureFormat) -> Option<MTLPixelFormat> {
    Some(match format {
        wgpu::TextureFormat::Bgra8Unorm => MTLPixelFormat::BGRA8Unorm,
        wgpu::TextureFormat::Rgba8Unorm => MTLPixelFormat::RGBA8Unorm,
        wgpu::TextureFormat::Rgb10a2Unorm => MTLPixelFormat::RGB10A2Unorm,
        wgpu::TextureFormat::Rgba16Float => MTLPixelFormat::RGBA16Float,
        wgpu::TextureFormat::Rgba16Unorm => MTLPixelFormat::RGBA16Unorm,
        _ => return None,
    })
}

/// The frame's camera and timing, for one interpolation.
pub struct Frame<'a> {
    /// This frame's world without the hand and HUD, at the screen's size.
    pub color: &'a wgpu::Texture,
    /// This frame's depth (`Depth32Float`, reversed) at `render_size`.
    pub depth: &'a wgpu::TextureView,
    pub object_motion: &'a wgpu::TextureView,
    pub reproject: glam::Mat4,
    pub render_size: [u32; 2],
    pub jitter: [f32; 2],
    pub near: f32,
    pub far: f32,
    pub fov_y: f32,
    /// Seconds between real frames.
    pub delta: f32,
    pub reset: bool,
}

/// MetalFX's interpolator for one screen and render size, and what it reads and writes.
pub struct Interpolator {
    raw: Retained<AnyObject>,
    size: [u32; 2],
    render_size: [u32; 2],
    format: wgpu::TextureFormat,
    motion: wgpu::Texture,
    motion_view: wgpu::TextureView,
    previous: wgpu::Texture,
    has_previous: bool,
    output: wgpu::Texture,
    output_view: wgpu::TextureView,
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    reproject: [[f32; 4]; 4],
    render_size: [f32; 2],
    jitter: [f32; 2],
}

impl Interpolator {
    /// MetalFX's interpolator from `render_size` (depth and motion) to `size` (the world and
    /// the made frame, in `format`), or `None` where it can't be made: no MetalFX, a format
    /// it doesn't take, or the system declined.
    pub fn new(
        device: &wgpu::Device,
        size: [u32; 2],
        render_size: [u32; 2],
        format: wgpu::TextureFormat,
    ) -> Option<Self> {
        let class = descriptor_class()?;
        let color_format = pixel_format(format)?;
        // SAFETY: Objective-C calls with the argument and result types MetalFX declares
        // (`MTLFXFrameInterpolatorDescriptor`, `MTLFXFrameInterpolator`); the device is
        // wgpu's, alive for this call.
        let (raw, output_usage) = unsafe {
            let hal = device.as_hal::<Metal>()?;
            let metal: &ProtocolObject<dyn MTLDevice> = hal.raw_device();
            let descriptor: Retained<AnyObject> = msg_send![class, new];
            let _: () = msg_send![&descriptor, setColorTextureFormat: color_format];
            let _: () = msg_send![&descriptor, setOutputTextureFormat: color_format];
            let _: () = msg_send![&descriptor, setDepthTextureFormat: MTLPixelFormat::Depth32Float];
            let _: () = msg_send![&descriptor, setMotionTextureFormat: MTLPixelFormat::RG16Float];
            let _: () = msg_send![&descriptor, setInputWidth: render_size[0] as usize];
            let _: () = msg_send![&descriptor, setInputHeight: render_size[1] as usize];
            let _: () = msg_send![&descriptor, setOutputWidth: size[0] as usize];
            let _: () = msg_send![&descriptor, setOutputHeight: size[1] as usize];
            let raw: Option<Retained<AnyObject>> =
                msg_send![&descriptor, newFrameInterpolatorWithDevice: metal];
            let raw = raw?;
            let output_usage: MTLTextureUsage = msg_send![&raw, outputTextureUsage];
            (raw, output_usage)
        };
        // The made frame, in the usage MetalFX asks for (which may include writes to a
        // format wgpu offers no storage for): made in Metal and handed to wgpu to sample.
        let output = Self::metal_texture(
            device,
            size,
            format,
            output_usage | MTLTextureUsage::ShaderRead,
            "MetalFX frame interpolation output",
        )?;
        let output_view = output.create_view(&Default::default());
        let motion = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("MetalFX motion vectors"),
            size: wgpu::Extent3d {
                width: render_size[0],
                height: render_size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: MOTION,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let motion_view = motion.create_view(&Default::default());
        let previous = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("MetalFX previous frame"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MetalFX motion vectors"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MetalFX motion vectors"),
            source: wgpu::ShaderSource::Wgsl(include_str!("motion.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MetalFX motion vectors"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("MetalFX motion vectors"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_motion"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_motion"),
                compilation_options: Default::default(),
                targets: &[Some(MOTION.into())],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MetalFX motion vectors"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Some(Self {
            raw,
            size,
            render_size,
            format,
            motion,
            motion_view,
            previous,
            has_previous: false,
            output,
            output_view,
            pipeline,
            layout,
            params,
        })
    }

    /// A texture made in Metal with `usage`, wrapped for wgpu to sample and copy.
    fn metal_texture(
        device: &wgpu::Device,
        size: [u32; 2],
        format: wgpu::TextureFormat,
        usage: MTLTextureUsage,
        label: &'static str,
    ) -> Option<wgpu::Texture> {
        let pixel = pixel_format(format)?;
        // SAFETY: a descriptor and texture made on wgpu's device, then handed to wgpu with
        // the format, type and size they were made with.
        unsafe {
            let hal_texture = {
                let hal = device.as_hal::<Metal>()?;
                let metal: &ProtocolObject<dyn MTLDevice> = hal.raw_device();
                let descriptor =
                    MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                        pixel,
                        size[0] as usize,
                        size[1] as usize,
                        false,
                    );
                descriptor.setUsage(usage);
                descriptor.setStorageMode(MTLStorageMode::Private);
                let raw = metal.newTextureWithDescriptor(&descriptor)?;
                wgpu::hal::metal::Device::texture_from_raw(
                    raw,
                    format,
                    MTLTextureType::Type2D,
                    1,
                    1,
                    wgpu::hal::CopyExtent {
                        width: size[0],
                        height: size[1],
                        depth: 1,
                    },
                    None,
                )
            };
            Some(device.create_texture_from_hal::<Metal>(
                hal_texture,
                &wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: size[0],
                        height: size[1],
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                },
                wgpu::TextureUses::RESOURCE,
            ))
        }
    }

    /// Whether this interpolator was made for these sizes and format.
    pub fn fits(&self, size: [u32; 2], render_size: [u32; 2], format: wgpu::TextureFormat) -> bool {
        self.size == size && self.render_size == render_size && self.format == format
    }

    /// The made frame, once `interpolate` ran.
    pub fn output(&self) -> &wgpu::TextureView {
        &self.output_view
    }

    /// Records the motion vectors into `encoder`, has `split` submit it (with everything
    /// recorded before; it leaves a new encoder in its place) and runs the interpolator
    /// after that on the queue, then keeps this frame's world for the next. Returns false
    /// when there was nothing to interpolate from (the first frame or a reset): the output
    /// then holds nothing new.
    pub fn interpolate(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        frame: &Frame<'_>,
        split: &mut dyn FnMut(&mut wgpu::CommandEncoder),
    ) -> bool {
        queue.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&Params {
                reproject: frame.reproject.to_cols_array_2d(),
                render_size: frame.render_size.map(|v| v as f32),
                jitter: frame.jitter,
            }),
        );
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MetalFX motion vectors"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(frame.depth),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(frame.object_motion),
                },
            ],
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("MetalFX motion vectors"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.motion_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.draw(0..3, 0..1);
        }
        let interpolate = self.has_previous && !frame.reset;
        // What the interpolator reads must be on the queue before its command buffer.
        split(encoder);
        if interpolate {
            // SAFETY: the textures are alive (owned here or by the frame) and in the formats
            // and usages the interpolator was made for; the command buffer is committed to
            // wgpu's own queue, after the submission that wrote its inputs.
            unsafe { self.encode(queue, frame) };
        }
        // This frame is the next one's previous: copied after the interpolation read it.
        encoder.copy_texture_to_texture(
            frame.color.as_image_copy(),
            self.previous.as_image_copy(),
            wgpu::Extent3d {
                width: self.size[0],
                height: self.size[1],
                depth_or_array_layers: 1,
            },
        );
        self.has_previous = true;
        interpolate
    }

    unsafe fn encode(&self, queue: &wgpu::Queue, frame: &Frame<'_>) {
        let raw_texture =
            |texture: &wgpu::Texture| -> Option<Retained<ProtocolObject<dyn MTLTexture>>> {
                // SAFETY: reads the texture's Metal object, which outlives this call.
                unsafe { texture.as_hal::<Metal>() }.map(|t| t.raw_handle().retain())
            };
        let (Some(color), Some(previous), Some(depth), Some(motion), Some(output)) = (
            raw_texture(frame.color),
            raw_texture(&self.previous),
            raw_texture(frame.depth.texture()),
            raw_texture(&self.motion),
            raw_texture(&self.output),
        ) else {
            return;
        };
        let Some(hal_queue) = (unsafe { queue.as_hal::<Metal>() }) else {
            return;
        };
        let Some(command_buffer) = hal_queue.as_raw().commandBuffer() else {
            return;
        };
        command_buffer.setLabel(Some(&objc2_foundation::NSString::from_str(
            "MetalFX frame interpolation",
        )));
        let raw = &self.raw;
        let [w, h] = frame.render_size.map(|v| v as f32);
        // SAFETY: setters and `encodeToCommandBuffer:` as `MTLFXFrameInterpolator` declares.
        unsafe {
            let _: () = msg_send![raw, setColorTexture: &*color];
            let _: () = msg_send![raw, setPrevColorTexture: &*previous];
            let _: () = msg_send![raw, setDepthTexture: &*depth];
            let _: () = msg_send![raw, setMotionTexture: &*motion];
            let _: () = msg_send![raw, setOutputTexture: &*output];
            // UV to render pixels.
            let _: () = msg_send![raw, setMotionVectorScaleX: w];
            let _: () = msg_send![raw, setMotionVectorScaleY: h];
            let _: () = msg_send![raw, setDeltaTime: frame.delta];
            let _: () = msg_send![raw, setNearPlane: frame.near];
            let _: () = msg_send![raw, setFarPlane: frame.far];
            let _: () = msg_send![raw, setFieldOfView: frame.fov_y.to_degrees()];
            let _: () = msg_send![raw, setAspectRatio: w / h];
            let _: () = msg_send![raw, setJitterOffsetX: frame.jitter[0]];
            let _: () = msg_send![raw, setJitterOffsetY: frame.jitter[1]];
            let _: () = msg_send![raw, setDepthReversed: Bool::YES];
            let _: () = msg_send![raw, setShouldResetHistory: Bool::new(frame.reset)];
            let _: () = msg_send![raw, encodeToCommandBuffer: &*command_buffer];
        }
        command_buffer.commit();
    }
}
