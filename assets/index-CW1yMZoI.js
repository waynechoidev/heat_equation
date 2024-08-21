var z=Object.defineProperty;var V=(e,t,i)=>t in e?z(e,t,{enumerable:!0,configurable:!0,writable:!0,value:i}):e[t]=i;var c=(e,t,i)=>V(e,typeof t!="symbol"?t+"":t,i);(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const n of document.querySelectorAll('link[rel="modulepreload"]'))r(n);new MutationObserver(n=>{for(const a of n)if(a.type==="childList")for(const s of a.addedNodes)s.tagName==="LINK"&&s.rel==="modulepreload"&&r(s)}).observe(document,{childList:!0,subtree:!0});function i(n){const a={};return n.integrity&&(a.integrity=n.integrity),n.referrerPolicy&&(a.referrerPolicy=n.referrerPolicy),n.crossOrigin==="use-credentials"?a.credentials="include":n.crossOrigin==="anonymous"?a.credentials="omit":a.credentials="same-origin",a}function r(n){if(n.ep)return;n.ep=!0;const a=i(n);fetch(n.href,a)}})();async function R(e){const i=await(await fetch(e)).blob();return await createImageBitmap(i,{colorSpaceConversion:"none"})}async function C(e,t){let i=e;const r=[i];let n=0;for(;n<t&&(i.width>1||i.height>1);)i=await A(i),r.push(i),n++;return r}async function A(e){const t=Math.max(1,e.width/2|0),i=Math.max(1,e.height/2|0),r=document.createElement("canvas");r.width=t,r.height=i;const n=r.getContext("2d");if(!n)throw new Error("Unable to get 2D context");return n.drawImage(e,0,0,t,i),createImageBitmap(r)}const L=Math.PI/180;function y(e){return e*L}var F=`const TEX_SIZE:u32 = 256;

struct VSOutput {
  @builtin(position) position: vec4f,
  @location(1) height: f32,
};

struct Vertex {
  @location(0) position: vec3f,
  @location(1) texCoord: vec2f,
};

struct MatrixUniforms {
  model: mat4x4f,
  view: mat4x4f,
  projection: mat4x4f,
};

fn getIdx(coord:vec2u) -> u32 {
    return coord.x + coord.y * TEX_SIZE;
}

@group(0) @binding(0) var<uniform> uni: MatrixUniforms;
@group(0) @binding(1) var heightMap: texture_2d<f32>;
@group(0) @binding(2) var mySampler: sampler;

@vertex fn vs(
  input: Vertex,
) -> VSOutput {
  var output: VSOutput;
  var position = input.position;
  let heightMap = textureSampleLevel(heightMap, mySampler, input.texCoord, 0);
  position.y = heightMap.r;

  output.position = uni.projection * uni.view * uni.model * vec4f(position, 1.0);
  output.height = heightMap.r;
  
  return output;
}`,N=`const TEX_SIZE:u32 = 256;

struct VSOutput {
  @builtin(position) position: vec4f,
  @location(1) height: f32,
};

struct Vertex {
  @location(0) position: vec3f,
  @location(1) texCoord: vec2f,
};

struct MatrixUniforms {
  model: mat4x4f,
  view: mat4x4f,
  projection: mat4x4f,
};

fn getIdx(coord:vec2u) -> u32 {
    return coord.x + coord.y * TEX_SIZE;
}
@group(0) @binding(1) var heightMap: texture_2d<f32>;
@group(0) @binding(2) var mySampler: sampler;

@fragment fn fs(input: VSOutput) -> @location(0) vec4f {
  let height = input.height;
  
  return vec4f(0.8, 0.9 * (1.0 - height), 0.0, 1.0);
}`,W=`const TEX_SIZE:u32 = 256;

struct VSOutput {
  @builtin(position) position: vec4f,
  @location(1) height: f32,
};

struct Vertex {
  @location(0) position: vec3f,
  @location(1) texCoord: vec2f,
};

struct MatrixUniforms {
  model: mat4x4f,
  view: mat4x4f,
  projection: mat4x4f,
};

fn getIdx(coord:vec2u) -> u32 {
    return coord.x + coord.y * TEX_SIZE;
}
@group(0) @binding(0) var<storage, read_write> srcBuffer: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let x = id.x;
    let y = id.y;
    let idx = getIdx(id.xy);

    let texture_width: u32 = TEX_SIZE;
    let texture_height: u32 = TEX_SIZE;
    
    let center_x: u32 = u32(texture_width / 2u);
    let center_y: u32 = u32(texture_height / 2u);
    
    let dx: u32 = u32(x) - center_x;
    let dy: u32 = u32(y) - center_y;
    let distance_sq: u32 = dx * dx + dy * dy;
    
    let radius_sq: u32 = 10 * TEX_SIZE;
    
    if (x < texture_width && y < texture_height && distance_sq < radius_sq) {
        srcBuffer[idx] = 1.0;
    } else {
        srcBuffer[idx] = 0.0;
    }
}`,Y=`const TEX_SIZE:u32 = 256;

struct VSOutput {
  @builtin(position) position: vec4f,
  @location(1) height: f32,
};

struct Vertex {
  @location(0) position: vec3f,
  @location(1) texCoord: vec2f,
};

struct MatrixUniforms {
  model: mat4x4f,
  view: mat4x4f,
  projection: mat4x4f,
};

fn getIdx(coord:vec2u) -> u32 {
    return coord.x + coord.y * TEX_SIZE;
}
@group(0) @binding(0) var<storage, read_write> srcBuffer: array<f32>;
@group(0) @binding(1) var<storage, read_write> tempBuffer: array<f32>;
@group(0) @binding(2) var<storage, read_write> divergence: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let x = id.x;
    let y = id.y;
    let idx = getIdx(id.xy);

    let left = vec2u(clamp(x-1, 0, TEX_SIZE), y);
    let right = vec2u(clamp(x+1, 0, TEX_SIZE), y);
    let up = vec2u(x, clamp(y+1, 0, TEX_SIZE));
    let down = vec2u(x, clamp(y-1, 0, TEX_SIZE));

    divergence[idx] = (srcBuffer[getIdx(right)] + srcBuffer[getIdx(left)] + srcBuffer[getIdx(up)] + srcBuffer[getIdx(down)] - 4 * srcBuffer[idx]) * 0.25;
    tempBuffer[idx] = srcBuffer[idx];
}`,q=`const TEX_SIZE:u32 = 256;

struct VSOutput {
  @builtin(position) position: vec4f,
  @location(1) height: f32,
};

struct Vertex {
  @location(0) position: vec3f,
  @location(1) texCoord: vec2f,
};

struct MatrixUniforms {
  model: mat4x4f,
  view: mat4x4f,
  projection: mat4x4f,
};

fn getIdx(coord:vec2u) -> u32 {
    return coord.x + coord.y * TEX_SIZE;
}
@group(0) @binding(0) var<storage, read_write> targetBuffer: array<f32>;
@group(0) @binding(1) var<storage, read_write> tempBuffer: array<f32>;
@group(0) @binding(2) var<storage, read_write> divergence: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let x = id.x;
    let y = id.y;
    let idx = getIdx(id.xy);

    let left = vec2u(clamp(x-1, 0, TEX_SIZE), y);
    let right = vec2u(clamp(x+1, 0, TEX_SIZE), y);
    let up = vec2u(x, clamp(y+1, 0, TEX_SIZE));
    let down = vec2u(x, clamp(y-1, 0, TEX_SIZE));

    let divergenceScale = 0.25;

    targetBuffer[idx] = (tempBuffer[getIdx(right)] + tempBuffer[getIdx(left)]
    + tempBuffer[getIdx(up)] + tempBuffer[getIdx(down)] - divergence[idx] * 4.0 * divergenceScale) * 0.25;
}`,H=`const TEX_SIZE:u32 = 256;

struct VSOutput {
  @builtin(position) position: vec4f,
  @location(1) height: f32,
};

struct Vertex {
  @location(0) position: vec3f,
  @location(1) texCoord: vec2f,
};

struct MatrixUniforms {
  model: mat4x4f,
  view: mat4x4f,
  projection: mat4x4f,
};

fn getIdx(coord:vec2u) -> u32 {
    return coord.x + coord.y * TEX_SIZE;
}
@group(0) @binding(0) var<storage, read> srcBuffer: array<f32>;
@group(0) @binding(1) var dstTexture: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let x = id.x;
    let y = id.y;
    let idx = getIdx(id.xy);

    let r:f32 = srcBuffer[idx];
    
    textureStore(dstTexture, vec2<i32>(i32(x), i32(y)), vec4<f32>(r, 0, 0, 255));
}`,S=1e-6,v=typeof Float32Array<"u"?Float32Array:Array;Math.hypot||(Math.hypot=function(){for(var e=0,t=arguments.length;t--;)e+=arguments[t]*arguments[t];return Math.sqrt(e)});function b(){var e=new v(16);return v!=Float32Array&&(e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[6]=0,e[7]=0,e[8]=0,e[9]=0,e[11]=0,e[12]=0,e[13]=0,e[14]=0),e[0]=1,e[5]=1,e[10]=1,e[15]=1,e}function j(e){return e[0]=1,e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[5]=1,e[6]=0,e[7]=0,e[8]=0,e[9]=0,e[10]=1,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function K(e,t,i){var r=i[0],n=i[1],a=i[2],s,o,u,l,d,f,h,p,g,_,m,x;return t===e?(e[12]=t[0]*r+t[4]*n+t[8]*a+t[12],e[13]=t[1]*r+t[5]*n+t[9]*a+t[13],e[14]=t[2]*r+t[6]*n+t[10]*a+t[14],e[15]=t[3]*r+t[7]*n+t[11]*a+t[15]):(s=t[0],o=t[1],u=t[2],l=t[3],d=t[4],f=t[5],h=t[6],p=t[7],g=t[8],_=t[9],m=t[10],x=t[11],e[0]=s,e[1]=o,e[2]=u,e[3]=l,e[4]=d,e[5]=f,e[6]=h,e[7]=p,e[8]=g,e[9]=_,e[10]=m,e[11]=x,e[12]=s*r+d*n+g*a+t[12],e[13]=o*r+f*n+_*a+t[13],e[14]=u*r+h*n+m*a+t[14],e[15]=l*r+p*n+x*a+t[15]),e}function J(e,t,i){var r=i[0],n=i[1],a=i[2];return e[0]=t[0]*r,e[1]=t[1]*r,e[2]=t[2]*r,e[3]=t[3]*r,e[4]=t[4]*n,e[5]=t[5]*n,e[6]=t[6]*n,e[7]=t[7]*n,e[8]=t[8]*a,e[9]=t[9]*a,e[10]=t[10]*a,e[11]=t[11]*a,e[12]=t[12],e[13]=t[13],e[14]=t[14],e[15]=t[15],e}function Z(e,t,i){var r=Math.sin(i),n=Math.cos(i),a=t[4],s=t[5],o=t[6],u=t[7],l=t[8],d=t[9],f=t[10],h=t[11];return t!==e&&(e[0]=t[0],e[1]=t[1],e[2]=t[2],e[3]=t[3],e[12]=t[12],e[13]=t[13],e[14]=t[14],e[15]=t[15]),e[4]=a*n+l*r,e[5]=s*n+d*r,e[6]=o*n+f*r,e[7]=u*n+h*r,e[8]=l*n-a*r,e[9]=d*n-s*r,e[10]=f*n-o*r,e[11]=h*n-u*r,e}function X(e,t,i){var r=Math.sin(i),n=Math.cos(i),a=t[0],s=t[1],o=t[2],u=t[3],l=t[8],d=t[9],f=t[10],h=t[11];return t!==e&&(e[4]=t[4],e[5]=t[5],e[6]=t[6],e[7]=t[7],e[12]=t[12],e[13]=t[13],e[14]=t[14],e[15]=t[15]),e[0]=a*n-l*r,e[1]=s*n-d*r,e[2]=o*n-f*r,e[3]=u*n-h*r,e[8]=a*r+l*n,e[9]=s*r+d*n,e[10]=o*r+f*n,e[11]=u*r+h*n,e}function k(e,t,i,r,n){var a=1/Math.tan(t/2),s;return e[0]=a/i,e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[5]=a,e[6]=0,e[7]=0,e[8]=0,e[9]=0,e[11]=-1,e[12]=0,e[13]=0,e[15]=0,n!=null&&n!==1/0?(s=1/(r-n),e[10]=(n+r)*s,e[14]=2*n*r*s):(e[10]=-1,e[14]=-2*r),e}var $=k;function Q(e,t,i,r){var n,a,s,o,u,l,d,f,h,p,g=t[0],_=t[1],m=t[2],x=r[0],P=r[1],I=r[2],G=i[0],M=i[1],U=i[2];return Math.abs(g-G)<S&&Math.abs(_-M)<S&&Math.abs(m-U)<S?j(e):(d=g-G,f=_-M,h=m-U,p=1/Math.hypot(d,f,h),d*=p,f*=p,h*=p,n=P*h-I*f,a=I*d-x*h,s=x*f-P*d,p=Math.hypot(n,a,s),p?(p=1/p,n*=p,a*=p,s*=p):(n=0,a=0,s=0),o=f*s-h*a,u=h*n-d*s,l=d*a-f*n,p=Math.hypot(o,u,l),p?(p=1/p,o*=p,u*=p,l*=p):(o=0,u=0,l=0),e[0]=n,e[1]=o,e[2]=d,e[3]=0,e[4]=a,e[5]=u,e[6]=f,e[7]=0,e[8]=s,e[9]=l,e[10]=h,e[11]=0,e[12]=-(n*g+a*_+s*m),e[13]=-(o*g+u*_+l*m),e[14]=-(d*g+f*_+h*m),e[15]=1,e)}function E(){var e=new v(3);return v!=Float32Array&&(e[0]=0,e[1]=0,e[2]=0),e}function T(e,t,i){var r=new v(3);return r[0]=e,r[1]=t,r[2]=i,r}function B(e,t,i){var r=t[0],n=t[1],a=t[2],s=i[3]*r+i[7]*n+i[11]*a+i[15];return s=s||1,e[0]=(i[0]*r+i[4]*n+i[8]*a+i[12])/s,e[1]=(i[1]*r+i[5]*n+i[9]*a+i[13])/s,e[2]=(i[2]*r+i[6]*n+i[10]*a+i[14])/s,e}(function(){var e=E();return function(t,i,r,n,a,s){var o,u;for(i||(i=3),r||(r=0),n?u=Math.min(n*i+r,t.length):u=t.length,o=r;o<u;o+=i)e[0]=t[o],e[1]=t[o+1],e[2]=t[o+2],a(e,e,s),t[o]=e[0],t[o+1]=e[1],t[o+2]=e[2];return t}})();function ee(){var e=new v(2);return v!=Float32Array&&(e[0]=0,e[1]=0),e}function D(e,t){var i=new v(2);return i[0]=e,i[1]=t,i}(function(){var e=ee();return function(t,i,r,n,a,s){var o,u;for(i||(i=2),r||(r=0),n?u=Math.min(n*i+r,t.length):u=t.length,o=r;o<u;o+=i)e[0]=t[o],e[1]=t[o+1],a(e,e,s),t[o]=e[0],t[o+1]=e[1];return t}})();class te{constructor({position:t,center:i,up:r,initialRotate:n}){c(this,"_position");c(this,"_center");c(this,"_up");c(this,"_rotate");this._position=t,this._center=i,this._up=r,this._rotate=n}get position(){const t=this.getViewRotationMatrix(),i=E();return B(i,this._position,t),i}get up(){const t=this.getViewRotationMatrix(),i=E();return B(i,this._up,t),i}getViewMatrix(){const t=b(),i=this.getViewRotationMatrix(),r=E(),n=E(),a=E();return B(r,this._position,i),B(n,this._center,i),B(a,this._up,i),Q(t,r,n,a),t}getViewRotationMatrix(){const t=b();return X(t,t,y(this._rotate[1])),Z(t,t,y(this._rotate[0])),t}}class ie{constructor(){c(this,"_device");c(this,"_canvasContext");c(this,"_commandEncoder");c(this,"WIDTH");c(this,"HEIGHT");c(this,"_previousFrameTime");this.WIDTH=window.innerWidth,this.HEIGHT=window.innerHeight,this._previousFrameTime=0}async requestDevice(){var i;const t=await((i=navigator.gpu)==null?void 0:i.requestAdapter());this._device=await(t==null?void 0:t.requestDevice()),this._device||(console.error("Cannot find a device"),alert("Your device does not support WebGPU"))}async getCanvasContext(){const t=document.querySelector("canvas");t||console.error("Cannot find a canvas"),t.width=this.WIDTH,t.height=this.HEIGHT,this._canvasContext=t.getContext("webgpu"),this._canvasContext||console.error("Cannot find a canvas context");const i={device:this._device,format:navigator.gpu.getPreferredCanvasFormat(),usage:GPUTextureUsage.RENDER_ATTACHMENT,alphaMode:"opaque"};this._canvasContext.configure(i)}async createRenderPipeline({label:t,vertexShader:i,fragmentShader:r,vertexBufferLayout:n,topology:a="triangle-list",bindGroupLayouts:s}){const o={label:t,layout:s?this._device.createPipelineLayout({bindGroupLayouts:s}):"auto",vertex:{module:this._device.createShaderModule({label:`${t} vertex shader`,code:i}),buffers:n},fragment:{module:this._device.createShaderModule({label:`${t} fragment shader`,code:r}),targets:[{format:navigator.gpu.getPreferredCanvasFormat()}]},primitive:{topology:a,cullMode:"back"},depthStencil:{depthWriteEnabled:!0,depthCompare:"less-equal",format:"depth24plus"}};return this._device.createRenderPipeline(o)}async createComputePipeline({label:t,computeShader:i}){const r={label:t,layout:"auto",compute:{module:this._device.createShaderModule({label:`${t} compute shader`,code:i})}};return this._device.createComputePipeline(r)}async createCubemapTexture(t,i=0){const r=await Promise.all(t.map(R)),n=this._device.createTexture({label:"yellow F on red",size:[r[0].width,r[0].height,r.length],mipLevelCount:i+1,format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});n||console.error("Failed to load cubemap texture");for(let a=0;a<6;a++)(await C(r[a],i)).forEach((o,u)=>{this._device.queue.copyExternalImageToTexture({source:o,flipY:!1},{texture:n,origin:[0,0,a],mipLevel:u},{width:o.width,height:o.height})});return n}async createTexture(t,i=0){const r=await R(t),n=await C(r,i),a=this._device.createTexture({label:"yellow F on red",size:[n[0].width,n[0].height],mipLevelCount:n.length,format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});return a||console.error("Failed to load texture"),n.forEach((s,o)=>{this._device.queue.copyExternalImageToTexture({source:s,flipY:!1},{texture:a,mipLevel:o},{width:s.width,height:s.height})}),a}getVerticesData(t){const i=[];for(let r=0;r<t.length;r++){const{position:n,texCoord:a}=t[r];i.push(...n,...a)}return i}async getRenderPassDesc(){const t=this._canvasContext.getCurrentTexture(),i=this._device.createTexture({size:[t.width,t.height],format:"depth24plus",usage:GPUTextureUsage.RENDER_ATTACHMENT}),r={view:t.createView(),clearValue:[1,1,1,1],loadOp:"clear",storeOp:"store"},n={view:i.createView(),depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"};return{label:"render pass",colorAttachments:[r],depthStencilAttachment:n}}async createEncoder(){this._commandEncoder=this._device.createCommandEncoder({label:"encoder"})}async submitCommandBuffer(){const t=this._commandEncoder.finish();this._device.queue.submit([t])}getDelta(){const t=performance.now(),i=t-this._previousFrameTime;return this._previousFrameTime=t,i}}function re(e){const t=[],i=[];for(let r=0;r<e;r++)for(let n=0;n<e;n++)t.push({position:T(2*r/e,0,2*n/e),texCoord:D(r/e,n/e)});for(let r=0;r<e-1;r++)for(let n=0;n<e-1;n++)i.push([r+n*e,r+1+n*e,r+(n+1)*e],[r+1+n*e,r+1+(n+1)*e,r+(n+1)*e]);return{vertices:t,indices:i,length:i.length*3}}class ne extends ie{constructor(){super();c(this,"_mainPipeline");c(this,"_computeInitializePipeline");c(this,"_computeDivergencePipeline");c(this,"_computeJacobiPipeline");c(this,"_computeTexturePipeline");c(this,"_vertexBuffer");c(this,"_indexBuffer");c(this,"_indicesLength");c(this,"_matrixUniformBuffer");c(this,"_heightMapStorageBuffer");c(this,"_heightMapTempStorageBuffer");c(this,"_divergenceStorageBuffer");c(this,"_heightMapTexture");c(this,"_sampler");c(this,"_mainBindGroup");c(this,"_computeInitializeBindGroup");c(this,"_computeDivergenceBindGroup");c(this,"_computeJacobiBindGroupOdd");c(this,"_computeJacobiBindGroupEven");c(this,"_computeTextureBindGroup");c(this,"_model");c(this,"_camera");c(this,"_projection");c(this,"TEX_SIZE",256);c(this,"WORKGROUP_SIZE",16);c(this,"_play");this._play=!1}async initialize(){await this.requestDevice(),await this.getCanvasContext(),await this.createPipelines(),await this.createVertexBuffers(),await this.createOtherBuffers(),await this.createTextures(),await this.createBindGroups(),await this.computeInitialize(),this.setMatrix()}async run(){await this.writeBuffers(),await this.createEncoder(),this._play&&await this.update(),await this.draw(),await this.submitCommandBuffer(),requestAnimationFrame(()=>this.run())}play(){this._play=!0}async createPipelines(){this._mainPipeline=await this.createRenderPipeline({label:"main pipeline",vertexShader:F,fragmentShader:N,vertexBufferLayout:[{arrayStride:5*Float32Array.BYTES_PER_ELEMENT,attributes:[{shaderLocation:0,offset:0,format:"float32x3"},{shaderLocation:1,offset:3*Float32Array.BYTES_PER_ELEMENT,format:"float32x2"}]}]}),this._computeInitializePipeline=await this.createComputePipeline({label:"initialize compute pipeline",computeShader:W}),this._computeDivergencePipeline=await this.createComputePipeline({label:"divergence compute pipeline",computeShader:Y}),this._computeJacobiPipeline=await this.createComputePipeline({label:"jacobi compute pipeline",computeShader:q}),this._computeTexturePipeline=await this.createComputePipeline({label:"texture compute pipeline",computeShader:H})}async createVertexBuffers(){const i=re(this.TEX_SIZE),r=new Float32Array(this.getVerticesData(i.vertices));this._vertexBuffer=this._device.createBuffer({label:"surface vertex buffer",size:r.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST}),this._device.queue.writeBuffer(this._vertexBuffer,0,r);const n=new Uint32Array(i.indices.flat());this._indicesLength=i.length,this._indexBuffer=this._device.createBuffer({label:"surface index buffer",size:n.byteLength,usage:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST}),this._device.queue.writeBuffer(this._indexBuffer,0,n)}async createOtherBuffers(){this._heightMapStorageBuffer=this._device.createBuffer({label:"height map storage buffer",size:this.TEX_SIZE*this.TEX_SIZE*Float32Array.BYTES_PER_ELEMENT,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});const i=new Float32Array(this.TEX_SIZE*this.TEX_SIZE);this._device.queue.writeBuffer(this._heightMapStorageBuffer,0,i),this._heightMapTempStorageBuffer=this._device.createBuffer({label:"height map temp storage buffer",size:this.TEX_SIZE*this.TEX_SIZE*Float32Array.BYTES_PER_ELEMENT,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),this._device.queue.writeBuffer(this._heightMapTempStorageBuffer,0,i),this._divergenceStorageBuffer=this._device.createBuffer({label:"divergence storage buffer",size:this.TEX_SIZE*this.TEX_SIZE*Float32Array.BYTES_PER_ELEMENT,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),this._device.queue.writeBuffer(this._divergenceStorageBuffer,0,i),this._matrixUniformBuffer=this._device.createBuffer({label:"matrix uniforms",size:48*Float32Array.BYTES_PER_ELEMENT,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST})}async createTextures(){this._heightMapTexture=this._device.createTexture({label:"height map texture",size:[this.TEX_SIZE,this.TEX_SIZE],format:"rgba8unorm",usage:GPUTextureUsage.COPY_DST|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING}),this._sampler=this._device.createSampler({magFilter:"linear",minFilter:"linear",mipmapFilter:"linear"})}async createBindGroups(){this._mainBindGroup=this._device.createBindGroup({label:"bind group for object",layout:this._mainPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this._matrixUniformBuffer}},{binding:1,resource:this._heightMapTexture.createView()},{binding:2,resource:this._sampler}]}),this._computeInitializeBindGroup=this._device.createBindGroup({label:"compute initialize bind group",layout:this._computeInitializePipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this._heightMapStorageBuffer}}]}),this._computeDivergenceBindGroup=this._device.createBindGroup({label:"compute divergence bind group",layout:this._computeDivergencePipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this._heightMapStorageBuffer}},{binding:1,resource:{buffer:this._heightMapTempStorageBuffer}},{binding:2,resource:{buffer:this._divergenceStorageBuffer}}]}),this._computeJacobiBindGroupOdd=this._device.createBindGroup({label:"compute jacobi bind group odd",layout:this._computeJacobiPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this._heightMapTempStorageBuffer}},{binding:1,resource:{buffer:this._heightMapStorageBuffer}},{binding:2,resource:{buffer:this._divergenceStorageBuffer}}]}),this._computeJacobiBindGroupEven=this._device.createBindGroup({label:"compute jacobi bind group even",layout:this._computeJacobiPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this._heightMapStorageBuffer}},{binding:1,resource:{buffer:this._heightMapTempStorageBuffer}},{binding:2,resource:{buffer:this._divergenceStorageBuffer}}]}),this._computeTextureBindGroup=this._device.createBindGroup({label:"compute movement bind group",layout:this._computeTexturePipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this._heightMapStorageBuffer}},{binding:1,resource:this._heightMapTexture.createView()}]})}setMatrix(){this._model=b();const i=this.WIDTH>500?.4:.3;J(this._model,this._model,T(i,i,i)),K(this._model,this._model,T(-.5,.2,0)),Z(this._model,this._model,y(30)),X(this._model,this._model,y(-15)),this._camera=new te({position:T(0,0,2.5),center:T(0,0,0),up:T(0,1,0),initialRotate:D(0,0)}),this._projection=b(),$(this._projection,y(45),this.WIDTH/this.HEIGHT,.1,100)}async writeBuffers(){const i=this._camera.getViewMatrix();this._device.queue.writeBuffer(this._matrixUniformBuffer,0,new Float32Array([...this._model,...i,...this._projection]))}async computeInitialize(){await this.createEncoder();const i=this._commandEncoder.beginComputePass({label:"compute initialize pass"});i.setPipeline(this._computeInitializePipeline),i.setBindGroup(0,this._computeInitializeBindGroup),i.dispatchWorkgroups(this.TEX_SIZE/this.WORKGROUP_SIZE,this.TEX_SIZE/this.WORKGROUP_SIZE,1),i.setPipeline(this._computeTexturePipeline),i.setBindGroup(0,this._computeTextureBindGroup),i.dispatchWorkgroups(this.TEX_SIZE/this.WORKGROUP_SIZE,this.TEX_SIZE/this.WORKGROUP_SIZE,1),i.end(),await this.submitCommandBuffer()}async draw(){const i=await this.getRenderPassDesc(),r=this._commandEncoder.beginRenderPass(i);r.setPipeline(this._mainPipeline),r==null||r.setBindGroup(0,this._mainBindGroup),r.setVertexBuffer(0,this._vertexBuffer),r.setIndexBuffer(this._indexBuffer,"uint32"),r.drawIndexed(this._indicesLength),r.end()}async update(){const i=this._commandEncoder.beginComputePass({label:"compute pass"});i.setPipeline(this._computeDivergencePipeline),i.setBindGroup(0,this._computeDivergenceBindGroup),i.dispatchWorkgroups(this.TEX_SIZE/this.WORKGROUP_SIZE,this.TEX_SIZE/this.WORKGROUP_SIZE,1),i.setPipeline(this._computeJacobiPipeline);for(let r=0;r<40;r++)r%2==0?i.setBindGroup(0,this._computeJacobiBindGroupOdd):i.setBindGroup(0,this._computeJacobiBindGroupEven),i.dispatchWorkgroups(this.TEX_SIZE/this.WORKGROUP_SIZE,this.TEX_SIZE/this.WORKGROUP_SIZE,1);i.setPipeline(this._computeTexturePipeline),i.setBindGroup(0,this._computeTextureBindGroup),i.dispatchWorkgroups(this.TEX_SIZE/this.WORKGROUP_SIZE,this.TEX_SIZE/this.WORKGROUP_SIZE,1),i.end()}}const w=new ne;async function ae(){await w.initialize(),await w.run()}ae();const O=document.getElementById("click");O.addEventListener("click",()=>{w.play(),O.hidden=!0});
