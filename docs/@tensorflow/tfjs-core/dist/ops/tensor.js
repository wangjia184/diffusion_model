/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import { inferShape } from '../tensor_util_env';
import { makeTensor } from './tensor_ops_util';
/**
 * Creates a `tf.Tensor` with the provided values, shape and dtype.
 *
 * ```js
 * // Pass an array of values to create a vector.
 * tf.tensor([1, 2, 3, 4]).print();
 * ```
 *
 * ```js
 * // Pass a nested array of values to make a matrix or a higher
 * // dimensional tensor.
 * tf.tensor([[1, 2], [3, 4]]).print();
 * ```
 *
 * ```js
 * // Pass a flat array and specify a shape yourself.
 * tf.tensor([1, 2, 3, 4], [2, 2]).print();
 * ```
 *
 * ```js
 * // Pass a `WebGLData` object and specify a shape yourself.
 *
 * // This makes it possible for TF.js applications to avoid GPU / CPU sync.
 * // For example, if your application includes a preprocessing step on the GPU,
 * // you could upload the GPU output directly to TF.js, rather than first
 * // downloading the values.
 *
 * // Example for WebGL2:
 * const customCanvas = document.createElement('canvas');
 * const customBackend = new tf.MathBackendWebGL(customCanvas);
 * tf.registerBackend('custom-webgl', () => customBackend);
 * await tf.setBackend('custom-webgl');
 * const gl = customBackend.gpgpu.gl;
 * const texture = gl.createTexture();
 * const tex2d = gl.TEXTURE_2D;
 * const width = 2;
 * const height = 2;
 *
 * gl.bindTexture(tex2d, texture);
 * gl.texParameteri(tex2d, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
 * gl.texParameteri(tex2d, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
 * gl.texParameteri(tex2d, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
 * gl.texParameteri(tex2d, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
 * gl.texImage2D(
 *   tex2d, 0, gl.RGBA32F, // internalFormat
 *   width, height, 0,
 *   gl.RGBA, // textureFormat
 *   gl.FLOAT, // textureType
 *   new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
 * );
 *
 * // Currently, the `texture` has 4 pixels:
 * // Pixel0 is {R:0, G:1, B:2, A:3}
 * // Pixel1 is {R:4, G:5, B:6, A:7}
 * // Pixel2 is {R:8, G:9, B:10, A:11}
 * // Pixel3 is {R:12, G:13, B:14, A:15}
 *
 * const logicalShape = [height * width * 2];
 * const a = tf.tensor({texture, height, width, channels: 'BR'}, logicalShape);
 * // Tensor value will be [2, 0, 6, 4, 10, 8, 14, 12], since [2, 0] is the
 * // values of 'B' and 'R' channels of Pixel0, [6, 4] is the values of 'B' and
 * 'R'
 * // channels of Pixel1...
 *
 * // For postprocessing on the GPU, it's possible to retrieve the texture
 * // backing any tensor by calling the tensor's `dataToGPU` method like
 * // so:
 *
 * const tex = a.dataToGPU();
 * ```
 *
 * ```js
 * // Pass a `WebGPUData` object and specify a shape yourself.
 *
 * // This makes it possible for TF.js applications to avoid GPU / CPU sync.
 * // For example, if your application includes a preprocessing step on the GPU,
 * // you could upload the GPU output directly to TF.js, rather than first
 * // downloading the values. Unlike WebGL, this optionally supports zero copy
 * // by WebGPUData.zeroCopy. When zeroCopy is false or undefined(default), this
 * // passing GPUBuffer can be destroyed after tensor is created. When zeroCopy
 * // is true, this GPUBuffer is bound directly by the tensor, so do not destroy
 * // this GPUBuffer until all access is done.
 *
 * // Example for WebGPU:
 * function createGPUBufferFromData(device, data, dtype) {
 *   const bytesPerElement = 4;
 *   const sizeInBytes = data.length * bytesPerElement;
 *
 *   const gpuWriteBuffer = device.createBuffer({
 *     mappedAtCreation: true,
 *     size: sizeInBytes,
 *     usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC
 *   });
 *   const arrayBuffer = gpuWriteBuffer.getMappedRange();
 *   if (dtype === 'float32') {
 *     new Float32Array(arrayBuffer).set(data);
 *   } else if (dtype === 'int32') {
 *     new Int32Array(arrayBuffer).set(data);
 *   } else {
 *     throw new Error(
 *         `Creating tensor from GPUBuffer only supports` +
 *         `'float32'|'int32' dtype, while the dtype is ${dtype}.`);
 *   }
 *   gpuWriteBuffer.unmap();
 *
 *   const gpuReadBuffer = device.createBuffer({
 *     mappedAtCreation: false,
 *     size: sizeInBytes,
 *     usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE |
 *         GPUBufferUsage.COPY_SRC
 *   });
 *
 *   const copyEncoder = device.createCommandEncoder();
 *   copyEncoder.copyBufferToBuffer(
 *       gpuWriteBuffer, 0, gpuReadBuffer, 0, sizeInBytes);
 *   const copyCommands = copyEncoder.finish();
 *   device.queue.submit([copyCommands]);
 *   gpuWriteBuffer.destroy();
 *   return gpuReadBuffer;
 * }
 *
 * const dtype = 'float32';
 * const device = tf.backend().device;
 * const aData = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
 * const bData = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4];
 * const expected = [2, 4, 6, 8, 6, 8, 10, 12, 10, 12, 14, 16, 14, 16, 18, 20];
 * const aBuffer = createGPUBufferFromData(device, aData, dtype);
 * const shape = [aData.length];
 * // To use zeroCopy, use {buffer: aBuffer, zeroCopy: true} instead and destroy
 * // aBuffer untill all access is done.
 * const a = tf.tensor({buffer: aBuffer}, shape, dtype);
 * const b = tf.tensor(bData, shape, dtype);
 * const result = tf.add(a, b);
 * a.dispose();
 * b.dispose();
 * result.dispose();
 * aBuffer.destroy();
 * ```
 * @param values The values of the tensor. Can be nested array of numbers,
 *     or a flat array, or a `TypedArray`, or a `WebGLData` object, or a
 * `WebGPUData` object. If the values are strings, they will be encoded as utf-8
 * and kept as `Uint8Array[]`. If the values is a `WebGLData` object, the dtype
 * could only be 'float32' or 'int32' and the object has to have: 1. texture, a
 * `WebGLTexture`, the texture must share the same `WebGLRenderingContext` with
 * TFJS's WebGL backend (you could create a custom WebGL backend from your
 * texture's canvas) and the internal texture format for the input texture must
 * be floating point or normalized integer; 2. height, the height of the
 * texture; 3. width, the width of the texture; 4. channels, a non-empty subset
 * of 'RGBA', indicating the values of which channels will be passed to the
 * tensor, such as 'R' or 'BR' (The order of the channels affect the order of
 * tensor values. ). (If the values passed from texture is less than the tensor
 * size, zeros will be padded at the rear.). If the values is a `WebGPUData`
 * object, the dtype could only be 'float32' or 'int32 and the object has to
 * have: buffer, a `GPUBuffer`. The buffer must: 1. share the same `GPUDevice`
 * with TFJS's WebGPU backend; 2. buffer.usage should at least support
 * GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC; 3. buffer.size should not
 * be smaller than the byte size of tensor shape. WebGPUData optionally supports
 * zero copy by flag zeroCopy. When zeroCopy is false or undefined(default),
 * this passing GPUBuffer can be destroyed after tensor is created. When
 * zeroCopy is true, this GPUBuffer is bound directly by the tensor, so do not
 * destroy this GPUBuffer until all access is done.
 * @param shape The shape of the tensor. Optional. If not provided,
 *   it is inferred from `values`.
 * @param dtype The data type.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
export function tensor(values, shape, dtype) {
    const inferredShape = inferShape(values, dtype);
    return makeTensor(values, shape, inferredShape, dtype);
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidGVuc29yLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvdGVuc29yLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUdILE9BQU8sRUFBQyxVQUFVLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUk5QyxPQUFPLEVBQUMsVUFBVSxFQUFDLE1BQU0sbUJBQW1CLENBQUM7QUFFN0M7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7R0FzS0c7QUFDSCxNQUFNLFVBQVUsTUFBTSxDQUNsQixNQUF1QyxFQUFFLEtBQW1CLEVBQzVELEtBQWdCO0lBQ2xCLE1BQU0sYUFBYSxHQUFHLFVBQVUsQ0FBQyxNQUFNLEVBQUUsS0FBSyxDQUFDLENBQUM7SUFDaEQsT0FBTyxVQUFVLENBQUMsTUFBTSxFQUFFLEtBQUssRUFBRSxhQUFhLEVBQUUsS0FBSyxDQUFjLENBQUM7QUFDdEUsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtUZW5zb3J9IGZyb20gJy4uL3RlbnNvcic7XG5pbXBvcnQge2luZmVyU2hhcGV9IGZyb20gJy4uL3RlbnNvcl91dGlsX2Vudic7XG5pbXBvcnQge1RlbnNvckxpa2V9IGZyb20gJy4uL3R5cGVzJztcbmltcG9ydCB7RGF0YVR5cGUsIFJhbmssIFNoYXBlTWFwLCBXZWJHTERhdGEsIFdlYkdQVURhdGF9IGZyb20gJy4uL3R5cGVzJztcblxuaW1wb3J0IHttYWtlVGVuc29yfSBmcm9tICcuL3RlbnNvcl9vcHNfdXRpbCc7XG5cbi8qKlxuICogQ3JlYXRlcyBhIGB0Zi5UZW5zb3JgIHdpdGggdGhlIHByb3ZpZGVkIHZhbHVlcywgc2hhcGUgYW5kIGR0eXBlLlxuICpcbiAqIGBgYGpzXG4gKiAvLyBQYXNzIGFuIGFycmF5IG9mIHZhbHVlcyB0byBjcmVhdGUgYSB2ZWN0b3IuXG4gKiB0Zi50ZW5zb3IoWzEsIDIsIDMsIDRdKS5wcmludCgpO1xuICogYGBgXG4gKlxuICogYGBganNcbiAqIC8vIFBhc3MgYSBuZXN0ZWQgYXJyYXkgb2YgdmFsdWVzIHRvIG1ha2UgYSBtYXRyaXggb3IgYSBoaWdoZXJcbiAqIC8vIGRpbWVuc2lvbmFsIHRlbnNvci5cbiAqIHRmLnRlbnNvcihbWzEsIDJdLCBbMywgNF1dKS5wcmludCgpO1xuICogYGBgXG4gKlxuICogYGBganNcbiAqIC8vIFBhc3MgYSBmbGF0IGFycmF5IGFuZCBzcGVjaWZ5IGEgc2hhcGUgeW91cnNlbGYuXG4gKiB0Zi50ZW5zb3IoWzEsIDIsIDMsIDRdLCBbMiwgMl0pLnByaW50KCk7XG4gKiBgYGBcbiAqXG4gKiBgYGBqc1xuICogLy8gUGFzcyBhIGBXZWJHTERhdGFgIG9iamVjdCBhbmQgc3BlY2lmeSBhIHNoYXBlIHlvdXJzZWxmLlxuICpcbiAqIC8vIFRoaXMgbWFrZXMgaXQgcG9zc2libGUgZm9yIFRGLmpzIGFwcGxpY2F0aW9ucyB0byBhdm9pZCBHUFUgLyBDUFUgc3luYy5cbiAqIC8vIEZvciBleGFtcGxlLCBpZiB5b3VyIGFwcGxpY2F0aW9uIGluY2x1ZGVzIGEgcHJlcHJvY2Vzc2luZyBzdGVwIG9uIHRoZSBHUFUsXG4gKiAvLyB5b3UgY291bGQgdXBsb2FkIHRoZSBHUFUgb3V0cHV0IGRpcmVjdGx5IHRvIFRGLmpzLCByYXRoZXIgdGhhbiBmaXJzdFxuICogLy8gZG93bmxvYWRpbmcgdGhlIHZhbHVlcy5cbiAqXG4gKiAvLyBFeGFtcGxlIGZvciBXZWJHTDI6XG4gKiBjb25zdCBjdXN0b21DYW52YXMgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdjYW52YXMnKTtcbiAqIGNvbnN0IGN1c3RvbUJhY2tlbmQgPSBuZXcgdGYuTWF0aEJhY2tlbmRXZWJHTChjdXN0b21DYW52YXMpO1xuICogdGYucmVnaXN0ZXJCYWNrZW5kKCdjdXN0b20td2ViZ2wnLCAoKSA9PiBjdXN0b21CYWNrZW5kKTtcbiAqIGF3YWl0IHRmLnNldEJhY2tlbmQoJ2N1c3RvbS13ZWJnbCcpO1xuICogY29uc3QgZ2wgPSBjdXN0b21CYWNrZW5kLmdwZ3B1LmdsO1xuICogY29uc3QgdGV4dHVyZSA9IGdsLmNyZWF0ZVRleHR1cmUoKTtcbiAqIGNvbnN0IHRleDJkID0gZ2wuVEVYVFVSRV8yRDtcbiAqIGNvbnN0IHdpZHRoID0gMjtcbiAqIGNvbnN0IGhlaWdodCA9IDI7XG4gKlxuICogZ2wuYmluZFRleHR1cmUodGV4MmQsIHRleHR1cmUpO1xuICogZ2wudGV4UGFyYW1ldGVyaSh0ZXgyZCwgZ2wuVEVYVFVSRV9XUkFQX1MsIGdsLkNMQU1QX1RPX0VER0UpO1xuICogZ2wudGV4UGFyYW1ldGVyaSh0ZXgyZCwgZ2wuVEVYVFVSRV9XUkFQX1QsIGdsLkNMQU1QX1RPX0VER0UpO1xuICogZ2wudGV4UGFyYW1ldGVyaSh0ZXgyZCwgZ2wuVEVYVFVSRV9NSU5fRklMVEVSLCBnbC5ORUFSRVNUKTtcbiAqIGdsLnRleFBhcmFtZXRlcmkodGV4MmQsIGdsLlRFWFRVUkVfTUFHX0ZJTFRFUiwgZ2wuTkVBUkVTVCk7XG4gKiBnbC50ZXhJbWFnZTJEKFxuICogICB0ZXgyZCwgMCwgZ2wuUkdCQTMyRiwgLy8gaW50ZXJuYWxGb3JtYXRcbiAqICAgd2lkdGgsIGhlaWdodCwgMCxcbiAqICAgZ2wuUkdCQSwgLy8gdGV4dHVyZUZvcm1hdFxuICogICBnbC5GTE9BVCwgLy8gdGV4dHVyZVR5cGVcbiAqICAgbmV3IEZsb2F0MzJBcnJheShbMCwgMSwgMiwgMywgNCwgNSwgNiwgNywgOCwgOSwgMTAsIDExLCAxMiwgMTMsIDE0LCAxNV0pXG4gKiApO1xuICpcbiAqIC8vIEN1cnJlbnRseSwgdGhlIGB0ZXh0dXJlYCBoYXMgNCBwaXhlbHM6XG4gKiAvLyBQaXhlbDAgaXMge1I6MCwgRzoxLCBCOjIsIEE6M31cbiAqIC8vIFBpeGVsMSBpcyB7Ujo0LCBHOjUsIEI6NiwgQTo3fVxuICogLy8gUGl4ZWwyIGlzIHtSOjgsIEc6OSwgQjoxMCwgQToxMX1cbiAqIC8vIFBpeGVsMyBpcyB7UjoxMiwgRzoxMywgQjoxNCwgQToxNX1cbiAqXG4gKiBjb25zdCBsb2dpY2FsU2hhcGUgPSBbaGVpZ2h0ICogd2lkdGggKiAyXTtcbiAqIGNvbnN0IGEgPSB0Zi50ZW5zb3Ioe3RleHR1cmUsIGhlaWdodCwgd2lkdGgsIGNoYW5uZWxzOiAnQlInfSwgbG9naWNhbFNoYXBlKTtcbiAqIC8vIFRlbnNvciB2YWx1ZSB3aWxsIGJlIFsyLCAwLCA2LCA0LCAxMCwgOCwgMTQsIDEyXSwgc2luY2UgWzIsIDBdIGlzIHRoZVxuICogLy8gdmFsdWVzIG9mICdCJyBhbmQgJ1InIGNoYW5uZWxzIG9mIFBpeGVsMCwgWzYsIDRdIGlzIHRoZSB2YWx1ZXMgb2YgJ0InIGFuZFxuICogJ1InXG4gKiAvLyBjaGFubmVscyBvZiBQaXhlbDEuLi5cbiAqXG4gKiAvLyBGb3IgcG9zdHByb2Nlc3Npbmcgb24gdGhlIEdQVSwgaXQncyBwb3NzaWJsZSB0byByZXRyaWV2ZSB0aGUgdGV4dHVyZVxuICogLy8gYmFja2luZyBhbnkgdGVuc29yIGJ5IGNhbGxpbmcgdGhlIHRlbnNvcidzIGBkYXRhVG9HUFVgIG1ldGhvZCBsaWtlXG4gKiAvLyBzbzpcbiAqXG4gKiBjb25zdCB0ZXggPSBhLmRhdGFUb0dQVSgpO1xuICogYGBgXG4gKlxuICogYGBganNcbiAqIC8vIFBhc3MgYSBgV2ViR1BVRGF0YWAgb2JqZWN0IGFuZCBzcGVjaWZ5IGEgc2hhcGUgeW91cnNlbGYuXG4gKlxuICogLy8gVGhpcyBtYWtlcyBpdCBwb3NzaWJsZSBmb3IgVEYuanMgYXBwbGljYXRpb25zIHRvIGF2b2lkIEdQVSAvIENQVSBzeW5jLlxuICogLy8gRm9yIGV4YW1wbGUsIGlmIHlvdXIgYXBwbGljYXRpb24gaW5jbHVkZXMgYSBwcmVwcm9jZXNzaW5nIHN0ZXAgb24gdGhlIEdQVSxcbiAqIC8vIHlvdSBjb3VsZCB1cGxvYWQgdGhlIEdQVSBvdXRwdXQgZGlyZWN0bHkgdG8gVEYuanMsIHJhdGhlciB0aGFuIGZpcnN0XG4gKiAvLyBkb3dubG9hZGluZyB0aGUgdmFsdWVzLiBVbmxpa2UgV2ViR0wsIHRoaXMgb3B0aW9uYWxseSBzdXBwb3J0cyB6ZXJvIGNvcHlcbiAqIC8vIGJ5IFdlYkdQVURhdGEuemVyb0NvcHkuIFdoZW4gemVyb0NvcHkgaXMgZmFsc2Ugb3IgdW5kZWZpbmVkKGRlZmF1bHQpLCB0aGlzXG4gKiAvLyBwYXNzaW5nIEdQVUJ1ZmZlciBjYW4gYmUgZGVzdHJveWVkIGFmdGVyIHRlbnNvciBpcyBjcmVhdGVkLiBXaGVuIHplcm9Db3B5XG4gKiAvLyBpcyB0cnVlLCB0aGlzIEdQVUJ1ZmZlciBpcyBib3VuZCBkaXJlY3RseSBieSB0aGUgdGVuc29yLCBzbyBkbyBub3QgZGVzdHJveVxuICogLy8gdGhpcyBHUFVCdWZmZXIgdW50aWwgYWxsIGFjY2VzcyBpcyBkb25lLlxuICpcbiAqIC8vIEV4YW1wbGUgZm9yIFdlYkdQVTpcbiAqIGZ1bmN0aW9uIGNyZWF0ZUdQVUJ1ZmZlckZyb21EYXRhKGRldmljZSwgZGF0YSwgZHR5cGUpIHtcbiAqICAgY29uc3QgYnl0ZXNQZXJFbGVtZW50ID0gNDtcbiAqICAgY29uc3Qgc2l6ZUluQnl0ZXMgPSBkYXRhLmxlbmd0aCAqIGJ5dGVzUGVyRWxlbWVudDtcbiAqXG4gKiAgIGNvbnN0IGdwdVdyaXRlQnVmZmVyID0gZGV2aWNlLmNyZWF0ZUJ1ZmZlcih7XG4gKiAgICAgbWFwcGVkQXRDcmVhdGlvbjogdHJ1ZSxcbiAqICAgICBzaXplOiBzaXplSW5CeXRlcyxcbiAqICAgICB1c2FnZTogR1BVQnVmZmVyVXNhZ2UuTUFQX1dSSVRFIHwgR1BVQnVmZmVyVXNhZ2UuQ09QWV9TUkNcbiAqICAgfSk7XG4gKiAgIGNvbnN0IGFycmF5QnVmZmVyID0gZ3B1V3JpdGVCdWZmZXIuZ2V0TWFwcGVkUmFuZ2UoKTtcbiAqICAgaWYgKGR0eXBlID09PSAnZmxvYXQzMicpIHtcbiAqICAgICBuZXcgRmxvYXQzMkFycmF5KGFycmF5QnVmZmVyKS5zZXQoZGF0YSk7XG4gKiAgIH0gZWxzZSBpZiAoZHR5cGUgPT09ICdpbnQzMicpIHtcbiAqICAgICBuZXcgSW50MzJBcnJheShhcnJheUJ1ZmZlcikuc2V0KGRhdGEpO1xuICogICB9IGVsc2Uge1xuICogICAgIHRocm93IG5ldyBFcnJvcihcbiAqICAgICAgICAgYENyZWF0aW5nIHRlbnNvciBmcm9tIEdQVUJ1ZmZlciBvbmx5IHN1cHBvcnRzYCArXG4gKiAgICAgICAgIGAnZmxvYXQzMid8J2ludDMyJyBkdHlwZSwgd2hpbGUgdGhlIGR0eXBlIGlzICR7ZHR5cGV9LmApO1xuICogICB9XG4gKiAgIGdwdVdyaXRlQnVmZmVyLnVubWFwKCk7XG4gKlxuICogICBjb25zdCBncHVSZWFkQnVmZmVyID0gZGV2aWNlLmNyZWF0ZUJ1ZmZlcih7XG4gKiAgICAgbWFwcGVkQXRDcmVhdGlvbjogZmFsc2UsXG4gKiAgICAgc2l6ZTogc2l6ZUluQnl0ZXMsXG4gKiAgICAgdXNhZ2U6IEdQVUJ1ZmZlclVzYWdlLkNPUFlfRFNUIHwgR1BVQnVmZmVyVXNhZ2UuU1RPUkFHRSB8XG4gKiAgICAgICAgIEdQVUJ1ZmZlclVzYWdlLkNPUFlfU1JDXG4gKiAgIH0pO1xuICpcbiAqICAgY29uc3QgY29weUVuY29kZXIgPSBkZXZpY2UuY3JlYXRlQ29tbWFuZEVuY29kZXIoKTtcbiAqICAgY29weUVuY29kZXIuY29weUJ1ZmZlclRvQnVmZmVyKFxuICogICAgICAgZ3B1V3JpdGVCdWZmZXIsIDAsIGdwdVJlYWRCdWZmZXIsIDAsIHNpemVJbkJ5dGVzKTtcbiAqICAgY29uc3QgY29weUNvbW1hbmRzID0gY29weUVuY29kZXIuZmluaXNoKCk7XG4gKiAgIGRldmljZS5xdWV1ZS5zdWJtaXQoW2NvcHlDb21tYW5kc10pO1xuICogICBncHVXcml0ZUJ1ZmZlci5kZXN0cm95KCk7XG4gKiAgIHJldHVybiBncHVSZWFkQnVmZmVyO1xuICogfVxuICpcbiAqIGNvbnN0IGR0eXBlID0gJ2Zsb2F0MzInO1xuICogY29uc3QgZGV2aWNlID0gdGYuYmFja2VuZCgpLmRldmljZTtcbiAqIGNvbnN0IGFEYXRhID0gWzEsIDIsIDMsIDQsIDUsIDYsIDcsIDgsIDksIDEwLCAxMSwgMTIsIDEzLCAxNCwgMTUsIDE2XTtcbiAqIGNvbnN0IGJEYXRhID0gWzEsIDIsIDMsIDQsIDEsIDIsIDMsIDQsIDEsIDIsIDMsIDQsIDEsIDIsIDMsIDRdO1xuICogY29uc3QgZXhwZWN0ZWQgPSBbMiwgNCwgNiwgOCwgNiwgOCwgMTAsIDEyLCAxMCwgMTIsIDE0LCAxNiwgMTQsIDE2LCAxOCwgMjBdO1xuICogY29uc3QgYUJ1ZmZlciA9IGNyZWF0ZUdQVUJ1ZmZlckZyb21EYXRhKGRldmljZSwgYURhdGEsIGR0eXBlKTtcbiAqIGNvbnN0IHNoYXBlID0gW2FEYXRhLmxlbmd0aF07XG4gKiAvLyBUbyB1c2UgemVyb0NvcHksIHVzZSB7YnVmZmVyOiBhQnVmZmVyLCB6ZXJvQ29weTogdHJ1ZX0gaW5zdGVhZCBhbmQgZGVzdHJveVxuICogLy8gYUJ1ZmZlciB1bnRpbGwgYWxsIGFjY2VzcyBpcyBkb25lLlxuICogY29uc3QgYSA9IHRmLnRlbnNvcih7YnVmZmVyOiBhQnVmZmVyfSwgc2hhcGUsIGR0eXBlKTtcbiAqIGNvbnN0IGIgPSB0Zi50ZW5zb3IoYkRhdGEsIHNoYXBlLCBkdHlwZSk7XG4gKiBjb25zdCByZXN1bHQgPSB0Zi5hZGQoYSwgYik7XG4gKiBhLmRpc3Bvc2UoKTtcbiAqIGIuZGlzcG9zZSgpO1xuICogcmVzdWx0LmRpc3Bvc2UoKTtcbiAqIGFCdWZmZXIuZGVzdHJveSgpO1xuICogYGBgXG4gKiBAcGFyYW0gdmFsdWVzIFRoZSB2YWx1ZXMgb2YgdGhlIHRlbnNvci4gQ2FuIGJlIG5lc3RlZCBhcnJheSBvZiBudW1iZXJzLFxuICogICAgIG9yIGEgZmxhdCBhcnJheSwgb3IgYSBgVHlwZWRBcnJheWAsIG9yIGEgYFdlYkdMRGF0YWAgb2JqZWN0LCBvciBhXG4gKiBgV2ViR1BVRGF0YWAgb2JqZWN0LiBJZiB0aGUgdmFsdWVzIGFyZSBzdHJpbmdzLCB0aGV5IHdpbGwgYmUgZW5jb2RlZCBhcyB1dGYtOFxuICogYW5kIGtlcHQgYXMgYFVpbnQ4QXJyYXlbXWAuIElmIHRoZSB2YWx1ZXMgaXMgYSBgV2ViR0xEYXRhYCBvYmplY3QsIHRoZSBkdHlwZVxuICogY291bGQgb25seSBiZSAnZmxvYXQzMicgb3IgJ2ludDMyJyBhbmQgdGhlIG9iamVjdCBoYXMgdG8gaGF2ZTogMS4gdGV4dHVyZSwgYVxuICogYFdlYkdMVGV4dHVyZWAsIHRoZSB0ZXh0dXJlIG11c3Qgc2hhcmUgdGhlIHNhbWUgYFdlYkdMUmVuZGVyaW5nQ29udGV4dGAgd2l0aFxuICogVEZKUydzIFdlYkdMIGJhY2tlbmQgKHlvdSBjb3VsZCBjcmVhdGUgYSBjdXN0b20gV2ViR0wgYmFja2VuZCBmcm9tIHlvdXJcbiAqIHRleHR1cmUncyBjYW52YXMpIGFuZCB0aGUgaW50ZXJuYWwgdGV4dHVyZSBmb3JtYXQgZm9yIHRoZSBpbnB1dCB0ZXh0dXJlIG11c3RcbiAqIGJlIGZsb2F0aW5nIHBvaW50IG9yIG5vcm1hbGl6ZWQgaW50ZWdlcjsgMi4gaGVpZ2h0LCB0aGUgaGVpZ2h0IG9mIHRoZVxuICogdGV4dHVyZTsgMy4gd2lkdGgsIHRoZSB3aWR0aCBvZiB0aGUgdGV4dHVyZTsgNC4gY2hhbm5lbHMsIGEgbm9uLWVtcHR5IHN1YnNldFxuICogb2YgJ1JHQkEnLCBpbmRpY2F0aW5nIHRoZSB2YWx1ZXMgb2Ygd2hpY2ggY2hhbm5lbHMgd2lsbCBiZSBwYXNzZWQgdG8gdGhlXG4gKiB0ZW5zb3IsIHN1Y2ggYXMgJ1InIG9yICdCUicgKFRoZSBvcmRlciBvZiB0aGUgY2hhbm5lbHMgYWZmZWN0IHRoZSBvcmRlciBvZlxuICogdGVuc29yIHZhbHVlcy4gKS4gKElmIHRoZSB2YWx1ZXMgcGFzc2VkIGZyb20gdGV4dHVyZSBpcyBsZXNzIHRoYW4gdGhlIHRlbnNvclxuICogc2l6ZSwgemVyb3Mgd2lsbCBiZSBwYWRkZWQgYXQgdGhlIHJlYXIuKS4gSWYgdGhlIHZhbHVlcyBpcyBhIGBXZWJHUFVEYXRhYFxuICogb2JqZWN0LCB0aGUgZHR5cGUgY291bGQgb25seSBiZSAnZmxvYXQzMicgb3IgJ2ludDMyIGFuZCB0aGUgb2JqZWN0IGhhcyB0b1xuICogaGF2ZTogYnVmZmVyLCBhIGBHUFVCdWZmZXJgLiBUaGUgYnVmZmVyIG11c3Q6IDEuIHNoYXJlIHRoZSBzYW1lIGBHUFVEZXZpY2VgXG4gKiB3aXRoIFRGSlMncyBXZWJHUFUgYmFja2VuZDsgMi4gYnVmZmVyLnVzYWdlIHNob3VsZCBhdCBsZWFzdCBzdXBwb3J0XG4gKiBHUFVCdWZmZXJVc2FnZS5TVE9SQUdFIHwgR1BVQnVmZmVyVXNhZ2UuQ09QWV9TUkM7IDMuIGJ1ZmZlci5zaXplIHNob3VsZCBub3RcbiAqIGJlIHNtYWxsZXIgdGhhbiB0aGUgYnl0ZSBzaXplIG9mIHRlbnNvciBzaGFwZS4gV2ViR1BVRGF0YSBvcHRpb25hbGx5IHN1cHBvcnRzXG4gKiB6ZXJvIGNvcHkgYnkgZmxhZyB6ZXJvQ29weS4gV2hlbiB6ZXJvQ29weSBpcyBmYWxzZSBvciB1bmRlZmluZWQoZGVmYXVsdCksXG4gKiB0aGlzIHBhc3NpbmcgR1BVQnVmZmVyIGNhbiBiZSBkZXN0cm95ZWQgYWZ0ZXIgdGVuc29yIGlzIGNyZWF0ZWQuIFdoZW5cbiAqIHplcm9Db3B5IGlzIHRydWUsIHRoaXMgR1BVQnVmZmVyIGlzIGJvdW5kIGRpcmVjdGx5IGJ5IHRoZSB0ZW5zb3IsIHNvIGRvIG5vdFxuICogZGVzdHJveSB0aGlzIEdQVUJ1ZmZlciB1bnRpbCBhbGwgYWNjZXNzIGlzIGRvbmUuXG4gKiBAcGFyYW0gc2hhcGUgVGhlIHNoYXBlIG9mIHRoZSB0ZW5zb3IuIE9wdGlvbmFsLiBJZiBub3QgcHJvdmlkZWQsXG4gKiAgIGl0IGlzIGluZmVycmVkIGZyb20gYHZhbHVlc2AuXG4gKiBAcGFyYW0gZHR5cGUgVGhlIGRhdGEgdHlwZS5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnVGVuc29ycycsIHN1YmhlYWRpbmc6ICdDcmVhdGlvbid9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiB0ZW5zb3I8UiBleHRlbmRzIFJhbms+KFxuICAgIHZhbHVlczogVGVuc29yTGlrZXxXZWJHTERhdGF8V2ViR1BVRGF0YSwgc2hhcGU/OiBTaGFwZU1hcFtSXSxcbiAgICBkdHlwZT86IERhdGFUeXBlKTogVGVuc29yPFI+IHtcbiAgY29uc3QgaW5mZXJyZWRTaGFwZSA9IGluZmVyU2hhcGUodmFsdWVzLCBkdHlwZSk7XG4gIHJldHVybiBtYWtlVGVuc29yKHZhbHVlcywgc2hhcGUsIGluZmVycmVkU2hhcGUsIGR0eXBlKSBhcyBUZW5zb3I8Uj47XG59XG4iXX0=