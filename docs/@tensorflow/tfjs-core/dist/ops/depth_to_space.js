/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
import { ENGINE } from '../engine';
import { DepthToSpace } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import * as util from '../util';
import { op } from './operation';
/**
 * Rearranges data from depth into blocks of spatial data. More specifically,
 * this op outputs a copy of the input tensor where values from the `depth`
 * dimension are moved in spatial blocks to the `height` and `width` dimensions.
 * The attr `blockSize` indicates the input block size and how the data is
 * moved.
 *
 *  - Chunks of data of size `blockSize * blockSize` from depth are rearranged
 * into non-overlapping blocks of size `blockSize x blockSize`
 *
 *  - The width the output tensor is `inputWidth * blockSize`, whereas the
 * height is `inputHeight * blockSize`
 *
 *  - The Y, X coordinates within each block of the output image are determined
 * by the high order component of the input channel index
 *
 *  - The depth of the input tensor must be divisible by `blockSize *
 * blockSize`
 *
 * The `dataFormat` attr specifies the layout of the input and output tensors
 * with the following options: "NHWC": [ `batch, height, width, channels` ]
 * "NCHW": [ `batch, channels, height, width` ]
 *
 * ```js
 * const x = tf.tensor4d([1, 2, 3, 4], [1, 1, 1, 4]);
 * const blockSize = 2;
 * const dataFormat = "NHWC";
 *
 * tf.depthToSpace(x, blockSize, dataFormat).print();
 * ```
 *
 * @param x The input tensor of rank 4
 * @param blockSIze  An `int` that is `>= 2`. The size of the spatial block
 * @param dataFormat An optional string from: "NHWC", "NCHW". Defaults to "NHWC"
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
function depthToSpace_(x, blockSize, dataFormat = 'NHWC') {
    const $x = convertToTensor(x, 'x', 'depthToSpace', 'float32');
    const inputHeight = (dataFormat === 'NHWC') ? $x.shape[1] : $x.shape[2];
    const inputWidth = (dataFormat === 'NHWC') ? $x.shape[2] : $x.shape[3];
    const inputDepth = (dataFormat === 'NHWC') ? $x.shape[3] : $x.shape[1];
    util.assert(blockSize > 1, () => `blockSize should be > 1 for depthToSpace, but was: ${blockSize}`);
    util.assert(inputHeight * blockSize >= 0, () => `Negative dimension size caused by overflow when multiplying
    ${inputHeight} and ${blockSize}  for depthToSpace with input shape
    ${$x.shape}`);
    util.assert(inputWidth * blockSize >= 0, () => `Negative dimension size caused by overflow when multiplying
    ${inputWidth} and ${blockSize} for depthToSpace with input shape
        ${$x.shape}`);
    util.assert((inputDepth % (blockSize * blockSize) === 0), () => `Dimension size must be evenly divisible by ${blockSize * blockSize} but is ${inputDepth} for depthToSpace with input shape ${$x.shape}`);
    const inputs = { x: $x };
    const attrs = { blockSize, dataFormat };
    return ENGINE.runKernel(DepthToSpace, inputs, attrs);
}
export const depthToSpace = /* @__PURE__ */ op({ depthToSpace_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZGVwdGhfdG9fc3BhY2UuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9kZXB0aF90b19zcGFjZS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQ2pDLE9BQU8sRUFBQyxZQUFZLEVBQXdDLE1BQU0saUJBQWlCLENBQUM7QUFJcEYsT0FBTyxFQUFDLGVBQWUsRUFBQyxNQUFNLG9CQUFvQixDQUFDO0FBRW5ELE9BQU8sS0FBSyxJQUFJLE1BQU0sU0FBUyxDQUFDO0FBRWhDLE9BQU8sRUFBQyxFQUFFLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFFL0I7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQW9DRztBQUNILFNBQVMsYUFBYSxDQUNsQixDQUF3QixFQUFFLFNBQWlCLEVBQzNDLGFBQTRCLE1BQU07SUFDcEMsTUFBTSxFQUFFLEdBQUcsZUFBZSxDQUFDLENBQUMsRUFBRSxHQUFHLEVBQUUsY0FBYyxFQUFFLFNBQVMsQ0FBYSxDQUFDO0lBRTFFLE1BQU0sV0FBVyxHQUFHLENBQUMsVUFBVSxLQUFLLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ3hFLE1BQU0sVUFBVSxHQUFHLENBQUMsVUFBVSxLQUFLLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ3ZFLE1BQU0sVUFBVSxHQUFHLENBQUMsVUFBVSxLQUFLLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBRXZFLElBQUksQ0FBQyxNQUFNLENBQ1AsU0FBUyxHQUFHLENBQUMsRUFDYixHQUFHLEVBQUUsQ0FBQyxzREFBc0QsU0FBUyxFQUFFLENBQUMsQ0FBQztJQUU3RSxJQUFJLENBQUMsTUFBTSxDQUNQLFdBQVcsR0FBRyxTQUFTLElBQUksQ0FBQyxFQUM1QixHQUFHLEVBQUUsQ0FBQztNQUNOLFdBQVcsUUFBUSxTQUFTO01BQzVCLEVBQUUsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDO0lBRWhCLElBQUksQ0FBQyxNQUFNLENBQ1AsVUFBVSxHQUFHLFNBQVMsSUFBSSxDQUFDLEVBQzNCLEdBQUcsRUFBRSxDQUFDO01BQ04sVUFBVSxRQUFRLFNBQVM7VUFDdkIsRUFBRSxDQUFDLEtBQUssRUFBRSxDQUFDLENBQUM7SUFFcEIsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLFVBQVUsR0FBRyxDQUFDLFNBQVMsR0FBRyxTQUFTLENBQUMsS0FBSyxDQUFDLENBQUMsRUFDNUMsR0FBRyxFQUFFLENBQUMsOENBQ0YsU0FBUyxHQUFHLFNBQVMsV0FDckIsVUFBVSxzQ0FBc0MsRUFBRSxDQUFDLEtBQUssRUFBRSxDQUFDLENBQUM7SUFFcEUsTUFBTSxNQUFNLEdBQXVCLEVBQUMsQ0FBQyxFQUFFLEVBQUUsRUFBQyxDQUFDO0lBQzNDLE1BQU0sS0FBSyxHQUFzQixFQUFDLFNBQVMsRUFBRSxVQUFVLEVBQUMsQ0FBQztJQUV6RCxPQUFPLE1BQU0sQ0FBQyxTQUFTLENBQ25CLFlBQVksRUFBRSxNQUFtQyxFQUNqRCxLQUFnQyxDQUFDLENBQUM7QUFDeEMsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLFlBQVksR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUMsYUFBYSxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtFTkdJTkV9IGZyb20gJy4uL2VuZ2luZSc7XG5pbXBvcnQge0RlcHRoVG9TcGFjZSwgRGVwdGhUb1NwYWNlQXR0cnMsIERlcHRoVG9TcGFjZUlucHV0c30gZnJvbSAnLi4va2VybmVsX25hbWVzJztcbmltcG9ydCB7TmFtZWRBdHRyTWFwfSBmcm9tICcuLi9rZXJuZWxfcmVnaXN0cnknO1xuaW1wb3J0IHtUZW5zb3I0RH0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7TmFtZWRUZW5zb3JNYXB9IGZyb20gJy4uL3RlbnNvcl90eXBlcyc7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7VGVuc29yTGlrZTREfSBmcm9tICcuLi90eXBlcyc7XG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4uL3V0aWwnO1xuXG5pbXBvcnQge29wfSBmcm9tICcuL29wZXJhdGlvbic7XG5cbi8qKlxuICogUmVhcnJhbmdlcyBkYXRhIGZyb20gZGVwdGggaW50byBibG9ja3Mgb2Ygc3BhdGlhbCBkYXRhLiBNb3JlIHNwZWNpZmljYWxseSxcbiAqIHRoaXMgb3Agb3V0cHV0cyBhIGNvcHkgb2YgdGhlIGlucHV0IHRlbnNvciB3aGVyZSB2YWx1ZXMgZnJvbSB0aGUgYGRlcHRoYFxuICogZGltZW5zaW9uIGFyZSBtb3ZlZCBpbiBzcGF0aWFsIGJsb2NrcyB0byB0aGUgYGhlaWdodGAgYW5kIGB3aWR0aGAgZGltZW5zaW9ucy5cbiAqIFRoZSBhdHRyIGBibG9ja1NpemVgIGluZGljYXRlcyB0aGUgaW5wdXQgYmxvY2sgc2l6ZSBhbmQgaG93IHRoZSBkYXRhIGlzXG4gKiBtb3ZlZC5cbiAqXG4gKiAgLSBDaHVua3Mgb2YgZGF0YSBvZiBzaXplIGBibG9ja1NpemUgKiBibG9ja1NpemVgIGZyb20gZGVwdGggYXJlIHJlYXJyYW5nZWRcbiAqIGludG8gbm9uLW92ZXJsYXBwaW5nIGJsb2NrcyBvZiBzaXplIGBibG9ja1NpemUgeCBibG9ja1NpemVgXG4gKlxuICogIC0gVGhlIHdpZHRoIHRoZSBvdXRwdXQgdGVuc29yIGlzIGBpbnB1dFdpZHRoICogYmxvY2tTaXplYCwgd2hlcmVhcyB0aGVcbiAqIGhlaWdodCBpcyBgaW5wdXRIZWlnaHQgKiBibG9ja1NpemVgXG4gKlxuICogIC0gVGhlIFksIFggY29vcmRpbmF0ZXMgd2l0aGluIGVhY2ggYmxvY2sgb2YgdGhlIG91dHB1dCBpbWFnZSBhcmUgZGV0ZXJtaW5lZFxuICogYnkgdGhlIGhpZ2ggb3JkZXIgY29tcG9uZW50IG9mIHRoZSBpbnB1dCBjaGFubmVsIGluZGV4XG4gKlxuICogIC0gVGhlIGRlcHRoIG9mIHRoZSBpbnB1dCB0ZW5zb3IgbXVzdCBiZSBkaXZpc2libGUgYnkgYGJsb2NrU2l6ZSAqXG4gKiBibG9ja1NpemVgXG4gKlxuICogVGhlIGBkYXRhRm9ybWF0YCBhdHRyIHNwZWNpZmllcyB0aGUgbGF5b3V0IG9mIHRoZSBpbnB1dCBhbmQgb3V0cHV0IHRlbnNvcnNcbiAqIHdpdGggdGhlIGZvbGxvd2luZyBvcHRpb25zOiBcIk5IV0NcIjogWyBgYmF0Y2gsIGhlaWdodCwgd2lkdGgsIGNoYW5uZWxzYCBdXG4gKiBcIk5DSFdcIjogWyBgYmF0Y2gsIGNoYW5uZWxzLCBoZWlnaHQsIHdpZHRoYCBdXG4gKlxuICogYGBganNcbiAqIGNvbnN0IHggPSB0Zi50ZW5zb3I0ZChbMSwgMiwgMywgNF0sIFsxLCAxLCAxLCA0XSk7XG4gKiBjb25zdCBibG9ja1NpemUgPSAyO1xuICogY29uc3QgZGF0YUZvcm1hdCA9IFwiTkhXQ1wiO1xuICpcbiAqIHRmLmRlcHRoVG9TcGFjZSh4LCBibG9ja1NpemUsIGRhdGFGb3JtYXQpLnByaW50KCk7XG4gKiBgYGBcbiAqXG4gKiBAcGFyYW0geCBUaGUgaW5wdXQgdGVuc29yIG9mIHJhbmsgNFxuICogQHBhcmFtIGJsb2NrU0l6ZSAgQW4gYGludGAgdGhhdCBpcyBgPj0gMmAuIFRoZSBzaXplIG9mIHRoZSBzcGF0aWFsIGJsb2NrXG4gKiBAcGFyYW0gZGF0YUZvcm1hdCBBbiBvcHRpb25hbCBzdHJpbmcgZnJvbTogXCJOSFdDXCIsIFwiTkNIV1wiLiBEZWZhdWx0cyB0byBcIk5IV0NcIlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdUZW5zb3JzJywgc3ViaGVhZGluZzogJ1RyYW5zZm9ybWF0aW9ucyd9XG4gKi9cbmZ1bmN0aW9uIGRlcHRoVG9TcGFjZV8oXG4gICAgeDogVGVuc29yNER8VGVuc29yTGlrZTRELCBibG9ja1NpemU6IG51bWJlcixcbiAgICBkYXRhRm9ybWF0OiAnTkhXQyd8J05DSFcnID0gJ05IV0MnKTogVGVuc29yNEQge1xuICBjb25zdCAkeCA9IGNvbnZlcnRUb1RlbnNvcih4LCAneCcsICdkZXB0aFRvU3BhY2UnLCAnZmxvYXQzMicpIGFzIFRlbnNvcjREO1xuXG4gIGNvbnN0IGlucHV0SGVpZ2h0ID0gKGRhdGFGb3JtYXQgPT09ICdOSFdDJykgPyAkeC5zaGFwZVsxXSA6ICR4LnNoYXBlWzJdO1xuICBjb25zdCBpbnB1dFdpZHRoID0gKGRhdGFGb3JtYXQgPT09ICdOSFdDJykgPyAkeC5zaGFwZVsyXSA6ICR4LnNoYXBlWzNdO1xuICBjb25zdCBpbnB1dERlcHRoID0gKGRhdGFGb3JtYXQgPT09ICdOSFdDJykgPyAkeC5zaGFwZVszXSA6ICR4LnNoYXBlWzFdO1xuXG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgYmxvY2tTaXplID4gMSxcbiAgICAgICgpID0+IGBibG9ja1NpemUgc2hvdWxkIGJlID4gMSBmb3IgZGVwdGhUb1NwYWNlLCBidXQgd2FzOiAke2Jsb2NrU2l6ZX1gKTtcblxuICB1dGlsLmFzc2VydChcbiAgICAgIGlucHV0SGVpZ2h0ICogYmxvY2tTaXplID49IDAsXG4gICAgICAoKSA9PiBgTmVnYXRpdmUgZGltZW5zaW9uIHNpemUgY2F1c2VkIGJ5IG92ZXJmbG93IHdoZW4gbXVsdGlwbHlpbmdcbiAgICAke2lucHV0SGVpZ2h0fSBhbmQgJHtibG9ja1NpemV9ICBmb3IgZGVwdGhUb1NwYWNlIHdpdGggaW5wdXQgc2hhcGVcbiAgICAkeyR4LnNoYXBlfWApO1xuXG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgaW5wdXRXaWR0aCAqIGJsb2NrU2l6ZSA+PSAwLFxuICAgICAgKCkgPT4gYE5lZ2F0aXZlIGRpbWVuc2lvbiBzaXplIGNhdXNlZCBieSBvdmVyZmxvdyB3aGVuIG11bHRpcGx5aW5nXG4gICAgJHtpbnB1dFdpZHRofSBhbmQgJHtibG9ja1NpemV9IGZvciBkZXB0aFRvU3BhY2Ugd2l0aCBpbnB1dCBzaGFwZVxuICAgICAgICAkeyR4LnNoYXBlfWApO1xuXG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgKGlucHV0RGVwdGggJSAoYmxvY2tTaXplICogYmxvY2tTaXplKSA9PT0gMCksXG4gICAgICAoKSA9PiBgRGltZW5zaW9uIHNpemUgbXVzdCBiZSBldmVubHkgZGl2aXNpYmxlIGJ5ICR7XG4gICAgICAgICAgYmxvY2tTaXplICogYmxvY2tTaXplfSBidXQgaXMgJHtcbiAgICAgICAgICBpbnB1dERlcHRofSBmb3IgZGVwdGhUb1NwYWNlIHdpdGggaW5wdXQgc2hhcGUgJHskeC5zaGFwZX1gKTtcblxuICBjb25zdCBpbnB1dHM6IERlcHRoVG9TcGFjZUlucHV0cyA9IHt4OiAkeH07XG4gIGNvbnN0IGF0dHJzOiBEZXB0aFRvU3BhY2VBdHRycyA9IHtibG9ja1NpemUsIGRhdGFGb3JtYXR9O1xuXG4gIHJldHVybiBFTkdJTkUucnVuS2VybmVsKFxuICAgICAgRGVwdGhUb1NwYWNlLCBpbnB1dHMgYXMgdW5rbm93biBhcyBOYW1lZFRlbnNvck1hcCxcbiAgICAgIGF0dHJzIGFzIHVua25vd24gYXMgTmFtZWRBdHRyTWFwKTtcbn1cblxuZXhwb3J0IGNvbnN0IGRlcHRoVG9TcGFjZSA9IC8qIEBfX1BVUkVfXyAqLyBvcCh7ZGVwdGhUb1NwYWNlX30pO1xuIl19