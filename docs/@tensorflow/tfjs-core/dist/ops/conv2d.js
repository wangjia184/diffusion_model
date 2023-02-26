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
import { Conv2D } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import * as util from '../util';
import * as conv_util from './conv_util';
import { op } from './operation';
import { reshape } from './reshape';
/**
 * Computes a 2D convolution over the input x.
 *
 * @param x The input tensor, of rank 4 or rank 3, of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is
 * assumed.
 * @param filter The filter, rank 4, of shape
 *     `[filterHeight, filterWidth, inDepth, outDepth]`.
 * @param strides The strides of the convolution: `[strideHeight,
 * strideWidth]`.
 * @param pad The type of padding algorithm.
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1x1.
 *   - For more info, see this guide:
 *     [https://www.tensorflow.org/api_docs/python/tf/nn/convolution](
 *          https://www.tensorflow.org/api_docs/python/tf/nn/convolution)
 * @param dataFormat: An optional string from: "NHWC", "NCHW". Defaults to
 *     "NHWC". Specify the data format of the input and output data. With the
 *     default format "NHWC", the data is stored in the order of: [batch,
 *     height, width, channels].
 * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`
 *     in which we sample input values across the height and width dimensions
 *     in atrous convolution. Defaults to `[1, 1]`. If `dilations` is a single
 *     number, then `dilationHeight == dilationWidth`. If it is greater than
 *     1, then all values of `strides` must be 1.
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 *
 * @doc {heading: 'Operations', subheading: 'Convolution'}
 */
function conv2d_(x, filter, strides, pad, dataFormat = 'NHWC', dilations = [1, 1], dimRoundingMode) {
    const $x = convertToTensor(x, 'x', 'conv2d', 'float32');
    const $filter = convertToTensor(filter, 'filter', 'conv2d', 'float32');
    let x4D = $x;
    let reshapedTo4D = false;
    if ($x.rank === 3) {
        reshapedTo4D = true;
        x4D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2]]);
    }
    util.assert(x4D.rank === 4, () => `Error in conv2d: input must be rank 4, but got rank ${x4D.rank}.`);
    util.assert($filter.rank === 4, () => `Error in conv2d: filter must be rank 4, but got rank ` +
        `${$filter.rank}.`);
    conv_util.checkPadOnDimRoundingMode('conv2d', pad, dimRoundingMode);
    const inDepth = dataFormat === 'NHWC' ? x4D.shape[3] : x4D.shape[1];
    util.assert(inDepth === $filter.shape[2], () => `Error in conv2d: depth of input (${inDepth}) must match ` +
        `input depth for filter ${$filter.shape[2]}.`);
    util.assert(conv_util.eitherStridesOrDilationsAreOne(strides, dilations), () => 'Error in conv2D: Either strides or dilations must be 1. ' +
        `Got strides ${strides} and dilations '${dilations}'`);
    util.assert(conv_util.stridesOrDilationsArePositive(dilations), () => 'Error in conv2D: Dilated rates should be larger than 0.');
    util.assert(conv_util.stridesOrDilationsArePositive(strides), () => 'Error in conv2D: Strides should be larger than 0.');
    const inputs = { x: x4D, filter: $filter };
    const attrs = { strides, pad, dataFormat, dilations, dimRoundingMode };
    // tslint:disable-next-line: no-unnecessary-type-assertion
    const res = ENGINE.runKernel(Conv2D, inputs, attrs);
    if (reshapedTo4D) {
        return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]);
    }
    return res;
}
export const conv2d = /* @__PURE__ */ op({ conv2d_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY29udjJkLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvY29udjJkLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUNILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDakMsT0FBTyxFQUFDLE1BQU0sRUFBNEIsTUFBTSxpQkFBaUIsQ0FBQztBQUlsRSxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFFbkQsT0FBTyxLQUFLLElBQUksTUFBTSxTQUFTLENBQUM7QUFFaEMsT0FBTyxLQUFLLFNBQVMsTUFBTSxhQUFhLENBQUM7QUFDekMsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUMvQixPQUFPLEVBQUMsT0FBTyxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBRWxDOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBK0JHO0FBQ0gsU0FBUyxPQUFPLENBQ1osQ0FBZSxFQUFFLE1BQTJCLEVBQzVDLE9BQWdDLEVBQ2hDLEdBQW9ELEVBQ3BELGFBQTRCLE1BQU0sRUFDbEMsWUFBcUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQzNDLGVBQXdDO0lBQzFDLE1BQU0sRUFBRSxHQUFHLGVBQWUsQ0FBQyxDQUFDLEVBQUUsR0FBRyxFQUFFLFFBQVEsRUFBRSxTQUFTLENBQUMsQ0FBQztJQUN4RCxNQUFNLE9BQU8sR0FBRyxlQUFlLENBQUMsTUFBTSxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsU0FBUyxDQUFDLENBQUM7SUFFdkUsSUFBSSxHQUFHLEdBQUcsRUFBYyxDQUFDO0lBQ3pCLElBQUksWUFBWSxHQUFHLEtBQUssQ0FBQztJQUV6QixJQUFJLEVBQUUsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUFFO1FBQ2pCLFlBQVksR0FBRyxJQUFJLENBQUM7UUFDcEIsR0FBRyxHQUFHLE9BQU8sQ0FBQyxFQUFFLEVBQUUsQ0FBQyxDQUFDLEVBQUUsRUFBRSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0tBQy9EO0lBRUQsSUFBSSxDQUFDLE1BQU0sQ0FDUCxHQUFHLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDZCxHQUFHLEVBQUUsQ0FBQyx1REFBdUQsR0FBRyxDQUFDLElBQUksR0FBRyxDQUFDLENBQUM7SUFDOUUsSUFBSSxDQUFDLE1BQU0sQ0FDUCxPQUFPLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDbEIsR0FBRyxFQUFFLENBQUMsdURBQXVEO1FBQ3pELEdBQUcsT0FBTyxDQUFDLElBQUksR0FBRyxDQUFDLENBQUM7SUFDNUIsU0FBUyxDQUFDLHlCQUF5QixDQUFDLFFBQVEsRUFBRSxHQUFHLEVBQUUsZUFBZSxDQUFDLENBQUM7SUFDcEUsTUFBTSxPQUFPLEdBQUcsVUFBVSxLQUFLLE1BQU0sQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUNwRSxJQUFJLENBQUMsTUFBTSxDQUNQLE9BQU8sS0FBSyxPQUFPLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUM1QixHQUFHLEVBQUUsQ0FBQyxvQ0FBb0MsT0FBTyxlQUFlO1FBQzVELDBCQUEwQixPQUFPLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQztJQUN2RCxJQUFJLENBQUMsTUFBTSxDQUNQLFNBQVMsQ0FBQyw4QkFBOEIsQ0FBQyxPQUFPLEVBQUUsU0FBUyxDQUFDLEVBQzVELEdBQUcsRUFBRSxDQUFDLDBEQUEwRDtRQUM1RCxlQUFlLE9BQU8sbUJBQW1CLFNBQVMsR0FBRyxDQUFDLENBQUM7SUFDL0QsSUFBSSxDQUFDLE1BQU0sQ0FDUCxTQUFTLENBQUMsNkJBQTZCLENBQUMsU0FBUyxDQUFDLEVBQ2xELEdBQUcsRUFBRSxDQUFDLHlEQUF5RCxDQUFDLENBQUM7SUFDckUsSUFBSSxDQUFDLE1BQU0sQ0FDUCxTQUFTLENBQUMsNkJBQTZCLENBQUMsT0FBTyxDQUFDLEVBQ2hELEdBQUcsRUFBRSxDQUFDLG1EQUFtRCxDQUFDLENBQUM7SUFFL0QsTUFBTSxNQUFNLEdBQWlCLEVBQUMsQ0FBQyxFQUFFLEdBQUcsRUFBRSxNQUFNLEVBQUUsT0FBTyxFQUFDLENBQUM7SUFDdkQsTUFBTSxLQUFLLEdBQ08sRUFBQyxPQUFPLEVBQUUsR0FBRyxFQUFFLFVBQVUsRUFBRSxTQUFTLEVBQUUsZUFBZSxFQUFDLENBQUM7SUFFekUsMERBQTBEO0lBQzFELE1BQU0sR0FBRyxHQUFHLE1BQU0sQ0FBQyxTQUFTLENBQ1osTUFBTSxFQUFFLE1BQW1DLEVBQzNDLEtBQWdDLENBQU0sQ0FBQztJQUV2RCxJQUFJLFlBQVksRUFBRTtRQUNoQixPQUFPLE9BQU8sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxHQUFHLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFNLENBQUM7S0FDdEU7SUFDRCxPQUFPLEdBQUcsQ0FBQztBQUNiLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxNQUFNLEdBQUcsZUFBZSxDQUFDLEVBQUUsQ0FBQyxFQUFDLE9BQU8sRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5pbXBvcnQge0VOR0lORX0gZnJvbSAnLi4vZW5naW5lJztcbmltcG9ydCB7Q29udjJELCBDb252MkRBdHRycywgQ29udjJESW5wdXRzfSBmcm9tICcuLi9rZXJuZWxfbmFtZXMnO1xuaW1wb3J0IHtOYW1lZEF0dHJNYXB9IGZyb20gJy4uL2tlcm5lbF9yZWdpc3RyeSc7XG5pbXBvcnQge1RlbnNvcjNELCBUZW5zb3I0RH0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7TmFtZWRUZW5zb3JNYXB9IGZyb20gJy4uL3RlbnNvcl90eXBlcyc7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7VGVuc29yTGlrZX0gZnJvbSAnLi4vdHlwZXMnO1xuaW1wb3J0ICogYXMgdXRpbCBmcm9tICcuLi91dGlsJztcblxuaW1wb3J0ICogYXMgY29udl91dGlsIGZyb20gJy4vY29udl91dGlsJztcbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcbmltcG9ydCB7cmVzaGFwZX0gZnJvbSAnLi9yZXNoYXBlJztcblxuLyoqXG4gKiBDb21wdXRlcyBhIDJEIGNvbnZvbHV0aW9uIG92ZXIgdGhlIGlucHV0IHguXG4gKlxuICogQHBhcmFtIHggVGhlIGlucHV0IHRlbnNvciwgb2YgcmFuayA0IG9yIHJhbmsgMywgb2Ygc2hhcGVcbiAqICAgICBgW2JhdGNoLCBoZWlnaHQsIHdpZHRoLCBpbkNoYW5uZWxzXWAuIElmIHJhbmsgMywgYmF0Y2ggb2YgMSBpc1xuICogYXNzdW1lZC5cbiAqIEBwYXJhbSBmaWx0ZXIgVGhlIGZpbHRlciwgcmFuayA0LCBvZiBzaGFwZVxuICogICAgIGBbZmlsdGVySGVpZ2h0LCBmaWx0ZXJXaWR0aCwgaW5EZXB0aCwgb3V0RGVwdGhdYC5cbiAqIEBwYXJhbSBzdHJpZGVzIFRoZSBzdHJpZGVzIG9mIHRoZSBjb252b2x1dGlvbjogYFtzdHJpZGVIZWlnaHQsXG4gKiBzdHJpZGVXaWR0aF1gLlxuICogQHBhcmFtIHBhZCBUaGUgdHlwZSBvZiBwYWRkaW5nIGFsZ29yaXRobS5cbiAqICAgIC0gYHNhbWVgIGFuZCBzdHJpZGUgMTogb3V0cHV0IHdpbGwgYmUgb2Ygc2FtZSBzaXplIGFzIGlucHV0LFxuICogICAgICAgcmVnYXJkbGVzcyBvZiBmaWx0ZXIgc2l6ZS5cbiAqICAgIC0gYHZhbGlkYDogb3V0cHV0IHdpbGwgYmUgc21hbGxlciB0aGFuIGlucHV0IGlmIGZpbHRlciBpcyBsYXJnZXJcbiAqICAgICAgIHRoYW4gMXgxLlxuICogICAtIEZvciBtb3JlIGluZm8sIHNlZSB0aGlzIGd1aWRlOlxuICogICAgIFtodHRwczovL3d3dy50ZW5zb3JmbG93Lm9yZy9hcGlfZG9jcy9weXRob24vdGYvbm4vY29udm9sdXRpb25dKFxuICogICAgICAgICAgaHR0cHM6Ly93d3cudGVuc29yZmxvdy5vcmcvYXBpX2RvY3MvcHl0aG9uL3RmL25uL2NvbnZvbHV0aW9uKVxuICogQHBhcmFtIGRhdGFGb3JtYXQ6IEFuIG9wdGlvbmFsIHN0cmluZyBmcm9tOiBcIk5IV0NcIiwgXCJOQ0hXXCIuIERlZmF1bHRzIHRvXG4gKiAgICAgXCJOSFdDXCIuIFNwZWNpZnkgdGhlIGRhdGEgZm9ybWF0IG9mIHRoZSBpbnB1dCBhbmQgb3V0cHV0IGRhdGEuIFdpdGggdGhlXG4gKiAgICAgZGVmYXVsdCBmb3JtYXQgXCJOSFdDXCIsIHRoZSBkYXRhIGlzIHN0b3JlZCBpbiB0aGUgb3JkZXIgb2Y6IFtiYXRjaCxcbiAqICAgICBoZWlnaHQsIHdpZHRoLCBjaGFubmVsc10uXG4gKiBAcGFyYW0gZGlsYXRpb25zIFRoZSBkaWxhdGlvbiByYXRlczogYFtkaWxhdGlvbkhlaWdodCwgZGlsYXRpb25XaWR0aF1gXG4gKiAgICAgaW4gd2hpY2ggd2Ugc2FtcGxlIGlucHV0IHZhbHVlcyBhY3Jvc3MgdGhlIGhlaWdodCBhbmQgd2lkdGggZGltZW5zaW9uc1xuICogICAgIGluIGF0cm91cyBjb252b2x1dGlvbi4gRGVmYXVsdHMgdG8gYFsxLCAxXWAuIElmIGBkaWxhdGlvbnNgIGlzIGEgc2luZ2xlXG4gKiAgICAgbnVtYmVyLCB0aGVuIGBkaWxhdGlvbkhlaWdodCA9PSBkaWxhdGlvbldpZHRoYC4gSWYgaXQgaXMgZ3JlYXRlciB0aGFuXG4gKiAgICAgMSwgdGhlbiBhbGwgdmFsdWVzIG9mIGBzdHJpZGVzYCBtdXN0IGJlIDEuXG4gKiBAcGFyYW0gZGltUm91bmRpbmdNb2RlIEEgc3RyaW5nIGZyb206ICdjZWlsJywgJ3JvdW5kJywgJ2Zsb29yJy4gSWYgbm9uZSBpc1xuICogICAgIHByb3ZpZGVkLCBpdCB3aWxsIGRlZmF1bHQgdG8gdHJ1bmNhdGUuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ09wZXJhdGlvbnMnLCBzdWJoZWFkaW5nOiAnQ29udm9sdXRpb24nfVxuICovXG5mdW5jdGlvbiBjb252MmRfPFQgZXh0ZW5kcyBUZW5zb3IzRHxUZW5zb3I0RD4oXG4gICAgeDogVHxUZW5zb3JMaWtlLCBmaWx0ZXI6IFRlbnNvcjREfFRlbnNvckxpa2UsXG4gICAgc3RyaWRlczogW251bWJlciwgbnVtYmVyXXxudW1iZXIsXG4gICAgcGFkOiAndmFsaWQnfCdzYW1lJ3xudW1iZXJ8Y29udl91dGlsLkV4cGxpY2l0UGFkZGluZyxcbiAgICBkYXRhRm9ybWF0OiAnTkhXQyd8J05DSFcnID0gJ05IV0MnLFxuICAgIGRpbGF0aW9uczogW251bWJlciwgbnVtYmVyXXxudW1iZXIgPSBbMSwgMV0sXG4gICAgZGltUm91bmRpbmdNb2RlPzogJ2Zsb29yJ3wncm91bmQnfCdjZWlsJyk6IFQge1xuICBjb25zdCAkeCA9IGNvbnZlcnRUb1RlbnNvcih4LCAneCcsICdjb252MmQnLCAnZmxvYXQzMicpO1xuICBjb25zdCAkZmlsdGVyID0gY29udmVydFRvVGVuc29yKGZpbHRlciwgJ2ZpbHRlcicsICdjb252MmQnLCAnZmxvYXQzMicpO1xuXG4gIGxldCB4NEQgPSAkeCBhcyBUZW5zb3I0RDtcbiAgbGV0IHJlc2hhcGVkVG80RCA9IGZhbHNlO1xuXG4gIGlmICgkeC5yYW5rID09PSAzKSB7XG4gICAgcmVzaGFwZWRUbzREID0gdHJ1ZTtcbiAgICB4NEQgPSByZXNoYXBlKCR4LCBbMSwgJHguc2hhcGVbMF0sICR4LnNoYXBlWzFdLCAkeC5zaGFwZVsyXV0pO1xuICB9XG5cbiAgdXRpbC5hc3NlcnQoXG4gICAgICB4NEQucmFuayA9PT0gNCxcbiAgICAgICgpID0+IGBFcnJvciBpbiBjb252MmQ6IGlucHV0IG11c3QgYmUgcmFuayA0LCBidXQgZ290IHJhbmsgJHt4NEQucmFua30uYCk7XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgJGZpbHRlci5yYW5rID09PSA0LFxuICAgICAgKCkgPT4gYEVycm9yIGluIGNvbnYyZDogZmlsdGVyIG11c3QgYmUgcmFuayA0LCBidXQgZ290IHJhbmsgYCArXG4gICAgICAgICAgYCR7JGZpbHRlci5yYW5rfS5gKTtcbiAgY29udl91dGlsLmNoZWNrUGFkT25EaW1Sb3VuZGluZ01vZGUoJ2NvbnYyZCcsIHBhZCwgZGltUm91bmRpbmdNb2RlKTtcbiAgY29uc3QgaW5EZXB0aCA9IGRhdGFGb3JtYXQgPT09ICdOSFdDJyA/IHg0RC5zaGFwZVszXSA6IHg0RC5zaGFwZVsxXTtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICBpbkRlcHRoID09PSAkZmlsdGVyLnNoYXBlWzJdLFxuICAgICAgKCkgPT4gYEVycm9yIGluIGNvbnYyZDogZGVwdGggb2YgaW5wdXQgKCR7aW5EZXB0aH0pIG11c3QgbWF0Y2ggYCArXG4gICAgICAgICAgYGlucHV0IGRlcHRoIGZvciBmaWx0ZXIgJHskZmlsdGVyLnNoYXBlWzJdfS5gKTtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICBjb252X3V0aWwuZWl0aGVyU3RyaWRlc09yRGlsYXRpb25zQXJlT25lKHN0cmlkZXMsIGRpbGF0aW9ucyksXG4gICAgICAoKSA9PiAnRXJyb3IgaW4gY29udjJEOiBFaXRoZXIgc3RyaWRlcyBvciBkaWxhdGlvbnMgbXVzdCBiZSAxLiAnICtcbiAgICAgICAgICBgR290IHN0cmlkZXMgJHtzdHJpZGVzfSBhbmQgZGlsYXRpb25zICcke2RpbGF0aW9uc30nYCk7XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgY29udl91dGlsLnN0cmlkZXNPckRpbGF0aW9uc0FyZVBvc2l0aXZlKGRpbGF0aW9ucyksXG4gICAgICAoKSA9PiAnRXJyb3IgaW4gY29udjJEOiBEaWxhdGVkIHJhdGVzIHNob3VsZCBiZSBsYXJnZXIgdGhhbiAwLicpO1xuICB1dGlsLmFzc2VydChcbiAgICAgIGNvbnZfdXRpbC5zdHJpZGVzT3JEaWxhdGlvbnNBcmVQb3NpdGl2ZShzdHJpZGVzKSxcbiAgICAgICgpID0+ICdFcnJvciBpbiBjb252MkQ6IFN0cmlkZXMgc2hvdWxkIGJlIGxhcmdlciB0aGFuIDAuJyk7XG5cbiAgY29uc3QgaW5wdXRzOiBDb252MkRJbnB1dHMgPSB7eDogeDRELCBmaWx0ZXI6ICRmaWx0ZXJ9O1xuICBjb25zdCBhdHRyczpcbiAgICAgIENvbnYyREF0dHJzID0ge3N0cmlkZXMsIHBhZCwgZGF0YUZvcm1hdCwgZGlsYXRpb25zLCBkaW1Sb3VuZGluZ01vZGV9O1xuXG4gIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTogbm8tdW5uZWNlc3NhcnktdHlwZS1hc3NlcnRpb25cbiAgY29uc3QgcmVzID0gRU5HSU5FLnJ1bktlcm5lbChcbiAgICAgICAgICAgICAgICAgIENvbnYyRCwgaW5wdXRzIGFzIHVua25vd24gYXMgTmFtZWRUZW5zb3JNYXAsXG4gICAgICAgICAgICAgICAgICBhdHRycyBhcyB1bmtub3duIGFzIE5hbWVkQXR0ck1hcCkgYXMgVDtcblxuICBpZiAocmVzaGFwZWRUbzREKSB7XG4gICAgcmV0dXJuIHJlc2hhcGUocmVzLCBbcmVzLnNoYXBlWzFdLCByZXMuc2hhcGVbMl0sIHJlcy5zaGFwZVszXV0pIGFzIFQ7XG4gIH1cbiAgcmV0dXJuIHJlcztcbn1cblxuZXhwb3J0IGNvbnN0IGNvbnYyZCA9IC8qIEBfX1BVUkVfXyAqLyBvcCh7Y29udjJkX30pO1xuIl19