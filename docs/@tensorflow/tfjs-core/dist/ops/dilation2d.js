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
import { Dilation2D } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import * as util from '../util';
import { op } from './operation';
import { reshape } from './reshape';
/**
 * Computes the grayscale dilation over the input `x`.
 *
 * @param x The input tensor, rank 3 or rank 4 of shape
 *     `[batch, height, width, depth]`. If rank 3, batch of 1 is assumed.
 * @param filter The filter tensor, rank 3, of shape
 *     `[filterHeight, filterWidth, depth]`.
 * @param strides The strides of the sliding window for each dimension of the
 *     input tensor: `[strideHeight, strideWidth]`.
 *     If `strides` is a single number,
 *     then `strideHeight == strideWidth`.
 * @param pad The type of padding algorithm.
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1*1x1.
 *    - For more info, see this guide:
 *     [https://www.tensorflow.org/api_docs/python/tf/nn/convolution](
 *          https://www.tensorflow.org/api_docs/python/tf/nn/convolution)
 * @param dataFormat Specify the data format of the input and output data.
 *      Defaults to 'NHWC'. Only 'NHWC' is currently supported. With the
 *      default format "NHWC", the data is stored in the order of: [batch,
 *      height, width, channels].
 * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`
 *     in which we sample input values across the height and width dimensions
 *     for atrous morphological dilation. Defaults to `[1, 1]`. If `dilations`
 *     is a single number, then `dilationHeight == dilationWidth`. If it is
 *     greater than 1, then all values of `strides` must be 1.
 *
 * @doc {heading: 'Operations', subheading: 'Convolution'}
 */
function dilation2d_(x, filter, strides, pad, dilations = [1, 1], dataFormat = 'NHWC') {
    const $x = convertToTensor(x, 'x', 'dilation2d');
    const $filter = convertToTensor(filter, 'filter', 'dilation2d');
    util.assert($x.rank === 3 || $x.rank === 4, () => `Error in dilation2d: input must be rank 3 or 4, but got rank ` +
        `${$x.rank}.`);
    util.assert($filter.rank === 3, () => `Error in dilation2d: filter must be rank 3, but got rank ` +
        `${$filter.rank}.`);
    util.assert(dataFormat === 'NHWC', () => `Error in dilation2d: Only NHWC is currently supported, ` +
        `but got dataFormat of ${dataFormat}`);
    let x4D = $x;
    let reshapedTo4D = false;
    if ($x.rank === 3) {
        x4D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2]]);
        reshapedTo4D = true;
    }
    util.assert(x4D.shape[3] === $filter.shape[2], () => `Error in dilation2d:  input and filter must have the same depth: ${x4D.shape[3]} vs ${$filter.shape[2]}`);
    const inputs = { x: x4D, filter: $filter };
    const attrs = { strides, pad, dilations };
    // tslint:disable-next-line: no-unnecessary-type-assertion
    const res = ENGINE.runKernel(Dilation2D, inputs, attrs);
    if (reshapedTo4D) {
        return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]);
    }
    return res;
}
export const dilation2d = /* @__PURE__ */ op({ dilation2d_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZGlsYXRpb24yZC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL2RpbGF0aW9uMmQudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMsVUFBVSxFQUFvQyxNQUFNLGlCQUFpQixDQUFDO0FBSTlFLE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUVuRCxPQUFPLEtBQUssSUFBSSxNQUFNLFNBQVMsQ0FBQztBQUVoQyxPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBQy9CLE9BQU8sRUFBQyxPQUFPLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFFbEM7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQThCRztBQUNILFNBQVMsV0FBVyxDQUNoQixDQUFlLEVBQUUsTUFBMkIsRUFDNUMsT0FBZ0MsRUFBRSxHQUFtQixFQUNyRCxZQUFxQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFDM0MsYUFBcUIsTUFBTTtJQUM3QixNQUFNLEVBQUUsR0FBRyxlQUFlLENBQUMsQ0FBQyxFQUFFLEdBQUcsRUFBRSxZQUFZLENBQUMsQ0FBQztJQUNqRCxNQUFNLE9BQU8sR0FBRyxlQUFlLENBQUMsTUFBTSxFQUFFLFFBQVEsRUFBRSxZQUFZLENBQUMsQ0FBQztJQUVoRSxJQUFJLENBQUMsTUFBTSxDQUNQLEVBQUUsQ0FBQyxJQUFJLEtBQUssQ0FBQyxJQUFJLEVBQUUsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUM5QixHQUFHLEVBQUUsQ0FBQywrREFBK0Q7UUFDakUsR0FBRyxFQUFFLENBQUMsSUFBSSxHQUFHLENBQUMsQ0FBQztJQUN2QixJQUFJLENBQUMsTUFBTSxDQUNQLE9BQU8sQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNsQixHQUFHLEVBQUUsQ0FBQywyREFBMkQ7UUFDN0QsR0FBRyxPQUFPLENBQUMsSUFBSSxHQUFHLENBQUMsQ0FBQztJQUM1QixJQUFJLENBQUMsTUFBTSxDQUNQLFVBQVUsS0FBSyxNQUFNLEVBQ3JCLEdBQUcsRUFBRSxDQUFDLHlEQUF5RDtRQUMzRCx5QkFBeUIsVUFBVSxFQUFFLENBQUMsQ0FBQztJQUUvQyxJQUFJLEdBQUcsR0FBRyxFQUFjLENBQUM7SUFDekIsSUFBSSxZQUFZLEdBQUcsS0FBSyxDQUFDO0lBRXpCLElBQUksRUFBRSxDQUFDLElBQUksS0FBSyxDQUFDLEVBQUU7UUFDakIsR0FBRyxHQUFHLE9BQU8sQ0FBQyxFQUFFLEVBQUUsQ0FBQyxDQUFDLEVBQUUsRUFBRSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQzlELFlBQVksR0FBRyxJQUFJLENBQUM7S0FDckI7SUFFRCxJQUFJLENBQUMsTUFBTSxDQUNQLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEtBQUssT0FBTyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFDakMsR0FBRyxFQUFFLENBQUMsb0VBQ0YsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsT0FBTyxPQUFPLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQztJQUUvQyxNQUFNLE1BQU0sR0FBcUIsRUFBQyxDQUFDLEVBQUUsR0FBRyxFQUFFLE1BQU0sRUFBRSxPQUFPLEVBQUMsQ0FBQztJQUMzRCxNQUFNLEtBQUssR0FBb0IsRUFBQyxPQUFPLEVBQUUsR0FBRyxFQUFFLFNBQVMsRUFBQyxDQUFDO0lBRXpELDBEQUEwRDtJQUMxRCxNQUFNLEdBQUcsR0FBRyxNQUFNLENBQUMsU0FBUyxDQUNaLFVBQVUsRUFBRSxNQUFtQyxFQUMvQyxLQUFnQyxDQUFNLENBQUM7SUFFdkQsSUFBSSxZQUFZLEVBQUU7UUFDaEIsT0FBTyxPQUFPLENBQUMsR0FBRyxFQUFFLENBQUMsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxHQUFHLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBTSxDQUFDO0tBQ3RFO0lBRUQsT0FBTyxHQUFHLENBQUM7QUFDYixDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sVUFBVSxHQUFHLGVBQWUsQ0FBQyxFQUFFLENBQUMsRUFBQyxXQUFXLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0VOR0lORX0gZnJvbSAnLi4vZW5naW5lJztcbmltcG9ydCB7RGlsYXRpb24yRCwgRGlsYXRpb24yREF0dHJzLCBEaWxhdGlvbjJESW5wdXRzfSBmcm9tICcuLi9rZXJuZWxfbmFtZXMnO1xuaW1wb3J0IHtOYW1lZEF0dHJNYXB9IGZyb20gJy4uL2tlcm5lbF9yZWdpc3RyeSc7XG5pbXBvcnQge1RlbnNvcjNELCBUZW5zb3I0RH0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7TmFtZWRUZW5zb3JNYXB9IGZyb20gJy4uL3RlbnNvcl90eXBlcyc7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7VGVuc29yTGlrZX0gZnJvbSAnLi4vdHlwZXMnO1xuaW1wb3J0ICogYXMgdXRpbCBmcm9tICcuLi91dGlsJztcblxuaW1wb3J0IHtvcH0gZnJvbSAnLi9vcGVyYXRpb24nO1xuaW1wb3J0IHtyZXNoYXBlfSBmcm9tICcuL3Jlc2hhcGUnO1xuXG4vKipcbiAqIENvbXB1dGVzIHRoZSBncmF5c2NhbGUgZGlsYXRpb24gb3ZlciB0aGUgaW5wdXQgYHhgLlxuICpcbiAqIEBwYXJhbSB4IFRoZSBpbnB1dCB0ZW5zb3IsIHJhbmsgMyBvciByYW5rIDQgb2Ygc2hhcGVcbiAqICAgICBgW2JhdGNoLCBoZWlnaHQsIHdpZHRoLCBkZXB0aF1gLiBJZiByYW5rIDMsIGJhdGNoIG9mIDEgaXMgYXNzdW1lZC5cbiAqIEBwYXJhbSBmaWx0ZXIgVGhlIGZpbHRlciB0ZW5zb3IsIHJhbmsgMywgb2Ygc2hhcGVcbiAqICAgICBgW2ZpbHRlckhlaWdodCwgZmlsdGVyV2lkdGgsIGRlcHRoXWAuXG4gKiBAcGFyYW0gc3RyaWRlcyBUaGUgc3RyaWRlcyBvZiB0aGUgc2xpZGluZyB3aW5kb3cgZm9yIGVhY2ggZGltZW5zaW9uIG9mIHRoZVxuICogICAgIGlucHV0IHRlbnNvcjogYFtzdHJpZGVIZWlnaHQsIHN0cmlkZVdpZHRoXWAuXG4gKiAgICAgSWYgYHN0cmlkZXNgIGlzIGEgc2luZ2xlIG51bWJlcixcbiAqICAgICB0aGVuIGBzdHJpZGVIZWlnaHQgPT0gc3RyaWRlV2lkdGhgLlxuICogQHBhcmFtIHBhZCBUaGUgdHlwZSBvZiBwYWRkaW5nIGFsZ29yaXRobS5cbiAqICAgIC0gYHNhbWVgIGFuZCBzdHJpZGUgMTogb3V0cHV0IHdpbGwgYmUgb2Ygc2FtZSBzaXplIGFzIGlucHV0LFxuICogICAgICAgcmVnYXJkbGVzcyBvZiBmaWx0ZXIgc2l6ZS5cbiAqICAgIC0gYHZhbGlkYDogb3V0cHV0IHdpbGwgYmUgc21hbGxlciB0aGFuIGlucHV0IGlmIGZpbHRlciBpcyBsYXJnZXJcbiAqICAgICAgIHRoYW4gMSoxeDEuXG4gKiAgICAtIEZvciBtb3JlIGluZm8sIHNlZSB0aGlzIGd1aWRlOlxuICogICAgIFtodHRwczovL3d3dy50ZW5zb3JmbG93Lm9yZy9hcGlfZG9jcy9weXRob24vdGYvbm4vY29udm9sdXRpb25dKFxuICogICAgICAgICAgaHR0cHM6Ly93d3cudGVuc29yZmxvdy5vcmcvYXBpX2RvY3MvcHl0aG9uL3RmL25uL2NvbnZvbHV0aW9uKVxuICogQHBhcmFtIGRhdGFGb3JtYXQgU3BlY2lmeSB0aGUgZGF0YSBmb3JtYXQgb2YgdGhlIGlucHV0IGFuZCBvdXRwdXQgZGF0YS5cbiAqICAgICAgRGVmYXVsdHMgdG8gJ05IV0MnLiBPbmx5ICdOSFdDJyBpcyBjdXJyZW50bHkgc3VwcG9ydGVkLiBXaXRoIHRoZVxuICogICAgICBkZWZhdWx0IGZvcm1hdCBcIk5IV0NcIiwgdGhlIGRhdGEgaXMgc3RvcmVkIGluIHRoZSBvcmRlciBvZjogW2JhdGNoLFxuICogICAgICBoZWlnaHQsIHdpZHRoLCBjaGFubmVsc10uXG4gKiBAcGFyYW0gZGlsYXRpb25zIFRoZSBkaWxhdGlvbiByYXRlczogYFtkaWxhdGlvbkhlaWdodCwgZGlsYXRpb25XaWR0aF1gXG4gKiAgICAgaW4gd2hpY2ggd2Ugc2FtcGxlIGlucHV0IHZhbHVlcyBhY3Jvc3MgdGhlIGhlaWdodCBhbmQgd2lkdGggZGltZW5zaW9uc1xuICogICAgIGZvciBhdHJvdXMgbW9ycGhvbG9naWNhbCBkaWxhdGlvbi4gRGVmYXVsdHMgdG8gYFsxLCAxXWAuIElmIGBkaWxhdGlvbnNgXG4gKiAgICAgaXMgYSBzaW5nbGUgbnVtYmVyLCB0aGVuIGBkaWxhdGlvbkhlaWdodCA9PSBkaWxhdGlvbldpZHRoYC4gSWYgaXQgaXNcbiAqICAgICBncmVhdGVyIHRoYW4gMSwgdGhlbiBhbGwgdmFsdWVzIG9mIGBzdHJpZGVzYCBtdXN0IGJlIDEuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ09wZXJhdGlvbnMnLCBzdWJoZWFkaW5nOiAnQ29udm9sdXRpb24nfVxuICovXG5mdW5jdGlvbiBkaWxhdGlvbjJkXzxUIGV4dGVuZHMgVGVuc29yM0R8VGVuc29yNEQ+KFxuICAgIHg6IFR8VGVuc29yTGlrZSwgZmlsdGVyOiBUZW5zb3IzRHxUZW5zb3JMaWtlLFxuICAgIHN0cmlkZXM6IFtudW1iZXIsIG51bWJlcl18bnVtYmVyLCBwYWQ6ICd2YWxpZCd8J3NhbWUnLFxuICAgIGRpbGF0aW9uczogW251bWJlciwgbnVtYmVyXXxudW1iZXIgPSBbMSwgMV0sXG4gICAgZGF0YUZvcm1hdDogJ05IV0MnID0gJ05IV0MnKTogVCB7XG4gIGNvbnN0ICR4ID0gY29udmVydFRvVGVuc29yKHgsICd4JywgJ2RpbGF0aW9uMmQnKTtcbiAgY29uc3QgJGZpbHRlciA9IGNvbnZlcnRUb1RlbnNvcihmaWx0ZXIsICdmaWx0ZXInLCAnZGlsYXRpb24yZCcpO1xuXG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgJHgucmFuayA9PT0gMyB8fCAkeC5yYW5rID09PSA0LFxuICAgICAgKCkgPT4gYEVycm9yIGluIGRpbGF0aW9uMmQ6IGlucHV0IG11c3QgYmUgcmFuayAzIG9yIDQsIGJ1dCBnb3QgcmFuayBgICtcbiAgICAgICAgICBgJHskeC5yYW5rfS5gKTtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICAkZmlsdGVyLnJhbmsgPT09IDMsXG4gICAgICAoKSA9PiBgRXJyb3IgaW4gZGlsYXRpb24yZDogZmlsdGVyIG11c3QgYmUgcmFuayAzLCBidXQgZ290IHJhbmsgYCArXG4gICAgICAgICAgYCR7JGZpbHRlci5yYW5rfS5gKTtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICBkYXRhRm9ybWF0ID09PSAnTkhXQycsXG4gICAgICAoKSA9PiBgRXJyb3IgaW4gZGlsYXRpb24yZDogT25seSBOSFdDIGlzIGN1cnJlbnRseSBzdXBwb3J0ZWQsIGAgK1xuICAgICAgICAgIGBidXQgZ290IGRhdGFGb3JtYXQgb2YgJHtkYXRhRm9ybWF0fWApO1xuXG4gIGxldCB4NEQgPSAkeCBhcyBUZW5zb3I0RDtcbiAgbGV0IHJlc2hhcGVkVG80RCA9IGZhbHNlO1xuXG4gIGlmICgkeC5yYW5rID09PSAzKSB7XG4gICAgeDREID0gcmVzaGFwZSgkeCwgWzEsICR4LnNoYXBlWzBdLCAkeC5zaGFwZVsxXSwgJHguc2hhcGVbMl1dKTtcbiAgICByZXNoYXBlZFRvNEQgPSB0cnVlO1xuICB9XG5cbiAgdXRpbC5hc3NlcnQoXG4gICAgICB4NEQuc2hhcGVbM10gPT09ICRmaWx0ZXIuc2hhcGVbMl0sXG4gICAgICAoKSA9PiBgRXJyb3IgaW4gZGlsYXRpb24yZDogIGlucHV0IGFuZCBmaWx0ZXIgbXVzdCBoYXZlIHRoZSBzYW1lIGRlcHRoOiAke1xuICAgICAgICAgIHg0RC5zaGFwZVszXX0gdnMgJHskZmlsdGVyLnNoYXBlWzJdfWApO1xuXG4gIGNvbnN0IGlucHV0czogRGlsYXRpb24yRElucHV0cyA9IHt4OiB4NEQsIGZpbHRlcjogJGZpbHRlcn07XG4gIGNvbnN0IGF0dHJzOiBEaWxhdGlvbjJEQXR0cnMgPSB7c3RyaWRlcywgcGFkLCBkaWxhdGlvbnN9O1xuXG4gIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTogbm8tdW5uZWNlc3NhcnktdHlwZS1hc3NlcnRpb25cbiAgY29uc3QgcmVzID0gRU5HSU5FLnJ1bktlcm5lbChcbiAgICAgICAgICAgICAgICAgIERpbGF0aW9uMkQsIGlucHV0cyBhcyB1bmtub3duIGFzIE5hbWVkVGVuc29yTWFwLFxuICAgICAgICAgICAgICAgICAgYXR0cnMgYXMgdW5rbm93biBhcyBOYW1lZEF0dHJNYXApIGFzIFQ7XG5cbiAgaWYgKHJlc2hhcGVkVG80RCkge1xuICAgIHJldHVybiByZXNoYXBlKHJlcywgW3Jlcy5zaGFwZVsxXSwgcmVzLnNoYXBlWzJdLCByZXMuc2hhcGVbM11dKSBhcyBUO1xuICB9XG5cbiAgcmV0dXJuIHJlcztcbn1cblxuZXhwb3J0IGNvbnN0IGRpbGF0aW9uMmQgPSAvKiBAX19QVVJFX18gKi8gb3Aoe2RpbGF0aW9uMmRffSk7XG4iXX0=