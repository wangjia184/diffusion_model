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
import { MaxPool3D } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import * as util from '../util';
import { checkPadOnDimRoundingMode } from './conv_util';
import { op } from './operation';
import { reshape } from './reshape';
/**
 * Computes the 3D max pooling.
 *
 * ```js
 * const x = tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]);
 * const result = tf.maxPool3d(x, 2, 1, 'valid');
 * result.print();
 * ```
 *
 * @param x The input tensor, of rank 5 or rank 4 of shape
 *     `[batch, depth, height, width, inChannels]`.
 * @param filterSize The filter size:
 *     `[filterDepth, filterHeight, filterWidth]`.
 *     If `filterSize` is a single number,
 *     then `filterDepth == filterHeight == filterWidth`.
 * @param strides The strides of the pooling:
 *     `[strideDepth, strideHeight, strideWidth]`.
 *     If `strides` is a single number,
 *     then `strideDepth == strideHeight == strideWidth`.
 * @param pad The type of padding algorithm.
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1*1x1.
 *    - For more info, see this guide:
 *     [https://www.tensorflow.org/api_docs/python/tf/nn/convolution](
 *          https://www.tensorflow.org/api_docs/python/tf/nn/convolution)
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 * @param dataFormat An optional string from: "NDHWC", "NCDHW". Defaults to
 *     "NDHWC". Specify the data format of the input and output data. With the
 *     default format "NDHWC", the data is stored in the order of: [batch,
 *     depth, height, width, channels]. Only "NDHWC" is currently supported.
 * @doc {heading: 'Operations', subheading: 'Convolution'}
 */
function maxPool3d_(x, filterSize = [1, 1, 1], strides, pad, dimRoundingMode, dataFormat = 'NDHWC') {
    const $x = convertToTensor(x, 'x', 'maxPool3d');
    let x5D = $x;
    let reshapedTo5D = false;
    if ($x.rank === 4) {
        reshapedTo5D = true;
        x5D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2], $x.shape[3]]);
    }
    util.assert(x5D.rank === 5, () => `Error in maxPool3d: x must be rank 5 but got rank ${x5D.rank}.`);
    util.assert(dataFormat === 'NDHWC', () => `Error in maxPool3d: Only NDHWC is currently supported, ` +
        `but got dataFormat of ${dataFormat}`);
    checkPadOnDimRoundingMode('maxPool3d', pad, dimRoundingMode);
    const inputs = { x: x5D };
    const attrs = { filterSize, strides, pad, dimRoundingMode, dataFormat };
    // tslint:disable-next-line: no-unnecessary-type-assertion
    const res = ENGINE.runKernel(MaxPool3D, inputs, attrs);
    if (reshapedTo5D) {
        return reshape(res, [res.shape[1], res.shape[2], res.shape[3], res.shape[4]]);
    }
    return res;
}
export const maxPool3d = /* @__PURE__ */ op({ maxPool3d_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibWF4X3Bvb2xfM2QuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9tYXhfcG9vbF8zZC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQ2pDLE9BQU8sRUFBQyxTQUFTLEVBQWtDLE1BQU0saUJBQWlCLENBQUM7QUFJM0UsT0FBTyxFQUFDLGVBQWUsRUFBQyxNQUFNLG9CQUFvQixDQUFDO0FBRW5ELE9BQU8sS0FBSyxJQUFJLE1BQU0sU0FBUyxDQUFDO0FBRWhDLE9BQU8sRUFBQyx5QkFBeUIsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUN0RCxPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBQy9CLE9BQU8sRUFBQyxPQUFPLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFFbEM7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7R0FrQ0c7QUFDSCxTQUFTLFVBQVUsQ0FDZixDQUFlLEVBQUUsYUFBOEMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUN4RSxPQUF3QyxFQUFFLEdBQTBCLEVBQ3BFLGVBQXdDLEVBQ3hDLGFBQThCLE9BQU87SUFDdkMsTUFBTSxFQUFFLEdBQUcsZUFBZSxDQUFDLENBQUMsRUFBRSxHQUFHLEVBQUUsV0FBVyxDQUFDLENBQUM7SUFFaEQsSUFBSSxHQUFHLEdBQUcsRUFBYyxDQUFDO0lBQ3pCLElBQUksWUFBWSxHQUFHLEtBQUssQ0FBQztJQUN6QixJQUFJLEVBQUUsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUFFO1FBQ2pCLFlBQVksR0FBRyxJQUFJLENBQUM7UUFDcEIsR0FBRyxHQUFHLE9BQU8sQ0FBQyxFQUFFLEVBQUUsQ0FBQyxDQUFDLEVBQUUsRUFBRSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsRUFBRSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7S0FDNUU7SUFFRCxJQUFJLENBQUMsTUFBTSxDQUNQLEdBQUcsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNkLEdBQUcsRUFBRSxDQUFDLHFEQUFxRCxHQUFHLENBQUMsSUFBSSxHQUFHLENBQUMsQ0FBQztJQUM1RSxJQUFJLENBQUMsTUFBTSxDQUNQLFVBQVUsS0FBSyxPQUFPLEVBQ3RCLEdBQUcsRUFBRSxDQUFDLHlEQUF5RDtRQUMzRCx5QkFBeUIsVUFBVSxFQUFFLENBQUMsQ0FBQztJQUMvQyx5QkFBeUIsQ0FBQyxXQUFXLEVBQUUsR0FBRyxFQUFFLGVBQWUsQ0FBQyxDQUFDO0lBQzdELE1BQU0sTUFBTSxHQUFvQixFQUFDLENBQUMsRUFBRSxHQUFHLEVBQUMsQ0FBQztJQUN6QyxNQUFNLEtBQUssR0FDVSxFQUFDLFVBQVUsRUFBRSxPQUFPLEVBQUUsR0FBRyxFQUFFLGVBQWUsRUFBRSxVQUFVLEVBQUMsQ0FBQztJQUU3RSwwREFBMEQ7SUFDMUQsTUFBTSxHQUFHLEdBQUcsTUFBTSxDQUFDLFNBQVMsQ0FDWixTQUFTLEVBQUUsTUFBbUMsRUFDOUMsS0FBZ0MsQ0FBTSxDQUFDO0lBRXZELElBQUksWUFBWSxFQUFFO1FBQ2hCLE9BQU8sT0FBTyxDQUNILEdBQUcsRUFBRSxDQUFDLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxHQUFHLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FDbkUsQ0FBQztLQUNQO0lBRUQsT0FBTyxHQUFHLENBQUM7QUFDYixDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sU0FBUyxHQUFHLGVBQWUsQ0FBQyxFQUFFLENBQUMsRUFBQyxVQUFVLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0VOR0lORX0gZnJvbSAnLi4vZW5naW5lJztcbmltcG9ydCB7TWF4UG9vbDNELCBNYXhQb29sM0RBdHRycywgTWF4UG9vbDNESW5wdXRzfSBmcm9tICcuLi9rZXJuZWxfbmFtZXMnO1xuaW1wb3J0IHtOYW1lZEF0dHJNYXB9IGZyb20gJy4uL2tlcm5lbF9yZWdpc3RyeSc7XG5pbXBvcnQge1RlbnNvcjRELCBUZW5zb3I1RH0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7TmFtZWRUZW5zb3JNYXB9IGZyb20gJy4uL3RlbnNvcl90eXBlcyc7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7VGVuc29yTGlrZX0gZnJvbSAnLi4vdHlwZXMnO1xuaW1wb3J0ICogYXMgdXRpbCBmcm9tICcuLi91dGlsJztcblxuaW1wb3J0IHtjaGVja1BhZE9uRGltUm91bmRpbmdNb2RlfSBmcm9tICcuL2NvbnZfdXRpbCc7XG5pbXBvcnQge29wfSBmcm9tICcuL29wZXJhdGlvbic7XG5pbXBvcnQge3Jlc2hhcGV9IGZyb20gJy4vcmVzaGFwZSc7XG5cbi8qKlxuICogQ29tcHV0ZXMgdGhlIDNEIG1heCBwb29saW5nLlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCB4ID0gdGYudGVuc29yNWQoWzEsIDIsIDMsIDQsIDUsIDYsIDcsIDhdLCBbMSwgMiwgMiwgMiwgMV0pO1xuICogY29uc3QgcmVzdWx0ID0gdGYubWF4UG9vbDNkKHgsIDIsIDEsICd2YWxpZCcpO1xuICogcmVzdWx0LnByaW50KCk7XG4gKiBgYGBcbiAqXG4gKiBAcGFyYW0geCBUaGUgaW5wdXQgdGVuc29yLCBvZiByYW5rIDUgb3IgcmFuayA0IG9mIHNoYXBlXG4gKiAgICAgYFtiYXRjaCwgZGVwdGgsIGhlaWdodCwgd2lkdGgsIGluQ2hhbm5lbHNdYC5cbiAqIEBwYXJhbSBmaWx0ZXJTaXplIFRoZSBmaWx0ZXIgc2l6ZTpcbiAqICAgICBgW2ZpbHRlckRlcHRoLCBmaWx0ZXJIZWlnaHQsIGZpbHRlcldpZHRoXWAuXG4gKiAgICAgSWYgYGZpbHRlclNpemVgIGlzIGEgc2luZ2xlIG51bWJlcixcbiAqICAgICB0aGVuIGBmaWx0ZXJEZXB0aCA9PSBmaWx0ZXJIZWlnaHQgPT0gZmlsdGVyV2lkdGhgLlxuICogQHBhcmFtIHN0cmlkZXMgVGhlIHN0cmlkZXMgb2YgdGhlIHBvb2xpbmc6XG4gKiAgICAgYFtzdHJpZGVEZXB0aCwgc3RyaWRlSGVpZ2h0LCBzdHJpZGVXaWR0aF1gLlxuICogICAgIElmIGBzdHJpZGVzYCBpcyBhIHNpbmdsZSBudW1iZXIsXG4gKiAgICAgdGhlbiBgc3RyaWRlRGVwdGggPT0gc3RyaWRlSGVpZ2h0ID09IHN0cmlkZVdpZHRoYC5cbiAqIEBwYXJhbSBwYWQgVGhlIHR5cGUgb2YgcGFkZGluZyBhbGdvcml0aG0uXG4gKiAgICAtIGBzYW1lYCBhbmQgc3RyaWRlIDE6IG91dHB1dCB3aWxsIGJlIG9mIHNhbWUgc2l6ZSBhcyBpbnB1dCxcbiAqICAgICAgIHJlZ2FyZGxlc3Mgb2YgZmlsdGVyIHNpemUuXG4gKiAgICAtIGB2YWxpZGA6IG91dHB1dCB3aWxsIGJlIHNtYWxsZXIgdGhhbiBpbnB1dCBpZiBmaWx0ZXIgaXMgbGFyZ2VyXG4gKiAgICAgICB0aGFuIDEqMXgxLlxuICogICAgLSBGb3IgbW9yZSBpbmZvLCBzZWUgdGhpcyBndWlkZTpcbiAqICAgICBbaHR0cHM6Ly93d3cudGVuc29yZmxvdy5vcmcvYXBpX2RvY3MvcHl0aG9uL3RmL25uL2NvbnZvbHV0aW9uXShcbiAqICAgICAgICAgIGh0dHBzOi8vd3d3LnRlbnNvcmZsb3cub3JnL2FwaV9kb2NzL3B5dGhvbi90Zi9ubi9jb252b2x1dGlvbilcbiAqIEBwYXJhbSBkaW1Sb3VuZGluZ01vZGUgQSBzdHJpbmcgZnJvbTogJ2NlaWwnLCAncm91bmQnLCAnZmxvb3InLiBJZiBub25lIGlzXG4gKiAgICAgcHJvdmlkZWQsIGl0IHdpbGwgZGVmYXVsdCB0byB0cnVuY2F0ZS5cbiAqIEBwYXJhbSBkYXRhRm9ybWF0IEFuIG9wdGlvbmFsIHN0cmluZyBmcm9tOiBcIk5ESFdDXCIsIFwiTkNESFdcIi4gRGVmYXVsdHMgdG9cbiAqICAgICBcIk5ESFdDXCIuIFNwZWNpZnkgdGhlIGRhdGEgZm9ybWF0IG9mIHRoZSBpbnB1dCBhbmQgb3V0cHV0IGRhdGEuIFdpdGggdGhlXG4gKiAgICAgZGVmYXVsdCBmb3JtYXQgXCJOREhXQ1wiLCB0aGUgZGF0YSBpcyBzdG9yZWQgaW4gdGhlIG9yZGVyIG9mOiBbYmF0Y2gsXG4gKiAgICAgZGVwdGgsIGhlaWdodCwgd2lkdGgsIGNoYW5uZWxzXS4gT25seSBcIk5ESFdDXCIgaXMgY3VycmVudGx5IHN1cHBvcnRlZC5cbiAqIEBkb2Mge2hlYWRpbmc6ICdPcGVyYXRpb25zJywgc3ViaGVhZGluZzogJ0NvbnZvbHV0aW9uJ31cbiAqL1xuZnVuY3Rpb24gbWF4UG9vbDNkXzxUIGV4dGVuZHMgVGVuc29yNER8VGVuc29yNUQ+KFxuICAgIHg6IFR8VGVuc29yTGlrZSwgZmlsdGVyU2l6ZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdfG51bWJlciA9IFsxLCAxLCAxXSxcbiAgICBzdHJpZGVzOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl18bnVtYmVyLCBwYWQ6ICd2YWxpZCd8J3NhbWUnfG51bWJlcixcbiAgICBkaW1Sb3VuZGluZ01vZGU/OiAnZmxvb3InfCdyb3VuZCd8J2NlaWwnLFxuICAgIGRhdGFGb3JtYXQ6ICdOREhXQyd8J05DREhXJyA9ICdOREhXQycpOiBUIHtcbiAgY29uc3QgJHggPSBjb252ZXJ0VG9UZW5zb3IoeCwgJ3gnLCAnbWF4UG9vbDNkJyk7XG5cbiAgbGV0IHg1RCA9ICR4IGFzIFRlbnNvcjVEO1xuICBsZXQgcmVzaGFwZWRUbzVEID0gZmFsc2U7XG4gIGlmICgkeC5yYW5rID09PSA0KSB7XG4gICAgcmVzaGFwZWRUbzVEID0gdHJ1ZTtcbiAgICB4NUQgPSByZXNoYXBlKCR4LCBbMSwgJHguc2hhcGVbMF0sICR4LnNoYXBlWzFdLCAkeC5zaGFwZVsyXSwgJHguc2hhcGVbM11dKTtcbiAgfVxuXG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgeDVELnJhbmsgPT09IDUsXG4gICAgICAoKSA9PiBgRXJyb3IgaW4gbWF4UG9vbDNkOiB4IG11c3QgYmUgcmFuayA1IGJ1dCBnb3QgcmFuayAke3g1RC5yYW5rfS5gKTtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICBkYXRhRm9ybWF0ID09PSAnTkRIV0MnLFxuICAgICAgKCkgPT4gYEVycm9yIGluIG1heFBvb2wzZDogT25seSBOREhXQyBpcyBjdXJyZW50bHkgc3VwcG9ydGVkLCBgICtcbiAgICAgICAgICBgYnV0IGdvdCBkYXRhRm9ybWF0IG9mICR7ZGF0YUZvcm1hdH1gKTtcbiAgY2hlY2tQYWRPbkRpbVJvdW5kaW5nTW9kZSgnbWF4UG9vbDNkJywgcGFkLCBkaW1Sb3VuZGluZ01vZGUpO1xuICBjb25zdCBpbnB1dHM6IE1heFBvb2wzRElucHV0cyA9IHt4OiB4NUR9O1xuICBjb25zdCBhdHRyczpcbiAgICAgIE1heFBvb2wzREF0dHJzID0ge2ZpbHRlclNpemUsIHN0cmlkZXMsIHBhZCwgZGltUm91bmRpbmdNb2RlLCBkYXRhRm9ybWF0fTtcblxuICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6IG5vLXVubmVjZXNzYXJ5LXR5cGUtYXNzZXJ0aW9uXG4gIGNvbnN0IHJlcyA9IEVOR0lORS5ydW5LZXJuZWwoXG4gICAgICAgICAgICAgICAgICBNYXhQb29sM0QsIGlucHV0cyBhcyB1bmtub3duIGFzIE5hbWVkVGVuc29yTWFwLFxuICAgICAgICAgICAgICAgICAgYXR0cnMgYXMgdW5rbm93biBhcyBOYW1lZEF0dHJNYXApIGFzIFQ7XG5cbiAgaWYgKHJlc2hhcGVkVG81RCkge1xuICAgIHJldHVybiByZXNoYXBlKFxuICAgICAgICAgICAgICAgcmVzLCBbcmVzLnNoYXBlWzFdLCByZXMuc2hhcGVbMl0sIHJlcy5zaGFwZVszXSwgcmVzLnNoYXBlWzRdXSkgYXNcbiAgICAgICAgVDtcbiAgfVxuXG4gIHJldHVybiByZXM7XG59XG5cbmV4cG9ydCBjb25zdCBtYXhQb29sM2QgPSAvKiBAX19QVVJFX18gKi8gb3Aoe21heFBvb2wzZF99KTtcbiJdfQ==