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
import { DepthwiseConv2dNative } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import * as util from '../util';
import * as conv_util from './conv_util';
import { op } from './operation';
import { reshape } from './reshape';
/**
 * Depthwise 2D convolution.
 *
 * Given a 4D `input` array and a `filter` array of shape
 * `[filterHeight, filterWidth, inChannels, channelMultiplier]` containing
 * `inChannels` convolutional filters of depth 1, this op applies a
 * different filter to each input channel (expanding from 1 channel to
 * `channelMultiplier` channels for each), then concatenates the results
 * together. The output has `inChannels * channelMultiplier` channels.
 *
 * See
 * [https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d](
 *     https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d)
 * for more details.
 *
 * @param x The input tensor, of rank 4 or rank 3, of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is
 * assumed.
 * @param filter The filter tensor, rank 4, of shape
 *     `[filterHeight, filterWidth, inChannels, channelMultiplier]`.
 * @param strides The strides of the convolution: `[strideHeight,
 * strideWidth]`. If strides is a single number, then `strideHeight ==
 * strideWidth`.
 * @param pad The type of padding algorithm.
 *   - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *   - `valid`: output will be smaller than input if filter is larger
 *       than 1x1.
 *   - For more info, see this guide:
 *     [https://www.tensorflow.org/api_docs/python/tf/nn/convolution](
 *          https://www.tensorflow.org/api_docs/python/tf/nn/convolution)
 * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`
 *     in which we sample input values across the height and width dimensions
 *     in atrous convolution. Defaults to `[1, 1]`. If `rate` is a single
 *     number, then `dilationHeight == dilationWidth`. If it is greater than
 *     1, then all values of `strides` must be 1.
 * @param dataFormat: An optional string from: "NHWC", "NCHW". Defaults to
 *     "NHWC". Specify the data format of the input and output data. With the
 *     default format "NHWC", the data is stored in the order of: [batch,
 *     height, width, channels]. Only "NHWC" is currently supported.
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 *
 * @doc {heading: 'Operations', subheading: 'Convolution'}
 */
function depthwiseConv2d_(x, filter, strides, pad, dataFormat = 'NHWC', dilations = [1, 1], dimRoundingMode) {
    const $x = convertToTensor(x, 'x', 'depthwiseConv2d', 'float32');
    const $filter = convertToTensor(filter, 'filter', 'depthwiseConv2d', 'float32');
    let x4D = $x;
    let reshapedTo4D = false;
    if ($x.rank === 3) {
        reshapedTo4D = true;
        x4D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2]]);
    }
    util.assert(x4D.rank === 4, () => `Error in depthwiseConv2d: input must be rank 4, but got ` +
        `rank ${x4D.rank}.`);
    util.assert($filter.rank === 4, () => `Error in depthwiseConv2d: filter must be rank 4, but got rank ` +
        `${$filter.rank}.`);
    const inChannels = dataFormat === 'NHWC' ? x4D.shape[3] : x4D.shape[1];
    util.assert(inChannels === $filter.shape[2], () => `Error in depthwiseConv2d: number of input channels ` +
        `(${inChannels}) must match the inChannels dimension in ` +
        `filter ${$filter.shape[2]}.`);
    conv_util.checkPadOnDimRoundingMode('depthwiseConv2d', pad, dimRoundingMode);
    const inputs = { x: x4D, filter: $filter };
    const attrs = { strides, pad, dataFormat, dilations, dimRoundingMode };
    // tslint:disable-next-line: no-unnecessary-type-assertion
    const res = ENGINE.runKernel(DepthwiseConv2dNative, inputs, attrs);
    if (reshapedTo4D) {
        return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]);
    }
    return res;
}
export const depthwiseConv2d = /* @__PURE__ */ op({ depthwiseConv2d_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZGVwdGh3aXNlX2NvbnYyZC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL2RlcHRod2lzZV9jb252MmQudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBQ0gsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMscUJBQXFCLEVBQTBELE1BQU0saUJBQWlCLENBQUM7QUFJL0csT0FBTyxFQUFDLGVBQWUsRUFBQyxNQUFNLG9CQUFvQixDQUFDO0FBRW5ELE9BQU8sS0FBSyxJQUFJLE1BQU0sU0FBUyxDQUFDO0FBRWhDLE9BQU8sS0FBSyxTQUFTLE1BQU0sYUFBYSxDQUFDO0FBQ3pDLE9BQU8sRUFBQyxFQUFFLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFDL0IsT0FBTyxFQUFDLE9BQU8sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUVsQzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7R0E0Q0c7QUFDSCxTQUFTLGdCQUFnQixDQUNyQixDQUFlLEVBQUUsTUFBMkIsRUFDNUMsT0FBZ0MsRUFDaEMsR0FBb0QsRUFDcEQsYUFBNEIsTUFBTSxFQUNsQyxZQUFxQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFDM0MsZUFBd0M7SUFDMUMsTUFBTSxFQUFFLEdBQUcsZUFBZSxDQUFDLENBQUMsRUFBRSxHQUFHLEVBQUUsaUJBQWlCLEVBQUUsU0FBUyxDQUFDLENBQUM7SUFDakUsTUFBTSxPQUFPLEdBQ1QsZUFBZSxDQUFDLE1BQU0sRUFBRSxRQUFRLEVBQUUsaUJBQWlCLEVBQUUsU0FBUyxDQUFDLENBQUM7SUFFcEUsSUFBSSxHQUFHLEdBQUcsRUFBYyxDQUFDO0lBQ3pCLElBQUksWUFBWSxHQUFHLEtBQUssQ0FBQztJQUN6QixJQUFJLEVBQUUsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUFFO1FBQ2pCLFlBQVksR0FBRyxJQUFJLENBQUM7UUFDcEIsR0FBRyxHQUFHLE9BQU8sQ0FBQyxFQUFFLEVBQUUsQ0FBQyxDQUFDLEVBQUUsRUFBRSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0tBQy9EO0lBQ0QsSUFBSSxDQUFDLE1BQU0sQ0FDUCxHQUFHLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDZCxHQUFHLEVBQUUsQ0FBQywwREFBMEQ7UUFDNUQsUUFBUSxHQUFHLENBQUMsSUFBSSxHQUFHLENBQUMsQ0FBQztJQUM3QixJQUFJLENBQUMsTUFBTSxDQUNQLE9BQU8sQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNsQixHQUFHLEVBQUUsQ0FBQyxnRUFBZ0U7UUFDbEUsR0FBRyxPQUFPLENBQUMsSUFBSSxHQUFHLENBQUMsQ0FBQztJQUM1QixNQUFNLFVBQVUsR0FBRyxVQUFVLEtBQUssTUFBTSxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ3ZFLElBQUksQ0FBQyxNQUFNLENBQ1AsVUFBVSxLQUFLLE9BQU8sQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQy9CLEdBQUcsRUFBRSxDQUFDLHFEQUFxRDtRQUN2RCxJQUFJLFVBQVUsMkNBQTJDO1FBQ3pELFVBQVUsT0FBTyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUM7SUFDdkMsU0FBUyxDQUFDLHlCQUF5QixDQUFDLGlCQUFpQixFQUFFLEdBQUcsRUFBRSxlQUFlLENBQUMsQ0FBQztJQUM3RSxNQUFNLE1BQU0sR0FBZ0MsRUFBQyxDQUFDLEVBQUUsR0FBRyxFQUFFLE1BQU0sRUFBRSxPQUFPLEVBQUMsQ0FBQztJQUN0RSxNQUFNLEtBQUssR0FDUCxFQUFDLE9BQU8sRUFBRSxHQUFHLEVBQUUsVUFBVSxFQUFFLFNBQVMsRUFBRSxlQUFlLEVBQUMsQ0FBQztJQUUzRCwwREFBMEQ7SUFDMUQsTUFBTSxHQUFHLEdBQUcsTUFBTSxDQUFDLFNBQVMsQ0FDWixxQkFBcUIsRUFBRSxNQUFtQyxFQUMxRCxLQUFnQyxDQUFNLENBQUM7SUFFdkQsSUFBSSxZQUFZLEVBQUU7UUFDaEIsT0FBTyxPQUFPLENBQUMsR0FBRyxFQUFFLENBQUMsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxHQUFHLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBTSxDQUFDO0tBQ3RFO0lBQ0QsT0FBTyxHQUFHLENBQUM7QUFDYixDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sZUFBZSxHQUFHLGVBQWUsQ0FBQyxFQUFFLENBQUMsRUFBQyxnQkFBZ0IsRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5pbXBvcnQge0VOR0lORX0gZnJvbSAnLi4vZW5naW5lJztcbmltcG9ydCB7RGVwdGh3aXNlQ29udjJkTmF0aXZlLCBEZXB0aHdpc2VDb252MmROYXRpdmVBdHRycywgRGVwdGh3aXNlQ29udjJkTmF0aXZlSW5wdXRzfSBmcm9tICcuLi9rZXJuZWxfbmFtZXMnO1xuaW1wb3J0IHtOYW1lZEF0dHJNYXB9IGZyb20gJy4uL2tlcm5lbF9yZWdpc3RyeSc7XG5pbXBvcnQge1RlbnNvcjNELCBUZW5zb3I0RH0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7TmFtZWRUZW5zb3JNYXB9IGZyb20gJy4uL3RlbnNvcl90eXBlcyc7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7VGVuc29yTGlrZX0gZnJvbSAnLi4vdHlwZXMnO1xuaW1wb3J0ICogYXMgdXRpbCBmcm9tICcuLi91dGlsJztcblxuaW1wb3J0ICogYXMgY29udl91dGlsIGZyb20gJy4vY29udl91dGlsJztcbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcbmltcG9ydCB7cmVzaGFwZX0gZnJvbSAnLi9yZXNoYXBlJztcblxuLyoqXG4gKiBEZXB0aHdpc2UgMkQgY29udm9sdXRpb24uXG4gKlxuICogR2l2ZW4gYSA0RCBgaW5wdXRgIGFycmF5IGFuZCBhIGBmaWx0ZXJgIGFycmF5IG9mIHNoYXBlXG4gKiBgW2ZpbHRlckhlaWdodCwgZmlsdGVyV2lkdGgsIGluQ2hhbm5lbHMsIGNoYW5uZWxNdWx0aXBsaWVyXWAgY29udGFpbmluZ1xuICogYGluQ2hhbm5lbHNgIGNvbnZvbHV0aW9uYWwgZmlsdGVycyBvZiBkZXB0aCAxLCB0aGlzIG9wIGFwcGxpZXMgYVxuICogZGlmZmVyZW50IGZpbHRlciB0byBlYWNoIGlucHV0IGNoYW5uZWwgKGV4cGFuZGluZyBmcm9tIDEgY2hhbm5lbCB0b1xuICogYGNoYW5uZWxNdWx0aXBsaWVyYCBjaGFubmVscyBmb3IgZWFjaCksIHRoZW4gY29uY2F0ZW5hdGVzIHRoZSByZXN1bHRzXG4gKiB0b2dldGhlci4gVGhlIG91dHB1dCBoYXMgYGluQ2hhbm5lbHMgKiBjaGFubmVsTXVsdGlwbGllcmAgY2hhbm5lbHMuXG4gKlxuICogU2VlXG4gKiBbaHR0cHM6Ly93d3cudGVuc29yZmxvdy5vcmcvYXBpX2RvY3MvcHl0aG9uL3RmL25uL2RlcHRod2lzZV9jb252MmRdKFxuICogICAgIGh0dHBzOi8vd3d3LnRlbnNvcmZsb3cub3JnL2FwaV9kb2NzL3B5dGhvbi90Zi9ubi9kZXB0aHdpc2VfY29udjJkKVxuICogZm9yIG1vcmUgZGV0YWlscy5cbiAqXG4gKiBAcGFyYW0geCBUaGUgaW5wdXQgdGVuc29yLCBvZiByYW5rIDQgb3IgcmFuayAzLCBvZiBzaGFwZVxuICogICAgIGBbYmF0Y2gsIGhlaWdodCwgd2lkdGgsIGluQ2hhbm5lbHNdYC4gSWYgcmFuayAzLCBiYXRjaCBvZiAxIGlzXG4gKiBhc3N1bWVkLlxuICogQHBhcmFtIGZpbHRlciBUaGUgZmlsdGVyIHRlbnNvciwgcmFuayA0LCBvZiBzaGFwZVxuICogICAgIGBbZmlsdGVySGVpZ2h0LCBmaWx0ZXJXaWR0aCwgaW5DaGFubmVscywgY2hhbm5lbE11bHRpcGxpZXJdYC5cbiAqIEBwYXJhbSBzdHJpZGVzIFRoZSBzdHJpZGVzIG9mIHRoZSBjb252b2x1dGlvbjogYFtzdHJpZGVIZWlnaHQsXG4gKiBzdHJpZGVXaWR0aF1gLiBJZiBzdHJpZGVzIGlzIGEgc2luZ2xlIG51bWJlciwgdGhlbiBgc3RyaWRlSGVpZ2h0ID09XG4gKiBzdHJpZGVXaWR0aGAuXG4gKiBAcGFyYW0gcGFkIFRoZSB0eXBlIG9mIHBhZGRpbmcgYWxnb3JpdGhtLlxuICogICAtIGBzYW1lYCBhbmQgc3RyaWRlIDE6IG91dHB1dCB3aWxsIGJlIG9mIHNhbWUgc2l6ZSBhcyBpbnB1dCxcbiAqICAgICAgIHJlZ2FyZGxlc3Mgb2YgZmlsdGVyIHNpemUuXG4gKiAgIC0gYHZhbGlkYDogb3V0cHV0IHdpbGwgYmUgc21hbGxlciB0aGFuIGlucHV0IGlmIGZpbHRlciBpcyBsYXJnZXJcbiAqICAgICAgIHRoYW4gMXgxLlxuICogICAtIEZvciBtb3JlIGluZm8sIHNlZSB0aGlzIGd1aWRlOlxuICogICAgIFtodHRwczovL3d3dy50ZW5zb3JmbG93Lm9yZy9hcGlfZG9jcy9weXRob24vdGYvbm4vY29udm9sdXRpb25dKFxuICogICAgICAgICAgaHR0cHM6Ly93d3cudGVuc29yZmxvdy5vcmcvYXBpX2RvY3MvcHl0aG9uL3RmL25uL2NvbnZvbHV0aW9uKVxuICogQHBhcmFtIGRpbGF0aW9ucyBUaGUgZGlsYXRpb24gcmF0ZXM6IGBbZGlsYXRpb25IZWlnaHQsIGRpbGF0aW9uV2lkdGhdYFxuICogICAgIGluIHdoaWNoIHdlIHNhbXBsZSBpbnB1dCB2YWx1ZXMgYWNyb3NzIHRoZSBoZWlnaHQgYW5kIHdpZHRoIGRpbWVuc2lvbnNcbiAqICAgICBpbiBhdHJvdXMgY29udm9sdXRpb24uIERlZmF1bHRzIHRvIGBbMSwgMV1gLiBJZiBgcmF0ZWAgaXMgYSBzaW5nbGVcbiAqICAgICBudW1iZXIsIHRoZW4gYGRpbGF0aW9uSGVpZ2h0ID09IGRpbGF0aW9uV2lkdGhgLiBJZiBpdCBpcyBncmVhdGVyIHRoYW5cbiAqICAgICAxLCB0aGVuIGFsbCB2YWx1ZXMgb2YgYHN0cmlkZXNgIG11c3QgYmUgMS5cbiAqIEBwYXJhbSBkYXRhRm9ybWF0OiBBbiBvcHRpb25hbCBzdHJpbmcgZnJvbTogXCJOSFdDXCIsIFwiTkNIV1wiLiBEZWZhdWx0cyB0b1xuICogICAgIFwiTkhXQ1wiLiBTcGVjaWZ5IHRoZSBkYXRhIGZvcm1hdCBvZiB0aGUgaW5wdXQgYW5kIG91dHB1dCBkYXRhLiBXaXRoIHRoZVxuICogICAgIGRlZmF1bHQgZm9ybWF0IFwiTkhXQ1wiLCB0aGUgZGF0YSBpcyBzdG9yZWQgaW4gdGhlIG9yZGVyIG9mOiBbYmF0Y2gsXG4gKiAgICAgaGVpZ2h0LCB3aWR0aCwgY2hhbm5lbHNdLiBPbmx5IFwiTkhXQ1wiIGlzIGN1cnJlbnRseSBzdXBwb3J0ZWQuXG4gKiBAcGFyYW0gZGltUm91bmRpbmdNb2RlIEEgc3RyaW5nIGZyb206ICdjZWlsJywgJ3JvdW5kJywgJ2Zsb29yJy4gSWYgbm9uZSBpc1xuICogICAgIHByb3ZpZGVkLCBpdCB3aWxsIGRlZmF1bHQgdG8gdHJ1bmNhdGUuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ09wZXJhdGlvbnMnLCBzdWJoZWFkaW5nOiAnQ29udm9sdXRpb24nfVxuICovXG5mdW5jdGlvbiBkZXB0aHdpc2VDb252MmRfPFQgZXh0ZW5kcyBUZW5zb3IzRHxUZW5zb3I0RD4oXG4gICAgeDogVHxUZW5zb3JMaWtlLCBmaWx0ZXI6IFRlbnNvcjREfFRlbnNvckxpa2UsXG4gICAgc3RyaWRlczogW251bWJlciwgbnVtYmVyXXxudW1iZXIsXG4gICAgcGFkOiAndmFsaWQnfCdzYW1lJ3xudW1iZXJ8Y29udl91dGlsLkV4cGxpY2l0UGFkZGluZyxcbiAgICBkYXRhRm9ybWF0OiAnTkhXQyd8J05DSFcnID0gJ05IV0MnLFxuICAgIGRpbGF0aW9uczogW251bWJlciwgbnVtYmVyXXxudW1iZXIgPSBbMSwgMV0sXG4gICAgZGltUm91bmRpbmdNb2RlPzogJ2Zsb29yJ3wncm91bmQnfCdjZWlsJyk6IFQge1xuICBjb25zdCAkeCA9IGNvbnZlcnRUb1RlbnNvcih4LCAneCcsICdkZXB0aHdpc2VDb252MmQnLCAnZmxvYXQzMicpO1xuICBjb25zdCAkZmlsdGVyID1cbiAgICAgIGNvbnZlcnRUb1RlbnNvcihmaWx0ZXIsICdmaWx0ZXInLCAnZGVwdGh3aXNlQ29udjJkJywgJ2Zsb2F0MzInKTtcblxuICBsZXQgeDREID0gJHggYXMgVGVuc29yNEQ7XG4gIGxldCByZXNoYXBlZFRvNEQgPSBmYWxzZTtcbiAgaWYgKCR4LnJhbmsgPT09IDMpIHtcbiAgICByZXNoYXBlZFRvNEQgPSB0cnVlO1xuICAgIHg0RCA9IHJlc2hhcGUoJHgsIFsxLCAkeC5zaGFwZVswXSwgJHguc2hhcGVbMV0sICR4LnNoYXBlWzJdXSk7XG4gIH1cbiAgdXRpbC5hc3NlcnQoXG4gICAgICB4NEQucmFuayA9PT0gNCxcbiAgICAgICgpID0+IGBFcnJvciBpbiBkZXB0aHdpc2VDb252MmQ6IGlucHV0IG11c3QgYmUgcmFuayA0LCBidXQgZ290IGAgK1xuICAgICAgICAgIGByYW5rICR7eDRELnJhbmt9LmApO1xuICB1dGlsLmFzc2VydChcbiAgICAgICRmaWx0ZXIucmFuayA9PT0gNCxcbiAgICAgICgpID0+IGBFcnJvciBpbiBkZXB0aHdpc2VDb252MmQ6IGZpbHRlciBtdXN0IGJlIHJhbmsgNCwgYnV0IGdvdCByYW5rIGAgK1xuICAgICAgICAgIGAkeyRmaWx0ZXIucmFua30uYCk7XG4gIGNvbnN0IGluQ2hhbm5lbHMgPSBkYXRhRm9ybWF0ID09PSAnTkhXQycgPyB4NEQuc2hhcGVbM10gOiB4NEQuc2hhcGVbMV07XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgaW5DaGFubmVscyA9PT0gJGZpbHRlci5zaGFwZVsyXSxcbiAgICAgICgpID0+IGBFcnJvciBpbiBkZXB0aHdpc2VDb252MmQ6IG51bWJlciBvZiBpbnB1dCBjaGFubmVscyBgICtcbiAgICAgICAgICBgKCR7aW5DaGFubmVsc30pIG11c3QgbWF0Y2ggdGhlIGluQ2hhbm5lbHMgZGltZW5zaW9uIGluIGAgK1xuICAgICAgICAgIGBmaWx0ZXIgJHskZmlsdGVyLnNoYXBlWzJdfS5gKTtcbiAgY29udl91dGlsLmNoZWNrUGFkT25EaW1Sb3VuZGluZ01vZGUoJ2RlcHRod2lzZUNvbnYyZCcsIHBhZCwgZGltUm91bmRpbmdNb2RlKTtcbiAgY29uc3QgaW5wdXRzOiBEZXB0aHdpc2VDb252MmROYXRpdmVJbnB1dHMgPSB7eDogeDRELCBmaWx0ZXI6ICRmaWx0ZXJ9O1xuICBjb25zdCBhdHRyczogRGVwdGh3aXNlQ29udjJkTmF0aXZlQXR0cnMgPVxuICAgICAge3N0cmlkZXMsIHBhZCwgZGF0YUZvcm1hdCwgZGlsYXRpb25zLCBkaW1Sb3VuZGluZ01vZGV9O1xuXG4gIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTogbm8tdW5uZWNlc3NhcnktdHlwZS1hc3NlcnRpb25cbiAgY29uc3QgcmVzID0gRU5HSU5FLnJ1bktlcm5lbChcbiAgICAgICAgICAgICAgICAgIERlcHRod2lzZUNvbnYyZE5hdGl2ZSwgaW5wdXRzIGFzIHVua25vd24gYXMgTmFtZWRUZW5zb3JNYXAsXG4gICAgICAgICAgICAgICAgICBhdHRycyBhcyB1bmtub3duIGFzIE5hbWVkQXR0ck1hcCkgYXMgVDtcblxuICBpZiAocmVzaGFwZWRUbzREKSB7XG4gICAgcmV0dXJuIHJlc2hhcGUocmVzLCBbcmVzLnNoYXBlWzFdLCByZXMuc2hhcGVbMl0sIHJlcy5zaGFwZVszXV0pIGFzIFQ7XG4gIH1cbiAgcmV0dXJuIHJlcztcbn1cblxuZXhwb3J0IGNvbnN0IGRlcHRod2lzZUNvbnYyZCA9IC8qIEBfX1BVUkVfXyAqLyBvcCh7ZGVwdGh3aXNlQ29udjJkX30pO1xuIl19