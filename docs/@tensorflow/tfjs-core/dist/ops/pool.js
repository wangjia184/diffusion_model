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
import { convertToTensor } from '../tensor_util_env';
import * as util from '../util';
import { avgPool } from './avg_pool';
import { batchToSpaceND } from './batch_to_space_nd';
import * as conv_util from './conv_util';
import { maxPool } from './max_pool';
import { op } from './operation';
import { reshape } from './reshape';
import { spaceToBatchND } from './space_to_batch_nd';
/**
 * Performs an N-D pooling operation
 *
 * @param input The input tensor, of rank 4 or rank 3 of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
 * @param windowShape The filter size: `[filterHeight, filterWidth]`. If
 *     `filterSize` is a single number, then `filterHeight == filterWidth`.
 * @param poolingType The type of pooling, either 'max' or 'avg'.
 * @param pad The type of padding algorithm:
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1x1.
 *    - For more info, see this guide:
 *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
 *         https://www.tensorflow.org/api_guides/python/nn#Convolution)
 * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`
 *     in which we sample input values across the height and width dimensions
 *     in dilated pooling. Defaults to `[1, 1]`. If `dilationRate` is a single
 *     number, then `dilationHeight == dilationWidth`. If it is greater than
 *     1, then all values of `strides` must be 1.
 * @param strides The strides of the pooling: `[strideHeight, strideWidth]`. If
 *     `strides` is a single number, then `strideHeight == strideWidth`.
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 *
 * @doc {heading: 'Operations', subheading: 'Convolution'}
 */
function pool_(input, windowShape, poolingType, pad, dilations, strides, dimRoundingMode) {
    if (dilations == null) {
        dilations = [1, 1];
    }
    if (strides == null) {
        strides = 1;
    }
    if (pad === 0) {
        pad = 'valid';
    }
    const $x = convertToTensor(input, 'x', 'maxPool');
    let x4D = $x;
    let reshapedTo4D = false;
    if ($x.rank === 3) {
        reshapedTo4D = true;
        x4D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2]]);
    }
    util.assert(conv_util.eitherStridesOrDilationsAreOne(strides, dilations), () => 'Error in pool: Either strides or dilations must be 1. ' +
        `Got strides ${strides} and dilations '${dilations}'`);
    const convInfo = conv_util.computePool2DInfo(x4D.shape, windowShape, strides, dilations, pad);
    const dilation = [convInfo.dilationHeight, convInfo.dilationWidth];
    // The following implementation does batchToSpace(pool(spaceToBatch(x)))
    // whenever dilation > 1 since the TF kernels do not support dilation > 1.
    // tslint:disable-next-line:max-line-length
    // https://github.com/tensorflow/tensorflow/blob/50f6bb67dc98c9b74630b6047aae7a4f8a40fd02/tensorflow/python/ops/nn_ops.py#L1037
    let basePadding;
    if (pad === 'same') {
        basePadding = withSpaceToBatchBasePaddings([convInfo.filterHeight, convInfo.filterWidth], dilation);
    }
    else {
        basePadding = [[0, 0], [0, 0]];
    }
    const isDilationOne = dilation[0] === 1 && dilation[1] === 1;
    const [adjustedPadding, adjustedCrops] = requiredSpaceToBatchPaddings([convInfo.inHeight, convInfo.inWidth], dilation, basePadding);
    const convertedPad = isDilationOne ? pad : 'valid';
    const convertedX = isDilationOne ? x4D : spaceToBatchND(x4D, dilation, adjustedPadding);
    const forwardOp = poolingType === 'avg' ?
        () => avgPool(convertedX, windowShape, strides, convertedPad, dimRoundingMode) :
        () => maxPool(convertedX, windowShape, strides, convertedPad, dimRoundingMode);
    const y = forwardOp();
    const res = isDilationOne ? y : batchToSpaceND(y, dilation, adjustedCrops);
    if (reshapedTo4D) {
        return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]);
    }
    return res;
}
// Helper function to compute crops and paddings for pool with dilation > 1.
// tslint:disable-next-line:max-line-length
// https://github.com/tensorflow/tensorflow/blob/50f6bb67dc98c9b74630b6047aae7a4f8a40fd02/tensorflow/python/ops/array_ops.py#L2184
function requiredSpaceToBatchPaddings(inputShape, blockShape, basePadding) {
    const padStart = basePadding.map(b => b[0]);
    const origPadEnd = basePadding.map(b => b[1]);
    const fullInputShape = inputShape.concat(padStart, origPadEnd);
    const padEndExtra = blockShape.map((b, i) => (b - fullInputShape[i] % b) % b);
    const padEnd = origPadEnd.map((s, i) => s + padEndExtra[i]);
    const paddings = blockShape.map((_, i) => [padStart[i], padEnd[i]]);
    const crops = blockShape.map((_, i) => [0, padEndExtra[i]]);
    return [paddings, crops];
}
// Helper function to compute base paddings for pool with dilation > 1.
// tslint:disable-next-line:max-line-length
// https://github.com/tensorflow/tensorflow/blob/50f6bb67dc98c9b74630b6047aae7a4f8a40fd02/tensorflow/python/ops/nn_ops.py#L524
function withSpaceToBatchBasePaddings(filterShape, dilation) {
    // Spatial dimensions of the filters and the upsampled filters in which we
    // introduce (rate - 1) zeros between consecutive filter values.
    const dilatedFilterShape = filterShape.map((s, i) => {
        return s + (s - 1) * (dilation[i] - 1);
    });
    const padExtraShape = dilatedFilterShape.map(s => s - 1);
    // When padding is odd, we pad more at end, following the same
    // convention as conv2d.
    const padExtraStart = padExtraShape.map(s => Math.floor(s / 2));
    const padExtraEnd = padExtraShape.map((s, i) => s - padExtraStart[i]);
    return padExtraShape.map((_, i) => {
        return [padExtraStart[i], padExtraEnd[i]];
    });
}
export const pool = /* @__PURE__ */ op({ pool_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicG9vbC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL3Bvb2wudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBR0gsT0FBTyxFQUFDLGVBQWUsRUFBQyxNQUFNLG9CQUFvQixDQUFDO0FBRW5ELE9BQU8sS0FBSyxJQUFJLE1BQU0sU0FBUyxDQUFDO0FBRWhDLE9BQU8sRUFBQyxPQUFPLEVBQUMsTUFBTSxZQUFZLENBQUM7QUFDbkMsT0FBTyxFQUFDLGNBQWMsRUFBQyxNQUFNLHFCQUFxQixDQUFDO0FBQ25ELE9BQU8sS0FBSyxTQUFTLE1BQU0sYUFBYSxDQUFDO0FBQ3pDLE9BQU8sRUFBQyxPQUFPLEVBQUMsTUFBTSxZQUFZLENBQUM7QUFDbkMsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUMvQixPQUFPLEVBQUMsT0FBTyxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQ2xDLE9BQU8sRUFBQyxjQUFjLEVBQUMsTUFBTSxxQkFBcUIsQ0FBQztBQUVuRDs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBMkJHO0FBQ0gsU0FBUyxLQUFLLENBQ1YsS0FBbUIsRUFBRSxXQUFvQyxFQUN6RCxXQUF3QixFQUN4QixHQUFvRCxFQUNwRCxTQUFtQyxFQUFFLE9BQWlDLEVBQ3RFLGVBQXdDO0lBQzFDLElBQUksU0FBUyxJQUFJLElBQUksRUFBRTtRQUNyQixTQUFTLEdBQUcsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7S0FDcEI7SUFDRCxJQUFJLE9BQU8sSUFBSSxJQUFJLEVBQUU7UUFDbkIsT0FBTyxHQUFHLENBQUMsQ0FBQztLQUNiO0lBQ0QsSUFBSSxHQUFHLEtBQUssQ0FBQyxFQUFFO1FBQ2IsR0FBRyxHQUFHLE9BQU8sQ0FBQztLQUNmO0lBRUQsTUFBTSxFQUFFLEdBQUcsZUFBZSxDQUFDLEtBQUssRUFBRSxHQUFHLEVB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NUMsQ0FBQyxDQUFDLENBQUM7QUFDTCxDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sSUFBSSxHQUFHLGVBQWUsQ0FBQyxFQUFFLENBQUMsRUFBQyxLQUFLLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTggR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge1RlbnNvcjNELCBUZW5zb3I0RH0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4uL3V0aWwnO1xuXG5pbXBvcnQge2F2Z1Bvb2x9IGZyb20gJy4vYXZnX3Bvb2wnO1xuaW1wb3J0IHtiYXRjaFRvU3BhY2VORH0gZnJvbSAnLi9iYXRjaF90b19zcGFjZV9uZCc7XG5pbXBvcnQgKiBhcyBjb252X3V0aWwgZnJvbSAnLi9jb252X3V0aWwnO1xuaW1wb3J0IHttYXhQb29sfSBmcm9tICcuL21heF9wb29sJztcbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcbmltcG9ydCB7cmVzaGFwZX0gZnJvbSAnLi9yZXNoYXBlJztcbmltcG9ydCB7c3BhY2VUb0JhdGNoTkR9IGZyb20gJy4vc3BhY2VfdG9fYmF0Y2hfbmQnO1xuXG4vKipcbiAqIFBlcmZvcm1zIGFuIE4tRCBwb29saW5nIG9wZXJhdGlvblxuICpcbiAqIEBwYXJhbSBpbnB1dCBUaGUgaW5wdXQgdGVuc29yLCBvZiByYW5rIDQgb3IgcmFuayAzIG9mIHNoYXBlXG4gKiAgICAgYFtiYXRjaCwgaGVpZ2h0LCB3aWR0aCwgaW5DaGFubmVsc11gLiBJZiByYW5rIDMsIGJhdGNoIG9mIDEgaXMgYXNzdW1lZC5cbiAqIEBwYXJhbSB3aW5kb3dTaGFwZSBUaGUgZmlsdGVyIHNpemU6IGBbZmlsdGVySGVpZ2h0LCBmaWx0ZXJXaWR0aF1gLiBJZlxuICogICAgIGBmaWx0ZXJTaXplYCBpcyBhIHNpbmdsZSBudW1iZXIsIHRoZW4gYGZpbHRlckhlaWdodCA9PSBmaWx0ZXJXaWR0aGAuXG4gKiBAcGFyYW0gcG9vbGluZ1R5cGUgVGhlIHR5cGUgb2YgcG9vbGluZywgZWl0aGVyICdtYXgnIG9yICdhdmcnLlxuICogQHBhcmFtIHBhZCBUaGUgdHlwZSBvZiBwYWRkaW5nIGFsZ29yaXRobTpcbiAqICAgIC0gYHNhbWVgIGFuZCBzdHJpZGUgMTogb3V0cHV0IHdpbGwgYmUgb2Ygc2FtZSBzaXplIGFzIGlucHV0LFxuICogICAgICAgcmVnYXJkbGVzcyBvZiBmaWx0ZXIgc2l6ZS5cbiAqICAgIC0gYHZhbGlkYDogb3V0cHV0IHdpbGwgYmUgc21hbGxlciB0aGFuIGlucHV0IGlmIGZpbHRlciBpcyBsYXJnZXJcbiAqICAgICAgIHRoYW4gMXgxLlxuICogICAgLSBGb3IgbW9yZSBpbmZvLCBzZWUgdGhpcyBndWlkZTpcbiAqICAgICBbaHR0cHM6Ly93d3cudGVuc29yZmxvdy5vcmcvYXBpX2d1aWRlcy9weXRob24vbm4jQ29udm9sdXRpb25dKFxuICogICAgICAgICBodHRwczovL3d3dy50ZW5zb3JmbG93Lm9yZy9hcGlfZ3VpZGVzL3B5dGhvbi9ubiNDb252b2x1dGlvbilcbiAqIEBwYXJhbSBkaWxhdGlvbnMgVGhlIGRpbGF0aW9uIHJhdGVzOiBgW2RpbGF0aW9uSGVpZ2h0LCBkaWxhdGlvbldpZHRoXWBcbiAqICAgICBpbiB3aGljaCB3ZSBzYW1wbGUgaW5wdXQgdmFsdWVzIGFjcm9zcyB0aGUgaGVpZ2h0IGFuZCB3aWR0aCBkaW1lbnNpb25zXG4gKiAgICAgaW4gZGlsYXRlZCBwb29saW5nLiBEZWZhdWx0cyB0byBgWzEsIDFdYC4gSWYgYGRpbGF0aW9uUmF0ZWAgaXMgYSBzaW5nbGVcbiAqICAgICBudW1iZXIsIHRoZW4gYGRpbGF0aW9uSGVpZ2h0ID09IGRpbGF0aW9uV2lkdGhgLiBJZiBpdCBpcyBncmVhdGVyIHRoYW5cbiAqICAgICAxLCB0aGVuIGFsbCB2YWx1ZXMgb2YgYHN0cmlkZXNgIG11c3QgYmUgMS5cbiAqIEBwYXJhbSBzdHJpZGVzIFRoZSBzdHJpZGVzIG9mIHRoZSBwb29saW5nOiBgW3N0cmlkZUhlaWdodCwgc3RyaWRlV2lkdGhdYC4gSWZcbiAqICAgICBgc3RyaWRlc2AgaXMgYSBzaW5nbGUgbnVtYmVyLCB0aGVuIGBzdHJpZGVIZWlnaHQgPT0gc3RyaWRlV2lkdGhgLlxuICogQHBhcmFtIGRpbVJvdW5kaW5nTW9kZSBBIHN0cmluZyBmcm9tOiAnY2VpbCcsICdyb3VuZCcsICdmbG9vcicuIElmIG5vbmUgaXNcbiAqICAgICBwcm92aWRlZCwgaXQgd2lsbCBkZWZhdWx0IHRvIHRydW5jYXRlLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdPcGVyYXRpb25zJywgc3ViaGVhZGluZzogJ0NvbnZvbHV0aW9uJ31cbiAqL1xuZnVuY3Rpb24gcG9vbF88VCBleHRlbmRzIFRlbnNvcjNEfFRlbnNvcjREPihcbiAgICBpbnB1dDogVHxUZW5zb3JMaWtlLCB3aW5kb3dTaGFwZTogW251bWJlciwgbnVtYmVyXXxudW1iZXIsXG4gICAgcG9vbGluZ1R5cGU6ICdhdmcnfCdtYXgnLFxuICAgIHBhZDogJ3ZhbGlkJ3wnc2FtZSd8bnVtYmVyfGNvbnZfdXRpbC5FeHBsaWNpdFBhZGRpbmcsXG4gICAgZGlsYXRpb25zPzogW251bWJlciwgbnVtYmVyXXxudW1iZXIsIHN0cmlkZXM/OiBbbnVtYmVyLCBudW1iZXJdfG51bWJlcixcbiAgICBkaW1Sb3VuZGluZ01vZGU/OiAnZmxvb3InfCdyb3VuZCd8J2NlaWwnKSB7XG4gIGlmIChkaWxhdGlvbnMgPT0gbnVsbCkge1xuICAgIGRpbGF0aW9ucyA9IFsxLCAxXTtcbiAgfVxuICBpZiAoc3RyaWRlcyA9PSBudWxsKSB7XG4gICAgc3RyaWRlcyA9IDE7XG4gIH1cbiAgaWYgKHBhZCA9PT0gMCkge1xuICAgIHBhZCA9ICd2YWxpZCc7XG4gIH1cblxuICBjb25zdCAkeCA9IGNvbnZlcnRUb1RlbnNvcihpbnB1dCwgJ3gnLCAnbWF4UG9vbCcpO1xuICBsZXQgeDREID0gJHggYXMgVGVuc29yNEQ7XG4gIGxldCByZXNoYXBlZFRvNEQgPSBmYWxzZTtcblxuICBpZiAoJHgucmFuayA9PT0gMykge1xuICAgIHJlc2hhcGVkVG80RCA9IHRydWU7XG4gICAgeDREID0gcmVzaGFwZSgkeCwgWzEsICR4LnNoYXBlWzBdLCAkeC5zaGFwZVsxXSwgJHguc2hhcGVbMl1dKTtcbiAgfVxuXG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgY29udl91dGlsLmVpdGhlclN0cmlkZXNPckRpbGF0aW9uc0FyZU9uZShzdHJpZGVzLCBkaWxhdGlvbnMpLFxuICAgICAgKCkgPT4gJ0Vycm9yIGluIHBvb2w6IEVpdGhlciBzdHJpZGVzIG9yIGRpbGF0aW9ucyBtdXN0IGJlIDEuICcgK1xuICAgICAgICAgIGBHb3Qgc3RyaWRlcyAke3N0cmlkZXN9IGFuZCBkaWxhdGlvbnMgJyR7ZGlsYXRpb25zfSdgKTtcblxuICBjb25zdCBjb252SW5mbyA9IGNvbnZfdXRpbC5jb21wdXRlUG9vbDJESW5mbyhcbiAgICAgIHg0RC5zaGFwZSwgd2luZG93U2hhcGUsIHN0cmlkZXMsIGRpbGF0aW9ucywgcGFkKTtcbiAgY29uc3QgZGlsYXRpb246IFtudW1iZXIsIG51bWJlcl0gPVxuICAgICAgW2NvbnZJbmZvLmRpbGF0aW9uSGVpZ2h0LCBjb252SW5mby5kaWxhdGlvbldpZHRoXTtcblxuICAvLyBUaGUgZm9sbG93aW5nIGltcGxlbWVudGF0aW9uIGRvZXMgYmF0Y2hUb1NwYWNlKHBvb2woc3BhY2VUb0JhdGNoKHgpKSlcbiAgLy8gd2hlbmV2ZXIgZGlsYXRpb24gPiAxIHNpbmNlIHRoZSBURiBrZXJuZWxzIGRvIG5vdCBzdXBwb3J0IGRpbGF0aW9uID4gMS5cbiAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm1heC1saW5lLWxlbmd0aFxuICAvLyBodHRwczovL2dpdGh1Yi5jb20vdGVuc29yZmxvdy90ZW5zb3JmbG93L2Jsb2IvNTBmNmJiNjdkYzk4YzliNzQ2MzBiNjA0N2FhZTdhNGY4YTQwZmQwMi90ZW5zb3JmbG93L3B5dGhvbi9vcHMvbm5fb3BzLnB5I0wxMDM3XG5cbiAgbGV0IGJhc2VQYWRkaW5nOiBudW1iZXJbXVtdO1xuICBpZiAocGFkID09PSAnc2FtZScpIHtcbiAgICBiYXNlUGFkZGluZyA9IHdpdGhTcGFjZVRvQmF0Y2hCYXNlUGFkZGluZ3MoXG4gICAgICAgIFtjb252SW5mby5maWx0ZXJIZWlnaHQsIGNvbnZJbmZvLmZpbHRlcldpZHRoXSwgZGlsYXRpb24pO1xuICB9IGVsc2Uge1xuICAgIGJhc2VQYWRkaW5nID0gW1swLCAwXSwgWzAsIDBdXTtcbiAgfVxuXG4gIGNvbnN0IGlzRGlsYXRpb25PbmUgPSBkaWxhdGlvblswXSA9PT0gMSAmJiBkaWxhdGlvblsxXSA9PT0gMTtcbiAgY29uc3QgW2FkanVzdGVkUGFkZGluZywgYWRqdXN0ZWRDcm9wc10gPSByZXF1aXJlZFNwYWNlVG9CYXRjaFBhZGRpbmdzKFxuICAgICAgW2NvbnZJbmZvLmluSGVpZ2h0LCBjb252SW5mby5pbldpZHRoXSwgZGlsYXRpb24sIGJhc2VQYWRkaW5nKTtcbiAgY29uc3QgY29udmVydGVkUGFkID0gaXNEaWxhdGlvbk9uZSA/IHBhZCA6ICd2YWxpZCc7XG4gIGNvbnN0IGNvbnZlcnRlZFggPVxuICAgICAgaXNEaWxhdGlvbk9uZSA/IHg0RCA6IHNwYWNlVG9CYXRjaE5EKHg0RCwgZGlsYXRpb24sIGFkanVzdGVkUGFkZGluZyk7XG5cbiAgY29uc3QgZm9yd2FyZE9wID0gcG9vbGluZ1R5cGUgPT09ICdhdmcnID9cbiAgICAgICgpID0+IGF2Z1Bvb2woY29udmVydGVkWCwgd2luZG93U2hhcGUsIHN0cmlkZXMsIGNvbnZlcnRlZFBhZCxcbiAgICAgICAgICAgICAgICAgICAgZGltUm91bmRpbmdNb2RlKSA6XG4gICAgICAoKSA9PiBtYXhQb29sKGNvbnZlcnRlZFgsIHdpbmRvd1NoYXBlLCBzdHJpZGVzLCBjb252ZXJ0ZWRQYWQsXG4gICAgICAgICAgICAgICAgICAgIGRpbVJvdW5kaW5nTW9kZSk7XG4gIGNvbnN0IHkgPSBmb3J3YXJkT3AoKTtcblxuICBjb25zdCByZXMgPSBpc0RpbGF0aW9uT25lID8geSA6IGJhdGNoVG9TcGFjZU5EKHksIGRpbGF0aW9uLCBhZGp1c3RlZENyb3BzKTtcblxuICBpZiAocmVzaGFwZWRUbzREKSB7XG4gICAgcmV0dXJuIHJlc2hhcGUocmVzLCBbcmVzLnNoYXBlWzFdLCByZXMuc2hhcGVbMl0sIHJlcy5zaGFwZVszXV0pIGFzIFQ7XG4gIH1cblxuICByZXR1cm4gcmVzIGFzIFQ7XG59XG5cbi8vIEhlbHBlciBmdW5jdGlvbiB0byBjb21wdXRlIGNyb3BzIGFuZCBwYWRkaW5ncyBmb3IgcG9vbCB3aXRoIGRpbGF0aW9uID4gMS5cbi8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTptYXgtbGluZS1sZW5ndGhcbi8vIGh0dHBzOi8vZ2l0aHViLmNvbS90ZW5zb3JmbG93L3RlbnNvcmZsb3cvYmxvYi81MGY2YmI2N2RjOThjOWI3NDYzMGI2MDQ3YWFlN2E0ZjhhNDBmZDAyL3RlbnNvcmZsb3cvcHl0aG9uL29wcy9hcnJheV9vcHMucHkjTDIxODRcbmZ1bmN0aW9uIHJlcXVpcmVkU3BhY2VUb0JhdGNoUGFkZGluZ3MoXG4gICAgaW5wdXRTaGFwZTogW251bWJlciwgbnVtYmVyXSwgYmxvY2tTaGFwZTogW251bWJlciwgbnVtYmVyXSxcbiAgICBiYXNlUGFkZGluZzogbnVtYmVyW11bXSkge1xuICBjb25zdCBwYWRTdGFydCA9IGJhc2VQYWRkaW5nLm1hcChiID0+IGJbMF0pO1xuICBjb25zdCBvcmlnUGFkRW5kID0gYmFzZVBhZGRpbmcubWFwKGIgPT4gYlsxXSk7XG4gIGNvbnN0IGZ1bGxJbnB1dFNoYXBlID0gaW5wdXRTaGFwZS5jb25jYXQocGFkU3RhcnQsIG9yaWdQYWRFbmQpO1xuICBjb25zdCBwYWRFbmRFeHRyYSA9IGJsb2NrU2hhcGUubWFwKChiLCBpKSA9PiAoYiAtIGZ1bGxJbnB1dFNoYXBlW2ldICUgYikgJSBiKTtcbiAgY29uc3QgcGFkRW5kID0gb3JpZ1BhZEVuZC5tYXAoKHMsIGkpID0+IHMgKyBwYWRFbmRFeHRyYVtpXSk7XG4gIGNvbnN0IHBhZGRpbmdzID0gYmxvY2tTaGFwZS5tYXAoKF8sIGkpID0+IFtwYWRTdGFydFtpXSwgcGFkRW5kW2ldXSk7XG4gIGNvbnN0IGNyb3BzID0gYmxvY2tTaGFwZS5tYXAoKF8sIGkpID0+IFswLCBwYWRFbmRFeHRyYVtpXV0pO1xuICByZXR1cm4gW3BhZGRpbmdzLCBjcm9wc107XG59XG5cbi8vIEhlbHBlciBmdW5jdGlvbiB0byBjb21wdXRlIGJhc2UgcGFkZGluZ3MgZm9yIHBvb2wgd2l0aCBkaWxhdGlvbiA+IDEuXG4vLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bWF4LWxpbmUtbGVuZ3RoXG4vLyBodHRwczovL2dpdGh1Yi5jb20vdGVuc29yZmxvdy90ZW5zb3JmbG93L2Jsb2IvNTBmNmJiNjdkYzk4YzliNzQ2MzBiNjA0N2FhZTdhNGY4YTQwZmQwMi90ZW5zb3JmbG93L3B5dGhvbi9vcHMvbm5fb3BzLnB5I0w1MjRcbmZ1bmN0aW9uIHdpdGhTcGFjZVRvQmF0Y2hCYXNlUGFkZGluZ3MoXG4gICAgZmlsdGVyU2hhcGU6IFtudW1iZXIsIG51bWJlcl0sIGRpbGF0aW9uOiBbbnVtYmVyLCBudW1iZXJdKSB7XG4gIC8vIFNwYXRpYWwgZGltZW5zaW9ucyBvZiB0aGUgZmlsdGVycyBhbmQgdGhlIHVwc2FtcGxlZCBmaWx0ZXJzIGluIHdoaWNoIHdlXG4gIC8vIGludHJvZHVjZSAocmF0ZSAtIDEpIHplcm9zIGJldHdlZW4gY29uc2VjdXRpdmUgZmlsdGVyIHZhbHVlcy5cbiAgY29uc3QgZGlsYXRlZEZpbHRlclNoYXBlID0gZmlsdGVyU2hhcGUubWFwKChzLCBpKSA9PiB7XG4gICAgcmV0dXJuIHMgKyAocyAtIDEpICogKGRpbGF0aW9uW2ldIC0gMSk7XG4gIH0pO1xuICBjb25zdCBwYWRFeHRyYVNoYXBlID0gZGlsYXRlZEZpbHRlclNoYXBlLm1hcChzID0+IHMgLSAxKTtcblxuICAvLyBXaGVuIHBhZGRpbmcgaXMgb2RkLCB3ZSBwYWQgbW9yZSBhdCBlbmQsIGZvbGxvd2luZyB0aGUgc2FtZVxuICAvLyBjb252ZW50aW9uIGFzIGNvbnYyZC5cbiAgY29uc3QgcGFkRXh0cmFTdGFydCA9IHBhZEV4dHJhU2hhcGUubWFwKHMgPT4gTWF0aC5mbG9vcihzIC8gMikpO1xuICBjb25zdCBwYWRFeHRyYUVuZCA9IHBhZEV4dHJhU2hhcGUubWFwKChzLCBpKSA9PiBzIC0gcGFkRXh0cmFTdGFydFtpXSk7XG4gIHJldHVybiBwYWRFeHRyYVNoYXBlLm1hcCgoXywgaSkgPT4ge1xuICAgIHJldHVybiBbcGFkRXh0cmFTdGFydFtpXSwgcGFkRXh0cmFFbmRbaV1dO1xuICB9KTtcbn1cblxuZXhwb3J0IGNvbnN0IHBvb2wgPSAvKiBAX19QVVJFX18gKi8gb3Aoe3Bvb2xffSk7XG4iXX0=