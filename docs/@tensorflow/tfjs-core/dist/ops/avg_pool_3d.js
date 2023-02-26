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
import { AvgPool3D } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import * as util from '../util';
import { cast } from './cast';
import { checkPadOnDimRoundingMode } from './conv_util';
import { op } from './operation';
import { reshape } from './reshape';
/**
 * Computes the 3D average pooling.
 *
 * ```js
 * const x = tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]);
 * const result = tf.avgPool3d(x, 2, 1, 'valid');
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
 *
 * @doc {heading: 'Operations', subheading: 'Convolution'}
 */
function avgPool3d_(x, filterSize, strides, pad, dimRoundingMode, dataFormat = 'NDHWC') {
    const $x = convertToTensor(x, 'x', 'avgPool3d', 'float32');
    let x5D = $x;
    let reshapedTo5D = false;
    if ($x.rank === 4) {
        reshapedTo5D = true;
        x5D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2], $x.shape[3]]);
    }
    util.assert(x5D.rank === 5, () => `Error in avgPool3d: x must be rank 5 but got rank ${x5D.rank}.`);
    util.assert(dataFormat === 'NDHWC', () => `Error in avgPool3d: Only NDHWC is currently supported, ` +
        `but got dataFormat of ${dataFormat}`);
    util.assert((typeof strides === 'number' && strides > 0) ||
        (Array.isArray(strides) && strides[0] > 0 && strides[1] > 0 &&
            strides[2] > 0), () => `Error in avgPool3d: Stride must be > 0, but got '${strides}'`);
    checkPadOnDimRoundingMode('avgPool3d', pad, dimRoundingMode);
    const inputs = { x: x5D };
    const attrs = { filterSize, strides, pad, dimRoundingMode, dataFormat };
    // tslint:disable-next-line: no-unnecessary-type-assertion
    let res = ENGINE.runKernel(AvgPool3D, inputs, attrs);
    res = cast(res, x5D.dtype);
    if (reshapedTo5D) {
        return reshape(res, [res.shape[1], res.shape[2], res.shape[3], res.shape[4]]);
    }
    return res;
}
export const avgPool3d = /* @__PURE__ */ op({ avgPool3d_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYXZnX3Bvb2xfM2QuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9hdmdfcG9vbF8zZC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQ2pDLE9BQU8sRUFBQyxTQUFTLEVBQWtDLE1BQU0saUJBQWlCLENBQUM7QUFJM0UsT0FBTyxFQUFDLGVBQWUsRUFBQyxNQUFNLG9CQUFvQixDQUFDO0FBRW5ELE9BQU8sS0FBSyxJQUFJLE1BQU0sU0FBUyxDQUFDO0FBRWhDLE9BQU8sRUFBQyxJQUFJLEVBQUMsTUFBTSxRQUFRLENBQUM7QUFDNUIsT0FBTyxFQUFDLHlCQUF5QixFQUFDLE1BQU0sYUFBYSxDQUFDO0FBQ3RELE9BQU8sRUFBQyxFQUFFLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFDL0IsT0FBTyxFQUFDLE9BQU8sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUVsQzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7R0FtQ0c7QUFDSCxTQUFTLFVBQVUsQ0FDZixDQUFlLEVBQUUsVUFBMkMsRUFDNUQsT0FBd0MsRUFBRSxHQUEwQixFQUNwRSxlQUF3QyxFQUN4QyxhQUE4QixPQUFPO0lBQ3ZDLE1BQU0sRUFBRSxHQUFHLGVBQWUsQ0FBQyxDQUFDLEVBQUUsR0FBRyxFQUFFLFdBQVcsRUFBRSxTQUFTLENBQUMsQ0FBQztJQUUzRCxJQUFJLEdBQUcsR0FBRyxFQUFjLENBQUM7SUFDekIsSUFBSSxZQUFZLEdBQUcsS0FBSyxDQUFDO0lBQ3pCLElBQUksRUFBRSxDQUFDLElBQUksS0FBSyxDQUFDLEVBQUU7UUFDakIsWUFBWSxHQUFHLElBQUksQ0FBQztRQUNwQixHQUFHLEdBQUcsT0FBTyxDQUFDLEVBQUUsRUFBRSxDQUFDLENBQUMsRUFBRSxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsRUFBRSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztLQUM1RTtJQUVELElBQUksQ0FBQyxNQUFNLENBQ1AsR0FBRyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ2QsR0FBRyxFQUFFLENBQUMscURBQXFELEdBQUcsQ0FBQyxJQUFJLEdBQUcsQ0FBQyxDQUFDO0lBQzVFLElBQUksQ0FBQyxNQUFNLENBQ1AsVUFBVSxLQUFLLE9BQU8sRUFDdEIsR0FBRyxFQUFFLENBQUMseURBQXlEO1FBQzNELHlCQUF5QixVQUFVLEVBQUUsQ0FBQyxDQUFDO0lBQy9DLElBQUksQ0FBQyxNQUFNLENBQ1AsQ0FBQyxPQUFPLE9BQU8sS0FBSyxRQUFRLElBQUksT0FBTyxHQUFHLENBQUMsQ0FBQztRQUN4QyxDQUFDLEtBQUssQ0FBQyxPQUFPLENBQUMsT0FBTyxDQUFDLElBQUksT0FBTyxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsSUFBSSxPQUFPLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQztZQUMxRCxPQUFPLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLEVBQ3BCLEdBQUcsRUFBRSxDQUFDLG9EQUFvRCxPQUFPLEdBQUcsQ0FBQyxDQUFDO0lBQzFFLHlCQUF5QixDQUFDLFdBQVcsRUFBRSxHQUFHLEVBQUUsZUFBZSxDQUFDLENBQUM7SUFDN0QsTUFBTSxNQUFNLEdBQW9CLEVBQUMsQ0FBQyxFQUFFLEdBQUcsRUFBQyxDQUFDO0lBQ3pDLE1BQU0sS0FBSyxHQUNVLEVBQUMsVUFBVSxFQUFFLE9BQU8sRUFBRSxHQUFHLEVBQUUsZUFBZSxFQUFFLFVBQVUsRUFBQyxDQUFDO0lBRTdFLDBEQUEwRDtJQUMxRCxJQUFJLEdBQUcsR0FBRyxNQUFNLENBQUMsU0FBUyxDQUNaLFNBQVMsRUFBRSxNQUFtQyxFQUM5QyxLQUFnQyxDQUFNLENBQUM7SUFFckQsR0FBRyxHQUFHLElBQUksQ0FBQyxHQUFHLEVBQUUsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDO0lBRTNCLElBQUksWUFBWSxFQUFFO1FBQ2hCLE9BQU8sT0FBTyxDQUNILEdBQUcsRUFBRSxDQUFDLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxHQUFHLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FDbkUsQ0FBQztLQUNQO0lBRUQsT0FBTyxHQUFHLENBQUM7QUFDYixDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sU0FBUyxHQUFHLGVBQWUsQ0FBQyxFQUFFLENBQUMsRUFBQyxVQUFVLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0VOR0lORX0gZnJvbSAnLi4vZW5naW5lJztcbmltcG9ydCB7QXZnUG9vbDNELCBBdmdQb29sM0RBdHRycywgQXZnUG9vbDNESW5wdXRzfSBmcm9tICcuLi9rZXJuZWxfbmFtZXMnO1xuaW1wb3J0IHtOYW1lZEF0dHJNYXB9IGZyb20gJy4uL2tlcm5lbF9yZWdpc3RyeSc7XG5pbXBvcnQge1RlbnNvcjRELCBUZW5zb3I1RH0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7TmFtZWRUZW5zb3JNYXB9IGZyb20gJy4uL3RlbnNvcl90eXBlcyc7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7VGVuc29yTGlrZX0gZnJvbSAnLi4vdHlwZXMnO1xuaW1wb3J0ICogYXMgdXRpbCBmcm9tICcuLi91dGlsJztcblxuaW1wb3J0IHtjYXN0fSBmcm9tICcuL2Nhc3QnO1xuaW1wb3J0IHtjaGVja1BhZE9uRGltUm91bmRpbmdNb2RlfSBmcm9tICcuL2NvbnZfdXRpbCc7XG5pbXBvcnQge29wfSBmcm9tICcuL29wZXJhdGlvbic7XG5pbXBvcnQge3Jlc2hhcGV9IGZyb20gJy4vcmVzaGFwZSc7XG5cbi8qKlxuICogQ29tcHV0ZXMgdGhlIDNEIGF2ZXJhZ2UgcG9vbGluZy5cbiAqXG4gKiBgYGBqc1xuICogY29uc3QgeCA9IHRmLnRlbnNvcjVkKFsxLCAyLCAzLCA0LCA1LCA2LCA3LCA4XSwgWzEsIDIsIDIsIDIsIDFdKTtcbiAqIGNvbnN0IHJlc3VsdCA9IHRmLmF2Z1Bvb2wzZCh4LCAyLCAxLCAndmFsaWQnKTtcbiAqIHJlc3VsdC5wcmludCgpO1xuICogYGBgXG4gKlxuICogQHBhcmFtIHggVGhlIGlucHV0IHRlbnNvciwgb2YgcmFuayA1IG9yIHJhbmsgNCBvZiBzaGFwZVxuICogICAgIGBbYmF0Y2gsIGRlcHRoLCBoZWlnaHQsIHdpZHRoLCBpbkNoYW5uZWxzXWAuXG4gKiBAcGFyYW0gZmlsdGVyU2l6ZSBUaGUgZmlsdGVyIHNpemU6XG4gKiAgICAgYFtmaWx0ZXJEZXB0aCwgZmlsdGVySGVpZ2h0LCBmaWx0ZXJXaWR0aF1gLlxuICogICAgIElmIGBmaWx0ZXJTaXplYCBpcyBhIHNpbmdsZSBudW1iZXIsXG4gKiAgICAgdGhlbiBgZmlsdGVyRGVwdGggPT0gZmlsdGVySGVpZ2h0ID09IGZpbHRlcldpZHRoYC5cbiAqIEBwYXJhbSBzdHJpZGVzIFRoZSBzdHJpZGVzIG9mIHRoZSBwb29saW5nOlxuICogICAgIGBbc3RyaWRlRGVwdGgsIHN0cmlkZUhlaWdodCwgc3RyaWRlV2lkdGhdYC5cbiAqICAgICBJZiBgc3RyaWRlc2AgaXMgYSBzaW5nbGUgbnVtYmVyLFxuICogICAgIHRoZW4gYHN0cmlkZURlcHRoID09IHN0cmlkZUhlaWdodCA9PSBzdHJpZGVXaWR0aGAuXG4gKiBAcGFyYW0gcGFkIFRoZSB0eXBlIG9mIHBhZGRpbmcgYWxnb3JpdGhtLlxuICogICAgLSBgc2FtZWAgYW5kIHN0cmlkZSAxOiBvdXRwdXQgd2lsbCBiZSBvZiBzYW1lIHNpemUgYXMgaW5wdXQsXG4gKiAgICAgICByZWdhcmRsZXNzIG9mIGZpbHRlciBzaXplLlxuICogICAgLSBgdmFsaWRgOiBvdXRwdXQgd2lsbCBiZSBzbWFsbGVyIHRoYW4gaW5wdXQgaWYgZmlsdGVyIGlzIGxhcmdlclxuICogICAgICAgdGhhbiAxKjF4MS5cbiAqICAgIC0gRm9yIG1vcmUgaW5mbywgc2VlIHRoaXMgZ3VpZGU6XG4gKiAgICAgW2h0dHBzOi8vd3d3LnRlbnNvcmZsb3cub3JnL2FwaV9kb2NzL3B5dGhvbi90Zi9ubi9jb252b2x1dGlvbl0oXG4gKiAgICAgICAgICBodHRwczovL3d3dy50ZW5zb3JmbG93Lm9yZy9hcGlfZG9jcy9weXRob24vdGYvbm4vY29udm9sdXRpb24pXG4gKiBAcGFyYW0gZGltUm91bmRpbmdNb2RlIEEgc3RyaW5nIGZyb206ICdjZWlsJywgJ3JvdW5kJywgJ2Zsb29yJy4gSWYgbm9uZSBpc1xuICogICAgIHByb3ZpZGVkLCBpdCB3aWxsIGRlZmF1bHQgdG8gdHJ1bmNhdGUuXG4gKiBAcGFyYW0gZGF0YUZvcm1hdCBBbiBvcHRpb25hbCBzdHJpbmcgZnJvbTogXCJOREhXQ1wiLCBcIk5DREhXXCIuIERlZmF1bHRzIHRvXG4gKiAgICAgXCJOREhXQ1wiLiBTcGVjaWZ5IHRoZSBkYXRhIGZvcm1hdCBvZiB0aGUgaW5wdXQgYW5kIG91dHB1dCBkYXRhLiBXaXRoIHRoZVxuICogICAgIGRlZmF1bHQgZm9ybWF0IFwiTkRIV0NcIiwgdGhlIGRhdGEgaXMgc3RvcmVkIGluIHRoZSBvcmRlciBvZjogW2JhdGNoLFxuICogICAgIGRlcHRoLCBoZWlnaHQsIHdpZHRoLCBjaGFubmVsc10uIE9ubHkgXCJOREhXQ1wiIGlzIGN1cnJlbnRseSBzdXBwb3J0ZWQuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ09wZXJhdGlvbnMnLCBzdWJoZWFkaW5nOiAnQ29udm9sdXRpb24nfVxuICovXG5mdW5jdGlvbiBhdmdQb29sM2RfPFQgZXh0ZW5kcyBUZW5zb3I0RHxUZW5zb3I1RD4oXG4gICAgeDogVHxUZW5zb3JMaWtlLCBmaWx0ZXJTaXplOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl18bnVtYmVyLFxuICAgIHN0cmlkZXM6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXXxudW1iZXIsIHBhZDogJ3ZhbGlkJ3wnc2FtZSd8bnVtYmVyLFxuICAgIGRpbVJvdW5kaW5nTW9kZT86ICdmbG9vcid8J3JvdW5kJ3wnY2VpbCcsXG4gICAgZGF0YUZvcm1hdDogJ05ESFdDJ3wnTkNESFcnID0gJ05ESFdDJyk6IFQge1xuICBjb25zdCAkeCA9IGNvbnZlcnRUb1RlbnNvcih4LCAneCcsICdhdmdQb29sM2QnLCAnZmxvYXQzMicpO1xuXG4gIGxldCB4NUQgPSAkeCBhcyBUZW5zb3I1RDtcbiAgbGV0IHJlc2hhcGVkVG81RCA9IGZhbHNlO1xuICBpZiAoJHgucmFuayA9PT0gNCkge1xuICAgIHJlc2hhcGVkVG81RCA9IHRydWU7XG4gICAgeDVEID0gcmVzaGFwZSgkeCwgWzEsICR4LnNoYXBlWzBdLCAkeC5zaGFwZVsxXSwgJHguc2hhcGVbMl0sICR4LnNoYXBlWzNdXSk7XG4gIH1cblxuICB1dGlsLmFzc2VydChcbiAgICAgIHg1RC5yYW5rID09PSA1LFxuICAgICAgKCkgPT4gYEVycm9yIGluIGF2Z1Bvb2wzZDogeCBtdXN0IGJlIHJhbmsgNSBidXQgZ290IHJhbmsgJHt4NUQucmFua30uYCk7XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgZGF0YUZvcm1hdCA9PT0gJ05ESFdDJyxcbiAgICAgICgpID0+IGBFcnJvciBpbiBhdmdQb29sM2Q6IE9ubHkgTkRIV0MgaXMgY3VycmVudGx5IHN1cHBvcnRlZCwgYCArXG4gICAgICAgICAgYGJ1dCBnb3QgZGF0YUZvcm1hdCBvZiAke2RhdGFGb3JtYXR9YCk7XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgKHR5cGVvZiBzdHJpZGVzID09PSAnbnVtYmVyJyAmJiBzdHJpZGVzID4gMCkgfHxcbiAgICAgICAgICAoQXJyYXkuaXNBcnJheShzdHJpZGVzKSAmJiBzdHJpZGVzWzBdID4gMCAmJiBzdHJpZGVzWzFdID4gMCAmJlxuICAgICAgICAgICBzdHJpZGVzWzJdID4gMCksXG4gICAgICAoKSA9PiBgRXJyb3IgaW4gYXZnUG9vbDNkOiBTdHJpZGUgbXVzdCBiZSA+IDAsIGJ1dCBnb3QgJyR7c3RyaWRlc30nYCk7XG4gIGNoZWNrUGFkT25EaW1Sb3VuZGluZ01vZGUoJ2F2Z1Bvb2wzZCcsIHBhZCwgZGltUm91bmRpbmdNb2RlKTtcbiAgY29uc3QgaW5wdXRzOiBBdmdQb29sM0RJbnB1dHMgPSB7eDogeDVEfTtcbiAgY29uc3QgYXR0cnM6XG4gICAgICBBdmdQb29sM0RBdHRycyA9IHtmaWx0ZXJTaXplLCBzdHJpZGVzLCBwYWQsIGRpbVJvdW5kaW5nTW9kZSwgZGF0YUZvcm1hdH07XG5cbiAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOiBuby11bm5lY2Vzc2FyeS10eXBlLWFzc2VydGlvblxuICBsZXQgcmVzID0gRU5HSU5FLnJ1bktlcm5lbChcbiAgICAgICAgICAgICAgICBBdmdQb29sM0QsIGlucHV0cyBhcyB1bmtub3duIGFzIE5hbWVkVGVuc29yTWFwLFxuICAgICAgICAgICAgICAgIGF0dHJzIGFzIHVua25vd24gYXMgTmFtZWRBdHRyTWFwKSBhcyBUO1xuXG4gIHJlcyA9IGNhc3QocmVzLCB4NUQuZHR5cGUpO1xuXG4gIGlmIChyZXNoYXBlZFRvNUQpIHtcbiAgICByZXR1cm4gcmVzaGFwZShcbiAgICAgICAgICAgICAgIHJlcywgW3Jlcy5zaGFwZVsxXSwgcmVzLnNoYXBlWzJdLCByZXMuc2hhcGVbM10sIHJlcy5zaGFwZVs0XV0pIGFzXG4gICAgICAgIFQ7XG4gIH1cblxuICByZXR1cm4gcmVzO1xufVxuXG5leHBvcnQgY29uc3QgYXZnUG9vbDNkID0gLyogQF9fUFVSRV9fICovIG9wKHthdmdQb29sM2RffSk7XG4iXX0=