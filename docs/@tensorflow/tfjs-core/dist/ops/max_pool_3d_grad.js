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
import { MaxPool3DGrad } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import * as util from '../util';
import { checkPadOnDimRoundingMode } from './conv_util';
import { op } from './operation';
import { reshape } from './reshape';
/**
 * Computes the backprop of a 3d max pool.
 *
 * @param dy The dy error, of rank 5 of shape
 *     [batchSize, depth, height, width, channels].
 * assumed.
 * @param input The original input image, of rank 5 or rank 4 of shape
 *     [batchSize, depth, height, width, channels].
 * @param output The original output image, of rank 5 of shape
 *     [batchSize, outDepth, outHeight, outWidth, channels].
 * @param filterSize The filter size:
 *     `[filterDepth, filterHeight, filterWidth]`.
 *     `filterSize` is a single number,
 *     then `filterDepth == filterHeight == filterWidth`.
 * @param strides The strides of the pooling:
 *     `[strideDepth, strideHeight, strideWidth]`. If
 *     `strides` is a single number, then `strideHeight == strideWidth`.
 * @param pad A string from: 'same', 'valid'. The type of padding algorithm
 *     used in the forward prop of the op.
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 */
function maxPool3dGrad_(dy, input, output, filterSize, strides, pad, dimRoundingMode) {
    const $dy = convertToTensor(dy, 'dy', 'maxPool3dGrad');
    const $input = convertToTensor(input, 'input', 'maxPool3dGrad');
    const $output = convertToTensor(output, 'output', 'maxPool3dGrad');
    let dy5D = $dy;
    let input5D = $input;
    let output5D = $output;
    let reshapedTo5D = false;
    if ($input.rank === 4) {
        reshapedTo5D = true;
        dy5D = reshape($dy, [1, $dy.shape[0], $dy.shape[1], $dy.shape[2], $dy.shape[3]]);
        input5D = reshape($input, [
            1, $input.shape[0], $input.shape[1], $input.shape[2], $input.shape[3]
        ]);
        output5D = reshape($output, [
            1, $output.shape[0], $output.shape[1], $output.shape[2], $output.shape[3]
        ]);
    }
    util.assert(dy5D.rank === 5, () => `Error in maxPool3dGrad: dy must be rank 5 but got rank ` +
        `${dy5D.rank}.`);
    util.assert(input5D.rank === 5, () => `Error in maxPool3dGrad: input must be rank 5 but got rank ` +
        `${input5D.rank}.`);
    util.assert(output5D.rank === 5, () => `Error in maxPool3dGrad: output must be rank 5 but got rank ` +
        `${output5D.rank}.`);
    checkPadOnDimRoundingMode('maxPool3dGrad', pad, dimRoundingMode);
    const inputs = { dy: dy5D, input: input5D, output: output5D };
    const attrs = { filterSize, strides, pad, dimRoundingMode };
    // tslint:disable-next-line: no-unnecessary-type-assertion
    const res = ENGINE.runKernel(MaxPool3DGrad, inputs, attrs);
    if (reshapedTo5D) {
        return reshape(res, [res.shape[1], res.shape[2], res.shape[3], res.shape[4]]);
    }
    return res;
}
export const maxPool3dGrad = /* @__PURE__ */ op({ maxPool3dGrad_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibWF4X3Bvb2xfM2RfZ3JhZC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL21heF9wb29sXzNkX2dyYWQudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMsYUFBYSxFQUEwQyxNQUFNLGlCQUFpQixDQUFDO0FBSXZGLE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUVuRCxPQUFPLEtBQUssSUFBSSxNQUFNLFNBQVMsQ0FBQztBQUVoQyxPQUFPLEVBQUMseUJBQXlCLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFDdEQsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUMvQixPQUFPLEVBQUMsT0FBTyxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBRWxDOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7R0FxQkc7QUFDSCxTQUFTLGNBQWMsQ0FDbkIsRUFBZ0IsRUFBRSxLQUFtQixFQUFFLE1BQW9CLEVBQzNELFVBQTJDLEVBQzNDLE9BQXdDLEVBQUUsR0FBMEIsRUFDcEUsZUFBd0M7SUFDMUMsTUFBTSxHQUFHLEdBQUcsZUFBZSxDQUFDLEVBQUUsRUFBRSxJQUFJLEVBQUUsZUFBZSxDQUFDLENBQUM7SUFDdkQsTUFBTSxNQUFNLEdBQUcsZUFBZSxDQUFDLEtBQUssRUFBRSxPQUFPLEVBQUUsZUFBZSxDQUFDLENBQUM7SUFDaEUsTUFBTSxPQUFPLEdBQUcsZUFBZSxDQUFDLE1BQU0sRUFBRSxRQUFRLEVBQUUsZUFBZSxDQUFDLENBQUM7SUFFbkUsSUFBSSxJQUFJLEdBQUcsR0FBZSxDQUFDO0lBQzNCLElBQUksT0FBTyxHQUFHLE1BQWtCLENBQUM7SUFDakMsSUFBSSxRQUFRLEdBQUcsT0FBbUIsQ0FBQztJQUNuQyxJQUFJLFlBQVksR0FBRyxLQUFLLENBQUM7SUFFekIsSUFBSSxNQUFNLENBQUMsSUFBSSxLQUFLLENBQUMsRUFBRTtRQUNyQixZQUFZLEdBQUcsSUFBSSxDQUFDO1FBQ3BCLElBQUksR0FBRyxPQUFPLENBQ1YsR0FBRyxFQUFFLENBQUMsQ0FBQyxFQUFFLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxHQUFHLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ3RFLE9BQU8sR0FBRyxPQUFPLENBQUMsTUFBTSxFQUFFO1lBQ3hCLENBQUMsRUFBRSxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQztTQUN0RSxDQUFDLENBQUM7UUFDSCxRQUFRLEdBQUcsT0FBTyxDQUFDLE9BQU8sRUFBRTtZQUMxQixDQUFDLEVBQUUsT0FBTyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxPQUFPLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLE9BQU8sQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsT0FBTyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUM7U0FDMUUsQ0FBQyxDQUFDO0tBQ0o7SUFFRCxJQUFJLENBQUMsTUFBTSxDQUNQLElBQUksQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNmLEdBQUcsRUFBRSxDQUFDLHlEQUF5RDtRQUMzRCxHQUFHLElBQUksQ0FBQyxJQUFJLEdBQUcsQ0FBQyxDQUFDO0lBQ3pCLElBQUksQ0FBQyxNQUFNLENBQ1AsT0FBTyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ2xCLEdBQUcsRUFBRSxDQUFDLDREQUE0RDtRQUM5RCxHQUFHLE9BQU8sQ0FBQyxJQUFJLEdBQUcsQ0FBQyxDQUFDO0lBQzVCLElBQUksQ0FBQyxNQUFNLENBQ1AsUUFBUSxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ25CLEdBQUcsRUFBRSxDQUFDLDZEQUE2RDtRQUMvRCxHQUFHLFFBQVEsQ0FBQyxJQUFJLEdBQUcsQ0FBQyxDQUFDO0lBQzdCLHlCQUF5QixDQUFDLGVBQWUsRUFBRSxHQUFHLEVBQUUsZUFBZSxDQUFDLENBQUM7SUFDakUsTUFBTSxNQUFNLEdBQ2MsRUFBQyxFQUFFLEVBQUUsSUFBSSxFQUFFLEtBQUssRUFBRSxPQUFPLEVBQUUsTUFBTSxFQUFFLFFBQVEsRUFBQyxDQUFDO0lBQ3ZFLE1BQU0sS0FBSyxHQUF1QixFQUFDLFVBQVUsRUFBRSxPQUFPLEVBQUUsR0FBRyxFQUFFLGVBQWUsRUFBQyxDQUFDO0lBRTlFLDBEQUEwRDtJQUMxRCxNQUFNLEdBQUcsR0FBRyxNQUFNLENBQUMsU0FBUyxDQUNaLGFBQWEsRUFBRSxNQUFtQyxFQUNsRCxLQUFnQyxDQUFNLENBQUM7SUFFdkQsSUFBSSxZQUFZLEVBQUU7UUFDaEIsT0FBTyxPQUFPLENBQ0gsR0FBRyxFQUFFLENBQUMsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxHQUFHLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUNuRSxDQUFDO0tBQ1A7SUFFRCxPQUFPLEdBQUcsQ0FBQztBQUNiLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxhQUFhLEdBQUcsZUFBZSxDQUFDLEVBQUUsQ0FBQyxFQUFDLGNBQWMsRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi9lbmdpbmUnO1xuaW1wb3J0IHtNYXhQb29sM0RHcmFkLCBNYXhQb29sM0RHcmFkQXR0cnMsIE1heFBvb2wzREdyYWRJbnB1dHN9IGZyb20gJy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge05hbWVkQXR0ck1hcH0gZnJvbSAnLi4va2VybmVsX3JlZ2lzdHJ5JztcbmltcG9ydCB7VGVuc29yNEQsIFRlbnNvcjVEfSBmcm9tICcuLi90ZW5zb3InO1xuaW1wb3J0IHtOYW1lZFRlbnNvck1hcH0gZnJvbSAnLi4vdGVuc29yX3R5cGVzJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4uL3V0aWwnO1xuXG5pbXBvcnQge2NoZWNrUGFkT25EaW1Sb3VuZGluZ01vZGV9IGZyb20gJy4vY29udl91dGlsJztcbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcbmltcG9ydCB7cmVzaGFwZX0gZnJvbSAnLi9yZXNoYXBlJztcblxuLyoqXG4gKiBDb21wdXRlcyB0aGUgYmFja3Byb3Agb2YgYSAzZCBtYXggcG9vbC5cbiAqXG4gKiBAcGFyYW0gZHkgVGhlIGR5IGVycm9yLCBvZiByYW5rIDUgb2Ygc2hhcGVcbiAqICAgICBbYmF0Y2hTaXplLCBkZXB0aCwgaGVpZ2h0LCB3aWR0aCwgY2hhbm5lbHNdLlxuICogYXNzdW1lZC5cbiAqIEBwYXJhbSBpbnB1dCBUaGUgb3JpZ2luYWwgaW5wdXQgaW1hZ2UsIG9mIHJhbmsgNSBvciByYW5rIDQgb2Ygc2hhcGVcbiAqICAgICBbYmF0Y2hTaXplLCBkZXB0aCwgaGVpZ2h0LCB3aWR0aCwgY2hhbm5lbHNdLlxuICogQHBhcmFtIG91dHB1dCBUaGUgb3JpZ2luYWwgb3V0cHV0IGltYWdlLCBvZiByYW5rIDUgb2Ygc2hhcGVcbiAqICAgICBbYmF0Y2hTaXplLCBvdXREZXB0aCwgb3V0SGVpZ2h0LCBvdXRXaWR0aCwgY2hhbm5lbHNdLlxuICogQHBhcmFtIGZpbHRlclNpemUgVGhlIGZpbHRlciBzaXplOlxuICogICAgIGBbZmlsdGVyRGVwdGgsIGZpbHRlckhlaWdodCwgZmlsdGVyV2lkdGhdYC5cbiAqICAgICBgZmlsdGVyU2l6ZWAgaXMgYSBzaW5nbGUgbnVtYmVyLFxuICogICAgIHRoZW4gYGZpbHRlckRlcHRoID09IGZpbHRlckhlaWdodCA9PSBmaWx0ZXJXaWR0aGAuXG4gKiBAcGFyYW0gc3RyaWRlcyBUaGUgc3RyaWRlcyBvZiB0aGUgcG9vbGluZzpcbiAqICAgICBgW3N0cmlkZURlcHRoLCBzdHJpZGVIZWlnaHQsIHN0cmlkZVdpZHRoXWAuIElmXG4gKiAgICAgYHN0cmlkZXNgIGlzIGEgc2luZ2xlIG51bWJlciwgdGhlbiBgc3RyaWRlSGVpZ2h0ID09IHN0cmlkZVdpZHRoYC5cbiAqIEBwYXJhbSBwYWQgQSBzdHJpbmcgZnJvbTogJ3NhbWUnLCAndmFsaWQnLiBUaGUgdHlwZSBvZiBwYWRkaW5nIGFsZ29yaXRobVxuICogICAgIHVzZWQgaW4gdGhlIGZvcndhcmQgcHJvcCBvZiB0aGUgb3AuXG4gKiBAcGFyYW0gZGltUm91bmRpbmdNb2RlIEEgc3RyaW5nIGZyb206ICdjZWlsJywgJ3JvdW5kJywgJ2Zsb29yJy4gSWYgbm9uZSBpc1xuICogICAgIHByb3ZpZGVkLCBpdCB3aWxsIGRlZmF1bHQgdG8gdHJ1bmNhdGUuXG4gKi9cbmZ1bmN0aW9uIG1heFBvb2wzZEdyYWRfPFQgZXh0ZW5kcyBUZW5zb3I0RHxUZW5zb3I1RD4oXG4gICAgZHk6IFR8VGVuc29yTGlrZSwgaW5wdXQ6IFR8VGVuc29yTGlrZSwgb3V0cHV0OiBUfFRlbnNvckxpa2UsXG4gICAgZmlsdGVyU2l6ZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdfG51bWJlcixcbiAgICBzdHJpZGVzOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl18bnVtYmVyLCBwYWQ6ICd2YWxpZCd8J3NhbWUnfG51bWJlcixcbiAgICBkaW1Sb3VuZGluZ01vZGU/OiAnZmxvb3InfCdyb3VuZCd8J2NlaWwnKTogVCB7XG4gIGNvbnN0ICRkeSA9IGNvbnZlcnRUb1RlbnNvcihkeSwgJ2R5JywgJ21heFBvb2wzZEdyYWQnKTtcbiAgY29uc3QgJGlucHV0ID0gY29udmVydFRvVGVuc29yKGlucHV0LCAnaW5wdXQnLCAnbWF4UG9vbDNkR3JhZCcpO1xuICBjb25zdCAkb3V0cHV0ID0gY29udmVydFRvVGVuc29yKG91dHB1dCwgJ291dHB1dCcsICdtYXhQb29sM2RHcmFkJyk7XG5cbiAgbGV0IGR5NUQgPSAkZHkgYXMgVGVuc29yNUQ7XG4gIGxldCBpbnB1dDVEID0gJGlucHV0IGFzIFRlbnNvcjVEO1xuICBsZXQgb3V0cHV0NUQgPSAkb3V0cHV0IGFzIFRlbnNvcjVEO1xuICBsZXQgcmVzaGFwZWRUbzVEID0gZmFsc2U7XG5cbiAgaWYgKCRpbnB1dC5yYW5rID09PSA0KSB7XG4gICAgcmVzaGFwZWRUbzVEID0gdHJ1ZTtcbiAgICBkeTVEID0gcmVzaGFwZShcbiAgICAgICAgJGR5LCBbMSwgJGR5LnNoYXBlWzBdLCAkZHkuc2hhcGVbMV0sICRkeS5zaGFwZVsyXSwgJGR5LnNoYXBlWzNdXSk7XG4gICAgaW5wdXQ1RCA9IHJlc2hhcGUoJGlucHV0LCBbXG4gICAgICAxLCAkaW5wdXQuc2hhcGVbMF0sICRpbnB1dC5zaGFwZVsxXSwgJGlucHV0LnNoYXBlWzJdLCAkaW5wdXQuc2hhcGVbM11cbiAgICBdKTtcbiAgICBvdXRwdXQ1RCA9IHJlc2hhcGUoJG91dHB1dCwgW1xuICAgICAgMSwgJG91dHB1dC5zaGFwZVswXSwgJG91dHB1dC5zaGFwZVsxXSwgJG91dHB1dC5zaGFwZVsyXSwgJG91dHB1dC5zaGFwZVszXVxuICAgIF0pO1xuICB9XG5cbiAgdXRpbC5hc3NlcnQoXG4gICAgICBkeTVELnJhbmsgPT09IDUsXG4gICAgICAoKSA9PiBgRXJyb3IgaW4gbWF4UG9vbDNkR3JhZDogZHkgbXVzdCBiZSByYW5rIDUgYnV0IGdvdCByYW5rIGAgK1xuICAgICAgICAgIGAke2R5NUQucmFua30uYCk7XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgaW5wdXQ1RC5yYW5rID09PSA1LFxuICAgICAgKCkgPT4gYEVycm9yIGluIG1heFBvb2wzZEdyYWQ6IGlucHV0IG11c3QgYmUgcmFuayA1IGJ1dCBnb3QgcmFuayBgICtcbiAgICAgICAgICBgJHtpbnB1dDVELnJhbmt9LmApO1xuICB1dGlsLmFzc2VydChcbiAgICAgIG91dHB1dDVELnJhbmsgPT09IDUsXG4gICAgICAoKSA9PiBgRXJyb3IgaW4gbWF4UG9vbDNkR3JhZDogb3V0cHV0IG11c3QgYmUgcmFuayA1IGJ1dCBnb3QgcmFuayBgICtcbiAgICAgICAgICBgJHtvdXRwdXQ1RC5yYW5rfS5gKTtcbiAgY2hlY2tQYWRPbkRpbVJvdW5kaW5nTW9kZSgnbWF4UG9vbDNkR3JhZCcsIHBhZCwgZGltUm91bmRpbmdNb2RlKTtcbiAgY29uc3QgaW5wdXRzOlxuICAgICAgTWF4UG9vbDNER3JhZElucHV0cyA9IHtkeTogZHk1RCwgaW5wdXQ6IGlucHV0NUQsIG91dHB1dDogb3V0cHV0NUR9O1xuICBjb25zdCBhdHRyczogTWF4UG9vbDNER3JhZEF0dHJzID0ge2ZpbHRlclNpemUsIHN0cmlkZXMsIHBhZCwgZGltUm91bmRpbmdNb2RlfTtcblxuICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6IG5vLXVubmVjZXNzYXJ5LXR5cGUtYXNzZXJ0aW9uXG4gIGNvbnN0IHJlcyA9IEVOR0lORS5ydW5LZXJuZWwoXG4gICAgICAgICAgICAgICAgICBNYXhQb29sM0RHcmFkLCBpbnB1dHMgYXMgdW5rbm93biBhcyBOYW1lZFRlbnNvck1hcCxcbiAgICAgICAgICAgICAgICAgIGF0dHJzIGFzIHVua25vd24gYXMgTmFtZWRBdHRyTWFwKSBhcyBUO1xuXG4gIGlmIChyZXNoYXBlZFRvNUQpIHtcbiAgICByZXR1cm4gcmVzaGFwZShcbiAgICAgICAgICAgICAgIHJlcywgW3Jlcy5zaGFwZVsxXSwgcmVzLnNoYXBlWzJdLCByZXMuc2hhcGVbM10sIHJlcy5zaGFwZVs0XV0pIGFzXG4gICAgICAgIFQ7XG4gIH1cblxuICByZXR1cm4gcmVzO1xufVxuXG5leHBvcnQgY29uc3QgbWF4UG9vbDNkR3JhZCA9IC8qIEBfX1BVUkVfXyAqLyBvcCh7bWF4UG9vbDNkR3JhZF99KTtcbiJdfQ==