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
import { AvgPool3DGrad } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import * as util from '../util';
import { checkPadOnDimRoundingMode } from './conv_util';
import { op } from './operation';
import { reshape } from './reshape';
/**
 * Computes the backprop of a 3d avg pool.
 *
 * @param dy The dy error, of rank 5 of shape
 *     [batchSize, depth, height, width, channels].
 * assumed.
 * @param input The original input image, of rank 5 or rank4 of shape
 *     [batchSize, depth, height, width, channels].
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
function avgPool3dGrad_(dy, input, filterSize, strides, pad, dimRoundingMode) {
    const $dy = convertToTensor(dy, 'dy', 'avgPool3dGrad');
    const $input = convertToTensor(input, 'input', 'avgPool3dGrad');
    let dy5D = $dy;
    let input5D = $input;
    let reshapedTo5D = false;
    if ($input.rank === 4) {
        reshapedTo5D = true;
        dy5D = reshape($dy, [1, $dy.shape[0], $dy.shape[1], $dy.shape[2], $dy.shape[3]]);
        input5D = reshape($input, [
            1, $input.shape[0], $input.shape[1], $input.shape[2], $input.shape[3]
        ]);
    }
    util.assert(dy5D.rank === 5, () => `Error in avgPool3dGrad: dy must be rank 5 but got rank ` +
        `${dy5D.rank}.`);
    util.assert(input5D.rank === 5, () => `Error in avgPool3dGrad: input must be rank 5 but got rank ` +
        `${input5D.rank}.`);
    checkPadOnDimRoundingMode('avgPool3dGrad', pad, dimRoundingMode);
    const inputs = { dy: dy5D, input: input5D };
    const attrs = { filterSize, strides, pad, dimRoundingMode };
    // tslint:disable-next-line: no-unnecessary-type-assertion
    const res = ENGINE.runKernel(AvgPool3DGrad, inputs, attrs);
    if (reshapedTo5D) {
        return reshape(res, [res.shape[1], res.shape[2], res.shape[3], res.shape[4]]);
    }
    return res;
}
export const avgPool3dGrad = /* @__PURE__ */ op({ avgPool3dGrad_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYXZnX3Bvb2xfM2RfZ3JhZC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL2F2Z19wb29sXzNkX2dyYWQudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQ0E7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMsYUFBYSxFQUEwQyxNQUFNLGlCQUFpQixDQUFDO0FBSXZGLE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUVuRCxPQUFPLEtBQUssSUFBSSxNQUFNLFNBQVMsQ0FBQztBQUVoQyxPQUFPLEVBQUMseUJBQXlCLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFDdEQsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUMvQixPQUFPLEVBQUMsT0FBTyxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBRWxDOzs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBbUJHO0FBQ0gsU0FBUyxjQUFjLENBQ25CLEVBQWdCLEVBQUUsS0FBbUIsRUFDckMsVUFBMkMsRUFDM0MsT0FBd0MsRUFBRSxHQUEwQixFQUNwRSxlQUF3QztJQUMxQyxNQUFNLEdBQUcsR0FBRyxlQUFlLENBQUMsRUFBRSxFQUFFLElBQUksRUFBRSxlQUFlLENBQUMsQ0FBQztJQUN2RCxNQUFNLE1BQU0sR0FBRyxlQUFlLENBQUMsS0FBSyxFQUFFLE9BQU8sRUFBRSxlQUFlLENBQUMsQ0FBQztJQUVoRSxJQUFJLElBQUksR0FBRyxHQUFlLENBQUM7SUFDM0IsSUFBSSxPQUFPLEdBQUcsTUFBa0IsQ0FBQztJQUNqQyxJQUFJLFlBQVksR0FBRyxLQUFLLENBQUM7SUFFekIsSUFBSSxNQUFNLENBQUMsSUFBSSxLQUFLLENBQUMsRUFBRTtRQUNyQixZQUFZLEdBQUcsSUFBSSxDQUFDO1FBQ3BCLElBQUksR0FBRyxPQUFPLENBQ1YsR0FBRyxFQUFFLENBQUMsQ0FBQyxFQUFFLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxHQUFHLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ3RFLE9BQU8sR0FBRyxPQUFPLENBQUMsTUFBTSxFQUFFO1lBQ3hCLENBQUMsRUFBRSxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQztTQUN0RSxDQUFDLENBQUM7S0FDSjtJQUVELElBQUksQ0FBQyxNQUFNLENBQ1AsSUFBSSxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ2YsR0FBRyxFQUFFLENBQUMseURBQXlEO1FBQzNELEdBQUcsSUFBSSxDQUFDLElBQUksR0FBRyxDQUFDLENBQUM7SUFDekIsSUFBSSxDQUFDLE1BQU0sQ0FDUCxPQUFPLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDbEIsR0FBRyxFQUFFLENBQUMsNERBQTREO1FBQzlELEdBQUcsT0FBTyxDQUFDLElBQUksR0FBRyxDQUFDLENBQUM7SUFDNUIseUJBQXlCLENBQUMsZUFBZSxFQUFFLEdBQUcsRUFBRSxlQUFlLENBQUMsQ0FBQztJQUNqRSxNQUFNLE1BQU0sR0FBd0IsRUFBQyxFQUFFLEVBQUUsSUFBSSxFQUFFLEtBQUssRUFBRSxPQUFPLEVBQUMsQ0FBQztJQUMvRCxNQUFNLEtBQUssR0FBdUIsRUFBQyxVQUFVLEVBQUUsT0FBTyxFQUFFLEdBQUcsRUFBRSxlQUFlLEVBQUMsQ0FBQztJQUU5RSwwREFBMEQ7SUFDMUQsTUFBTSxHQUFHLEdBQUcsTUFBTSxDQUFDLFNBQVMsQ0FDWixhQUFhLEVBQUUsTUFBbUMsRUFDbEQsS0FBZ0MsQ0FBTSxDQUFDO0lBRXZELElBQUksWUFBWSxFQUFFO1FBQ2hCLE9BQU8sT0FBTyxDQUNILEdBQUcsRUFBRSxDQUFDLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxHQUFHLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FDbkUsQ0FBQztLQUNQO0lBRUQsT0FBTyxHQUFHLENBQUM7QUFDYixDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sYUFBYSxHQUFHLGVBQWUsQ0FBQyxFQUFFLENBQUMsRUFBQyxjQUFjLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiXG4vKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi9lbmdpbmUnO1xuaW1wb3J0IHtBdmdQb29sM0RHcmFkLCBBdmdQb29sM0RHcmFkQXR0cnMsIEF2Z1Bvb2wzREdyYWRJbnB1dHN9IGZyb20gJy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge05hbWVkQXR0ck1hcH0gZnJvbSAnLi4va2VybmVsX3JlZ2lzdHJ5JztcbmltcG9ydCB7VGVuc29yNEQsIFRlbnNvcjVEfSBmcm9tICcuLi90ZW5zb3InO1xuaW1wb3J0IHtOYW1lZFRlbnNvck1hcH0gZnJvbSAnLi4vdGVuc29yX3R5cGVzJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4uL3V0aWwnO1xuXG5pbXBvcnQge2NoZWNrUGFkT25EaW1Sb3VuZGluZ01vZGV9IGZyb20gJy4vY29udl91dGlsJztcbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcbmltcG9ydCB7cmVzaGFwZX0gZnJvbSAnLi9yZXNoYXBlJztcblxuLyoqXG4gKiBDb21wdXRlcyB0aGUgYmFja3Byb3Agb2YgYSAzZCBhdmcgcG9vbC5cbiAqXG4gKiBAcGFyYW0gZHkgVGhlIGR5IGVycm9yLCBvZiByYW5rIDUgb2Ygc2hhcGVcbiAqICAgICBbYmF0Y2hTaXplLCBkZXB0aCwgaGVpZ2h0LCB3aWR0aCwgY2hhbm5lbHNdLlxuICogYXNzdW1lZC5cbiAqIEBwYXJhbSBpbnB1dCBUaGUgb3JpZ2luYWwgaW5wdXQgaW1hZ2UsIG9mIHJhbmsgNSBvciByYW5rNCBvZiBzaGFwZVxuICogICAgIFtiYXRjaFNpemUsIGRlcHRoLCBoZWlnaHQsIHdpZHRoLCBjaGFubmVsc10uXG4gKiBAcGFyYW0gZmlsdGVyU2l6ZSBUaGUgZmlsdGVyIHNpemU6XG4gKiAgICAgYFtmaWx0ZXJEZXB0aCwgZmlsdGVySGVpZ2h0LCBmaWx0ZXJXaWR0aF1gLlxuICogICAgIGBmaWx0ZXJTaXplYCBpcyBhIHNpbmdsZSBudW1iZXIsXG4gKiAgICAgdGhlbiBgZmlsdGVyRGVwdGggPT0gZmlsdGVySGVpZ2h0ID09IGZpbHRlcldpZHRoYC5cbiAqIEBwYXJhbSBzdHJpZGVzIFRoZSBzdHJpZGVzIG9mIHRoZSBwb29saW5nOlxuICogICAgIGBbc3RyaWRlRGVwdGgsIHN0cmlkZUhlaWdodCwgc3RyaWRlV2lkdGhdYC4gSWZcbiAqICAgICBgc3RyaWRlc2AgaXMgYSBzaW5nbGUgbnVtYmVyLCB0aGVuIGBzdHJpZGVIZWlnaHQgPT0gc3RyaWRlV2lkdGhgLlxuICogQHBhcmFtIHBhZCBBIHN0cmluZyBmcm9tOiAnc2FtZScsICd2YWxpZCcuIFRoZSB0eXBlIG9mIHBhZGRpbmcgYWxnb3JpdGhtXG4gKiAgICAgdXNlZCBpbiB0aGUgZm9yd2FyZCBwcm9wIG9mIHRoZSBvcC5cbiAqIEBwYXJhbSBkaW1Sb3VuZGluZ01vZGUgQSBzdHJpbmcgZnJvbTogJ2NlaWwnLCAncm91bmQnLCAnZmxvb3InLiBJZiBub25lIGlzXG4gKiAgICAgcHJvdmlkZWQsIGl0IHdpbGwgZGVmYXVsdCB0byB0cnVuY2F0ZS5cbiAqL1xuZnVuY3Rpb24gYXZnUG9vbDNkR3JhZF88VCBleHRlbmRzIFRlbnNvcjREfFRlbnNvcjVEPihcbiAgICBkeTogVHxUZW5zb3JMaWtlLCBpbnB1dDogVHxUZW5zb3JMaWtlLFxuICAgIGZpbHRlclNpemU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXXxudW1iZXIsXG4gICAgc3RyaWRlczogW251bWJlciwgbnVtYmVyLCBudW1iZXJdfG51bWJlciwgcGFkOiAndmFsaWQnfCdzYW1lJ3xudW1iZXIsXG4gICAgZGltUm91bmRpbmdNb2RlPzogJ2Zsb29yJ3wncm91bmQnfCdjZWlsJyk6IFQge1xuICBjb25zdCAkZHkgPSBjb252ZXJ0VG9UZW5zb3IoZHksICdkeScsICdhdmdQb29sM2RHcmFkJyk7XG4gIGNvbnN0ICRpbnB1dCA9IGNvbnZlcnRUb1RlbnNvcihpbnB1dCwgJ2lucHV0JywgJ2F2Z1Bvb2wzZEdyYWQnKTtcblxuICBsZXQgZHk1RCA9ICRkeSBhcyBUZW5zb3I1RDtcbiAgbGV0IGlucHV0NUQgPSAkaW5wdXQgYXMgVGVuc29yNUQ7XG4gIGxldCByZXNoYXBlZFRvNUQgPSBmYWxzZTtcblxuICBpZiAoJGlucHV0LnJhbmsgPT09IDQpIHtcbiAgICByZXNoYXBlZFRvNUQgPSB0cnVlO1xuICAgIGR5NUQgPSByZXNoYXBlKFxuICAgICAgICAkZHksIFsxLCAkZHkuc2hhcGVbMF0sICRkeS5zaGFwZVsxXSwgJGR5LnNoYXBlWzJdLCAkZHkuc2hhcGVbM11dKTtcbiAgICBpbnB1dDVEID0gcmVzaGFwZSgkaW5wdXQsIFtcbiAgICAgIDEsICRpbnB1dC5zaGFwZVswXSwgJGlucHV0LnNoYXBlWzFdLCAkaW5wdXQuc2hhcGVbMl0sICRpbnB1dC5zaGFwZVszXVxuICAgIF0pO1xuICB9XG5cbiAgdXRpbC5hc3NlcnQoXG4gICAgICBkeTVELnJhbmsgPT09IDUsXG4gICAgICAoKSA9PiBgRXJyb3IgaW4gYXZnUG9vbDNkR3JhZDogZHkgbXVzdCBiZSByYW5rIDUgYnV0IGdvdCByYW5rIGAgK1xuICAgICAgICAgIGAke2R5NUQucmFua30uYCk7XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgaW5wdXQ1RC5yYW5rID09PSA1LFxuICAgICAgKCkgPT4gYEVycm9yIGluIGF2Z1Bvb2wzZEdyYWQ6IGlucHV0IG11c3QgYmUgcmFuayA1IGJ1dCBnb3QgcmFuayBgICtcbiAgICAgICAgICBgJHtpbnB1dDVELnJhbmt9LmApO1xuICBjaGVja1BhZE9uRGltUm91bmRpbmdNb2RlKCdhdmdQb29sM2RHcmFkJywgcGFkLCBkaW1Sb3VuZGluZ01vZGUpO1xuICBjb25zdCBpbnB1dHM6IEF2Z1Bvb2wzREdyYWRJbnB1dHMgPSB7ZHk6IGR5NUQsIGlucHV0OiBpbnB1dDVEfTtcbiAgY29uc3QgYXR0cnM6IEF2Z1Bvb2wzREdyYWRBdHRycyA9IHtmaWx0ZXJTaXplLCBzdHJpZGVzLCBwYWQsIGRpbVJvdW5kaW5nTW9kZX07XG5cbiAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOiBuby11bm5lY2Vzc2FyeS10eXBlLWFzc2VydGlvblxuICBjb25zdCByZXMgPSBFTkdJTkUucnVuS2VybmVsKFxuICAgICAgICAgICAgICAgICAgQXZnUG9vbDNER3JhZCwgaW5wdXRzIGFzIHVua25vd24gYXMgTmFtZWRUZW5zb3JNYXAsXG4gICAgICAgICAgICAgICAgICBhdHRycyBhcyB1bmtub3duIGFzIE5hbWVkQXR0ck1hcCkgYXMgVDtcblxuICBpZiAocmVzaGFwZWRUbzVEKSB7XG4gICAgcmV0dXJuIHJlc2hhcGUoXG4gICAgICAgICAgICAgICByZXMsIFtyZXMuc2hhcGVbMV0sIHJlcy5zaGFwZVsyXSwgcmVzLnNoYXBlWzNdLCByZXMuc2hhcGVbNF1dKSBhc1xuICAgICAgICBUO1xuICB9XG5cbiAgcmV0dXJuIHJlcztcbn1cblxuZXhwb3J0IGNvbnN0IGF2Z1Bvb2wzZEdyYWQgPSAvKiBAX19QVVJFX18gKi8gb3Aoe2F2Z1Bvb2wzZEdyYWRffSk7XG4iXX0=