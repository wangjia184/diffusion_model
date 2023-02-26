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
import { AvgPoolGrad } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import * as util from '../util';
import { op } from './operation';
import { reshape } from './reshape';
/**
 * Computes the backprop of an 2D avg pool.
 *
 * @param dy The dy error, of rank 4 or rank 3 of shape
 *     [batchSize, height, width, channels]. If rank 3, batch of 1 is
 * assumed.
 * @param input The input image, of rank 4 or rank 3 of shape
 *     [batchSize, height, width, channels]. If rank 3, batch of 1 is
 * assumed.
 * @param filterSize The filter size: `[filterHeight, filterWidth]`. If
 *     `filterSize` is a single number, then `filterHeight == filterWidth`.
 * @param strides The strides of the pooling: `[strideHeight, strideWidth]`. If
 *     `strides` is a single number, then `strideHeight == strideWidth`.
 * @param pad The type of padding algorithm used in the forward prop of the op.
 *     'same', 'valid', for more info, see this guide:
 *     [https://www.tensorflow.org/api_docs/python/tf/nn/convolution](
 *         https://www.tensorflow.org/api_docs/python/tf/nn/convolution)
 */
function avgPoolGrad_(dy, input, filterSize, strides, pad) {
    const $dy = convertToTensor(dy, 'dy', 'avgPoolGrad');
    const $input = convertToTensor(input, 'input', 'avgPoolGrad');
    util.assert($input.rank === $dy.rank, () => `Rank of input (${$input.rank}) does not match rank of dy (${$dy.rank})`);
    let input4D = $input;
    let dy4D = $dy;
    let reshapedTo4D = false;
    if ($input.rank === 3) {
        reshapedTo4D = true;
        input4D =
            reshape($input, [1, $input.shape[0], $input.shape[1], $input.shape[2]]);
        dy4D = reshape($dy, [1, $dy.shape[0], $dy.shape[1], $dy.shape[2]]);
    }
    util.assert(dy4D.rank === 4, () => `Error in avgPoolGrad: dy must be rank 4 but got rank ` +
        `${dy4D.rank}.`);
    util.assert(input4D.rank === 4, () => `Error in avgPoolGrad: input must be rank 4 but got rank ` +
        `${input4D.rank}.`);
    const inputs = { dy: dy4D, input: input4D };
    const attrs = { filterSize, strides, pad };
    // tslint:disable-next-line: no-unnecessary-type-assertion
    const res = ENGINE.runKernel(AvgPoolGrad, inputs, attrs);
    if (reshapedTo4D) {
        return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]);
    }
    return res;
}
export const avgPoolGrad = /* @__PURE__ */ op({ avgPoolGrad_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYXZnX3Bvb2xfZ3JhZC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL2F2Z19wb29sX2dyYWQudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMsV0FBVyxFQUFzQyxNQUFNLGlCQUFpQixDQUFDO0FBSWpGLE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUVuRCxPQUFPLEtBQUssSUFBSSxNQUFNLFNBQVMsQ0FBQztBQUdoQyxPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBQy9CLE9BQU8sRUFBQyxPQUFPLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFFbEM7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBaUJHO0FBQ0gsU0FBUyxZQUFZLENBQ2pCLEVBQWdCLEVBQUUsS0FBbUIsRUFBRSxVQUFtQyxFQUMxRSxPQUFnQyxFQUNoQyxHQUEwQztJQUM1QyxNQUFNLEdBQUcsR0FBRyxlQUFlLENBQUMsRUFBRSxFQUFFLElBQUksRUFBRSxhQUFhLENBQUMsQ0FBQztJQUNyRCxNQUFNLE1BQU0sR0FBRyxlQUFlLENBQUMsS0FBSyxFQUFFLE9BQU8sRUFBRSxhQUFhLENBQUMsQ0FBQztJQUU5RCxJQUFJLENBQUMsTUFBTSxDQUNQLE1BQU0sQ0FBQyxJQUFJLEtBQUssR0FBRyxDQUFDLElBQUksRUFDeEIsR0FBRyxFQUFFLENBQUMsa0JBQWtCLE1BQU0sQ0FBQyxJQUFJLGdDQUMvQixHQUFHLENBQUMsSUFBSSxHQUFHLENBQUMsQ0FBQztJQUVyQixJQUFJLE9BQU8sR0FBRyxNQUFrQixDQUFDO0lBQ2pDLElBQUksSUFBSSxHQUFHLEdBQWUsQ0FBQztJQUMzQixJQUFJLFlBQVksR0FBRyxLQUFLLENBQUM7SUFFekIsSUFBSSxNQUFNLENBQUMsSUFBSSxLQUFLLENBQUMsRUFBRTtRQUNyQixZQUFZLEdBQUcsSUFBSSxDQUFDO1FBQ3BCLE9BQU87WUFDSCxPQUFPLENBQUMsTUFBTSxFQUFFLENBQUMsQ0FBQyxFQUFFLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUM1RSxJQUFJLEdBQUcsT0FBTyxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsRUFBRSxHQUFHLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7S0FDcEU7SUFFRCxJQUFJLENBQUMsTUFBTSxDQUNQLElBQUksQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNmLEdBQUcsRUFBRSxDQUFDLHVEQUF1RDtRQUN6RCxHQUFHLElBQUksQ0FBQyxJQUFJLEdBQUcsQ0FBQyxDQUFDO0lBQ3pCLElBQUksQ0FBQyxNQUFNLENBQ1AsT0FBTyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ2xCLEdBQUcsRUFBRSxDQUFDLDBEQUEwRDtRQUM1RCxHQUFHLE9BQU8sQ0FBQyxJQUFJLEdBQUcsQ0FBQyxDQUFDO0lBRTVCLE1BQU0sTUFBTSxHQUFzQixFQUFDLEVBQUUsRUFBRSxJQUFJLEVBQUUsS0FBSyxFQUFFLE9BQU8sRUFBQyxDQUFDO0lBRTdELE1BQU0sS0FBSyxHQUFxQixFQUFDLFVBQVUsRUFBRSxPQUFPLEVBQUUsR0FBRyxFQUFDLENBQUM7SUFFM0QsMERBQTBEO0lBQzFELE1BQU0sR0FBRyxHQUFHLE1BQU0sQ0FBQyxTQUFTLENBQ1osV0FBVyxFQUFFLE1BQW1DLEVBQ2hELEtBQWdDLENBQU0sQ0FBQztJQUV2RCxJQUFJLFlBQVksRUFBRTtRQUNoQixPQUFPLE9BQU8sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxHQUFHLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFNLENBQUM7S0FDdEU7SUFDRCxPQUFPLEdBQUcsQ0FBQztBQUNiLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxXQUFXLEdBQUcsZUFBZSxDQUFDLEVBQUUsQ0FBQyxFQUFDLFlBQVksRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi9lbmdpbmUnO1xuaW1wb3J0IHtBdmdQb29sR3JhZCwgQXZnUG9vbEdyYWRBdHRycywgQXZnUG9vbEdyYWRJbnB1dHN9IGZyb20gJy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge05hbWVkQXR0ck1hcH0gZnJvbSAnLi4va2VybmVsX3JlZ2lzdHJ5JztcbmltcG9ydCB7VGVuc29yM0QsIFRlbnNvcjREfSBmcm9tICcuLi90ZW5zb3InO1xuaW1wb3J0IHtOYW1lZFRlbnNvck1hcH0gZnJvbSAnLi4vdGVuc29yX3R5cGVzJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4uL3V0aWwnO1xuXG5pbXBvcnQge0V4cGxpY2l0UGFkZGluZ30gZnJvbSAnLi9jb252X3V0aWwnO1xuaW1wb3J0IHtvcH0gZnJvbSAnLi9vcGVyYXRpb24nO1xuaW1wb3J0IHtyZXNoYXBlfSBmcm9tICcuL3Jlc2hhcGUnO1xuXG4vKipcbiAqIENvbXB1dGVzIHRoZSBiYWNrcHJvcCBvZiBhbiAyRCBhdmcgcG9vbC5cbiAqXG4gKiBAcGFyYW0gZHkgVGhlIGR5IGVycm9yLCBvZiByYW5rIDQgb3IgcmFuayAzIG9mIHNoYXBlXG4gKiAgICAgW2JhdGNoU2l6ZSwgaGVpZ2h0LCB3aWR0aCwgY2hhbm5lbHNdLiBJZiByYW5rIDMsIGJhdGNoIG9mIDEgaXNcbiAqIGFzc3VtZWQuXG4gKiBAcGFyYW0gaW5wdXQgVGhlIGlucHV0IGltYWdlLCBvZiByYW5rIDQgb3IgcmFuayAzIG9mIHNoYXBlXG4gKiAgICAgW2JhdGNoU2l6ZSwgaGVpZ2h0LCB3aWR0aCwgY2hhbm5lbHNdLiBJZiByYW5rIDMsIGJhdGNoIG9mIDEgaXNcbiAqIGFzc3VtZWQuXG4gKiBAcGFyYW0gZmlsdGVyU2l6ZSBUaGUgZmlsdGVyIHNpemU6IGBbZmlsdGVySGVpZ2h0LCBmaWx0ZXJXaWR0aF1gLiBJZlxuICogICAgIGBmaWx0ZXJTaXplYCBpcyBhIHNpbmdsZSBudW1iZXIsIHRoZW4gYGZpbHRlckhlaWdodCA9PSBmaWx0ZXJXaWR0aGAuXG4gKiBAcGFyYW0gc3RyaWRlcyBUaGUgc3RyaWRlcyBvZiB0aGUgcG9vbGluZzogYFtzdHJpZGVIZWlnaHQsIHN0cmlkZVdpZHRoXWAuIElmXG4gKiAgICAgYHN0cmlkZXNgIGlzIGEgc2luZ2xlIG51bWJlciwgdGhlbiBgc3RyaWRlSGVpZ2h0ID09IHN0cmlkZVdpZHRoYC5cbiAqIEBwYXJhbSBwYWQgVGhlIHR5cGUgb2YgcGFkZGluZyBhbGdvcml0aG0gdXNlZCBpbiB0aGUgZm9yd2FyZCBwcm9wIG9mIHRoZSBvcC5cbiAqICAgICAnc2FtZScsICd2YWxpZCcsIGZvciBtb3JlIGluZm8sIHNlZSB0aGlzIGd1aWRlOlxuICogICAgIFtodHRwczovL3d3dy50ZW5zb3JmbG93Lm9yZy9hcGlfZG9jcy9weXRob24vdGYvbm4vY29udm9sdXRpb25dKFxuICogICAgICAgICBodHRwczovL3d3dy50ZW5zb3JmbG93Lm9yZy9hcGlfZG9jcy9weXRob24vdGYvbm4vY29udm9sdXRpb24pXG4gKi9cbmZ1bmN0aW9uIGF2Z1Bvb2xHcmFkXzxUIGV4dGVuZHMgVGVuc29yM0R8VGVuc29yNEQ+KFxuICAgIGR5OiBUfFRlbnNvckxpa2UsIGlucHV0OiBUfFRlbnNvckxpa2UsIGZpbHRlclNpemU6IFtudW1iZXIsIG51bWJlcl18bnVtYmVyLFxuICAgIHN0cmlkZXM6IFtudW1iZXIsIG51bWJlcl18bnVtYmVyLFxuICAgIHBhZDogJ3ZhbGlkJ3wnc2FtZSd8bnVtYmVyfEV4cGxpY2l0UGFkZGluZyk6IFQge1xuICBjb25zdCAkZHkgPSBjb252ZXJ0VG9UZW5zb3IoZHksICdkeScsICdhdmdQb29sR3JhZCcpO1xuICBjb25zdCAkaW5wdXQgPSBjb252ZXJ0VG9UZW5zb3IoaW5wdXQsICdpbnB1dCcsICdhdmdQb29sR3JhZCcpO1xuXG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgJGlucHV0LnJhbmsgPT09ICRkeS5yYW5rLFxuICAgICAgKCkgPT4gYFJhbmsgb2YgaW5wdXQgKCR7JGlucHV0LnJhbmt9KSBkb2VzIG5vdCBtYXRjaCByYW5rIG9mIGR5ICgke1xuICAgICAgICAgICRkeS5yYW5rfSlgKTtcblxuICBsZXQgaW5wdXQ0RCA9ICRpbnB1dCBhcyBUZW5zb3I0RDtcbiAgbGV0IGR5NEQgPSAkZHkgYXMgVGVuc29yNEQ7XG4gIGxldCByZXNoYXBlZFRvNEQgPSBmYWxzZTtcblxuICBpZiAoJGlucHV0LnJhbmsgPT09IDMpIHtcbiAgICByZXNoYXBlZFRvNEQgPSB0cnVlO1xuICAgIGlucHV0NEQgPVxuICAgICAgICByZXNoYXBlKCRpbnB1dCwgWzEsICRpbnB1dC5zaGFwZVswXSwgJGlucHV0LnNoYXBlWzFdLCAkaW5wdXQuc2hhcGVbMl1dKTtcbiAgICBkeTREID0gcmVzaGFwZSgkZHksIFsxLCAkZHkuc2hhcGVbMF0sICRkeS5zaGFwZVsxXSwgJGR5LnNoYXBlWzJdXSk7XG4gIH1cblxuICB1dGlsLmFzc2VydChcbiAgICAgIGR5NEQucmFuayA9PT0gNCxcbiAgICAgICgpID0+IGBFcnJvciBpbiBhdmdQb29sR3JhZDogZHkgbXVzdCBiZSByYW5rIDQgYnV0IGdvdCByYW5rIGAgK1xuICAgICAgICAgIGAke2R5NEQucmFua30uYCk7XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgaW5wdXQ0RC5yYW5rID09PSA0LFxuICAgICAgKCkgPT4gYEVycm9yIGluIGF2Z1Bvb2xHcmFkOiBpbnB1dCBtdXN0IGJlIHJhbmsgNCBidXQgZ290IHJhbmsgYCArXG4gICAgICAgICAgYCR7aW5wdXQ0RC5yYW5rfS5gKTtcblxuICBjb25zdCBpbnB1dHM6IEF2Z1Bvb2xHcmFkSW5wdXRzID0ge2R5OiBkeTRELCBpbnB1dDogaW5wdXQ0RH07XG5cbiAgY29uc3QgYXR0cnM6IEF2Z1Bvb2xHcmFkQXR0cnMgPSB7ZmlsdGVyU2l6ZSwgc3RyaWRlcywgcGFkfTtcblxuICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6IG5vLXVubmVjZXNzYXJ5LXR5cGUtYXNzZXJ0aW9uXG4gIGNvbnN0IHJlcyA9IEVOR0lORS5ydW5LZXJuZWwoXG4gICAgICAgICAgICAgICAgICBBdmdQb29sR3JhZCwgaW5wdXRzIGFzIHVua25vd24gYXMgTmFtZWRUZW5zb3JNYXAsXG4gICAgICAgICAgICAgICAgICBhdHRycyBhcyB1bmtub3duIGFzIE5hbWVkQXR0ck1hcCkgYXMgVDtcblxuICBpZiAocmVzaGFwZWRUbzREKSB7XG4gICAgcmV0dXJuIHJlc2hhcGUocmVzLCBbcmVzLnNoYXBlWzFdLCByZXMuc2hhcGVbMl0sIHJlcy5zaGFwZVszXV0pIGFzIFQ7XG4gIH1cbiAgcmV0dXJuIHJlcztcbn1cblxuZXhwb3J0IGNvbnN0IGF2Z1Bvb2xHcmFkID0gLyogQF9fUFVSRV9fICovIG9wKHthdmdQb29sR3JhZF99KTtcbiJdfQ==