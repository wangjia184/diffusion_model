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
import { LRN } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import * as util from '../util';
import { op } from './operation';
import { reshape } from './reshape';
/**
 * Normalizes the activation of a local neighborhood across or within
 * channels.
 *
 * @param x The input tensor. The 4-D input tensor is treated as a 3-D array
 *     of 1D vectors (along the last dimension), and each vector is
 *     normalized independently.
 * @param depthRadius The number of adjacent channels in the 1D normalization
 *     window.
 * @param bias A constant bias term for the basis.
 * @param alpha A scale factor, usually positive.
 * @param beta An exponent.
 *
 * @doc {heading: 'Operations', subheading: 'Normalization'}
 */
function localResponseNormalization_(x, depthRadius = 5, bias = 1, alpha = 1, beta = 0.5) {
    const $x = convertToTensor(x, 'x', 'localResponseNormalization');
    util.assert($x.rank === 4 || $x.rank === 3, () => `Error in localResponseNormalization: x must be rank 3 or 4 but got
               rank ${$x.rank}.`);
    util.assert(util.isInt(depthRadius), () => `Error in localResponseNormalization: depthRadius must be an ` +
        `integer but got depthRadius ${depthRadius}.`);
    let x4D = $x;
    let reshapedTo4D = false;
    if ($x.rank === 3) {
        reshapedTo4D = true;
        x4D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2]]);
    }
    const inputs = { x: x4D };
    const attrs = { depthRadius, bias, alpha, beta };
    // tslint:disable-next-line: no-unnecessary-type-assertion
    const res = ENGINE.runKernel(LRN, inputs, attrs);
    if (reshapedTo4D) {
        return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]);
    }
    else {
        return res;
    }
}
export const localResponseNormalization = /* @__PURE__ */ op({ localResponseNormalization_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibG9jYWxfcmVzcG9uc2Vfbm9ybWFsaXphdGlvbi5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL2xvY2FsX3Jlc3BvbnNlX25vcm1hbGl6YXRpb24udHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMsR0FBRyxFQUFzQixNQUFNLGlCQUFpQixDQUFDO0FBSXpELE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUVuRCxPQUFPLEtBQUssSUFBSSxNQUFNLFNBQVMsQ0FBQztBQUVoQyxPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBQy9CLE9BQU8sRUFBQyxPQUFPLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFFbEM7Ozs7Ozs7Ozs7Ozs7O0dBY0c7QUFDSCxTQUFTLDJCQUEyQixDQUNoQyxDQUFlLEVBQUUsV0FBVyxHQUFHLENBQUMsRUFBRSxJQUFJLEdBQUcsQ0FBQyxFQUFFLEtBQUssR0FBRyxDQUFDLEVBQUUsSUFBSSxHQUFHLEdBQUc7SUFDbkUsTUFBTSxFQUFFLEdBQUcsZUFBZSxDQUFDLENBQUMsRUFBRSxHQUFHLEVBQUUsNEJBQTRCLENBQUMsQ0FBQztJQUNqRSxJQUFJLENBQUMsTUFBTSxDQUNQLEVBQUUsQ0FBQyxJQUFJLEtBQUssQ0FBQyxJQUFJLEVBQUUsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUM5QixHQUFHLEVBQUUsQ0FBQztzQkFDVSxFQUFFLENBQUMsSUFBSSxHQUFHLENBQUMsQ0FBQztJQUNoQyxJQUFJLENBQUMsTUFBTSxDQUNQLElBQUksQ0FBQyxLQUFLLENBQUMsV0FBVyxDQUFDLEVBQ3ZCLEdBQUcsRUFBRSxDQUFDLDhEQUE4RDtRQUNoRSwrQkFBK0IsV0FBVyxHQUFHLENBQUMsQ0FBQztJQUV2RCxJQUFJLEdBQUcsR0FBRyxFQUFjLENBQUM7SUFDekIsSUFBSSxZQUFZLEdBQUcsS0FBSyxDQUFDO0lBQ3pCLElBQUksRUFBRSxDQUFDLElBQUksS0FBSyxDQUFDLEVBQUU7UUFDakIsWUFBWSxHQUFHLElBQUksQ0FBQztRQUNwQixHQUFHLEdBQUcsT0FBTyxDQUFDLEVBQUUsRUFBRSxDQUFDLENBQUMsRUFBRSxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsRUFBRSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7S0FDL0Q7SUFFRCxNQUFNLE1BQU0sR0FBYyxFQUFDLENBQUMsRUFBRSxHQUFHLEVBQUMsQ0FBQztJQUVuQyxNQUFNLEtBQUssR0FBYSxFQUFDLFdBQVcsRUFBRSxJQUFJLEVBQUUsS0FBSyxFQUFFLElBQUksRUFBQyxDQUFDO0lBRXpELDBEQUEwRDtJQUMxRCxNQUFNLEdBQUcsR0FBRyxNQUFNLENBQUMsU0FBUyxDQUNaLEdBQUcsRUFBRSxNQUFtQyxFQUN4QyxLQUFnQyxDQUFNLENBQUM7SUFFdkQsSUFBSSxZQUFZLEVBQUU7UUFDaEIsT0FBTyxPQUFPLENBQUMsR0FBRyxFQUFFLENBQUMsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxHQUFHLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBTSxDQUFDO0tBQ3RFO1NBQU07UUFDTCxPQUFPLEdBQUcsQ0FBQztLQUNaO0FBQ0gsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLDBCQUEwQixHQUFHLGVBQWUsQ0FBQyxFQUFFLENBQUMsRUFBQywyQkFBMkIsRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi9lbmdpbmUnO1xuaW1wb3J0IHtMUk4sIExSTkF0dHJzLCBMUk5JbnB1dHN9IGZyb20gJy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge05hbWVkQXR0ck1hcH0gZnJvbSAnLi4va2VybmVsX3JlZ2lzdHJ5JztcbmltcG9ydCB7VGVuc29yM0QsIFRlbnNvcjREfSBmcm9tICcuLi90ZW5zb3InO1xuaW1wb3J0IHtOYW1lZFRlbnNvck1hcH0gZnJvbSAnLi4vdGVuc29yX3R5cGVzJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4uL3V0aWwnO1xuXG5pbXBvcnQge29wfSBmcm9tICcuL29wZXJhdGlvbic7XG5pbXBvcnQge3Jlc2hhcGV9IGZyb20gJy4vcmVzaGFwZSc7XG5cbi8qKlxuICogTm9ybWFsaXplcyB0aGUgYWN0aXZhdGlvbiBvZiBhIGxvY2FsIG5laWdoYm9yaG9vZCBhY3Jvc3Mgb3Igd2l0aGluXG4gKiBjaGFubmVscy5cbiAqXG4gKiBAcGFyYW0geCBUaGUgaW5wdXQgdGVuc29yLiBUaGUgNC1EIGlucHV0IHRlbnNvciBpcyB0cmVhdGVkIGFzIGEgMy1EIGFycmF5XG4gKiAgICAgb2YgMUQgdmVjdG9ycyAoYWxvbmcgdGhlIGxhc3QgZGltZW5zaW9uKSwgYW5kIGVhY2ggdmVjdG9yIGlzXG4gKiAgICAgbm9ybWFsaXplZCBpbmRlcGVuZGVudGx5LlxuICogQHBhcmFtIGRlcHRoUmFkaXVzIFRoZSBudW1iZXIgb2YgYWRqYWNlbnQgY2hhbm5lbHMgaW4gdGhlIDFEIG5vcm1hbGl6YXRpb25cbiAqICAgICB3aW5kb3cuXG4gKiBAcGFyYW0gYmlhcyBBIGNvbnN0YW50IGJpYXMgdGVybSBmb3IgdGhlIGJhc2lzLlxuICogQHBhcmFtIGFscGhhIEEgc2NhbGUgZmFjdG9yLCB1c3VhbGx5IHBvc2l0aXZlLlxuICogQHBhcmFtIGJldGEgQW4gZXhwb25lbnQuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ09wZXJhdGlvbnMnLCBzdWJoZWFkaW5nOiAnTm9ybWFsaXphdGlvbid9XG4gKi9cbmZ1bmN0aW9uIGxvY2FsUmVzcG9uc2VOb3JtYWxpemF0aW9uXzxUIGV4dGVuZHMgVGVuc29yM0R8VGVuc29yNEQ+KFxuICAgIHg6IFR8VGVuc29yTGlrZSwgZGVwdGhSYWRpdXMgPSA1LCBiaWFzID0gMSwgYWxwaGEgPSAxLCBiZXRhID0gMC41KTogVCB7XG4gIGNvbnN0ICR4ID0gY29udmVydFRvVGVuc29yKHgsICd4JywgJ2xvY2FsUmVzcG9uc2VOb3JtYWxpemF0aW9uJyk7XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgJHgucmFuayA9PT0gNCB8fCAkeC5yYW5rID09PSAzLFxuICAgICAgKCkgPT4gYEVycm9yIGluIGxvY2FsUmVzcG9uc2VOb3JtYWxpemF0aW9uOiB4IG11c3QgYmUgcmFuayAzIG9yIDQgYnV0IGdvdFxuICAgICAgICAgICAgICAgcmFuayAkeyR4LnJhbmt9LmApO1xuICB1dGlsLmFzc2VydChcbiAgICAgIHV0aWwuaXNJbnQoZGVwdGhSYWRpdXMpLFxuICAgICAgKCkgPT4gYEVycm9yIGluIGxvY2FsUmVzcG9uc2VOb3JtYWxpemF0aW9uOiBkZXB0aFJhZGl1cyBtdXN0IGJlIGFuIGAgK1xuICAgICAgICAgIGBpbnRlZ2VyIGJ1dCBnb3QgZGVwdGhSYWRpdXMgJHtkZXB0aFJhZGl1c30uYCk7XG5cbiAgbGV0IHg0RCA9ICR4IGFzIFRlbnNvcjREO1xuICBsZXQgcmVzaGFwZWRUbzREID0gZmFsc2U7XG4gIGlmICgkeC5yYW5rID09PSAzKSB7XG4gICAgcmVzaGFwZWRUbzREID0gdHJ1ZTtcbiAgICB4NEQgPSByZXNoYXBlKCR4LCBbMSwgJHguc2hhcGVbMF0sICR4LnNoYXBlWzFdLCAkeC5zaGFwZVsyXV0pO1xuICB9XG5cbiAgY29uc3QgaW5wdXRzOiBMUk5JbnB1dHMgPSB7eDogeDREfTtcblxuICBjb25zdCBhdHRyczogTFJOQXR0cnMgPSB7ZGVwdGhSYWRpdXMsIGJpYXMsIGFscGhhLCBiZXRhfTtcblxuICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6IG5vLXVubmVjZXNzYXJ5LXR5cGUtYXNzZXJ0aW9uXG4gIGNvbnN0IHJlcyA9IEVOR0lORS5ydW5LZXJuZWwoXG4gICAgICAgICAgICAgICAgICBMUk4sIGlucHV0cyBhcyB1bmtub3duIGFzIE5hbWVkVGVuc29yTWFwLFxuICAgICAgICAgICAgICAgICAgYXR0cnMgYXMgdW5rbm93biBhcyBOYW1lZEF0dHJNYXApIGFzIFQ7XG5cbiAgaWYgKHJlc2hhcGVkVG80RCkge1xuICAgIHJldHVybiByZXNoYXBlKHJlcywgW3Jlcy5zaGFwZVsxXSwgcmVzLnNoYXBlWzJdLCByZXMuc2hhcGVbM11dKSBhcyBUO1xuICB9IGVsc2Uge1xuICAgIHJldHVybiByZXM7XG4gIH1cbn1cblxuZXhwb3J0IGNvbnN0IGxvY2FsUmVzcG9uc2VOb3JtYWxpemF0aW9uID0gLyogQF9fUFVSRV9fICovIG9wKHtsb2NhbFJlc3BvbnNlTm9ybWFsaXphdGlvbl99KTtcbiJdfQ==