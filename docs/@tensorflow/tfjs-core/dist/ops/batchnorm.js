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
import { FusedBatchNorm } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import * as util from '../util';
import { xAs4D } from './batchnorm_util';
import { op } from './operation';
import { reshape } from './reshape';
/**
 * Batch normalization.
 *
 * As described in
 * [http://arxiv.org/abs/1502.03167](http://arxiv.org/abs/1502.03167).
 *
 * Mean, variance, scale, and offset can be of two shapes:
 *   - The same shape as the input.
 *   - In the common case, the depth dimension is the last dimension of x, so
 *     the values would be a `tf.Tensor1D` of shape [depth].
 *
 * Also available are stricter rank-specific methods with the same signature
 * as this method that assert that parameters passed are of given rank
 *   - `tf.batchNorm2d`
 *   - `tf.batchNorm3d`
 *   - `tf.batchNorm4d`
 *
 * @param x The input Tensor.
 * @param mean A mean Tensor.
 * @param variance A variance Tensor.
 * @param offset An offset Tensor.
 * @param scale A scale Tensor.
 * @param varianceEpsilon A small float number to avoid dividing by 0.
 *
 * @doc {heading: 'Operations', subheading: 'Normalization'}
 */
function batchNorm_(x, mean, variance, offset, scale, varianceEpsilon) {
    if (varianceEpsilon == null) {
        varianceEpsilon = 0.001;
    }
    const $x = convertToTensor(x, 'x', 'batchNorm');
    const $mean = convertToTensor(mean, 'mean', 'batchNorm');
    const $variance = convertToTensor(variance, 'variance', 'batchNorm');
    let $scale;
    if (scale != null) {
        $scale = convertToTensor(scale, 'scale', 'batchNorm');
    }
    let $offset;
    if (offset != null) {
        $offset = convertToTensor(offset, 'offset', 'batchNorm');
    }
    util.assert($mean.rank === $variance.rank, () => 'Batch normalization gradient requires mean and variance to have ' +
        'equal ranks.');
    util.assert($offset == null || $mean.rank === $offset.rank, () => 'Batch normalization gradient requires mean and offset to have ' +
        'equal ranks.');
    util.assert($scale == null || $mean.rank === $scale.rank, () => 'Batch normalization gradient requires mean and scale to have ' +
        'equal ranks.');
    const x4D = xAs4D($x);
    const inputs = {
        x: x4D,
        scale: $scale,
        offset: $offset,
        mean: $mean,
        variance: $variance
    };
    const attrs = { varianceEpsilon };
    // tslint:disable-next-line: no-unnecessary-type-assertion
    const res = ENGINE.runKernel(FusedBatchNorm, inputs, attrs);
    return reshape(res, $x.shape);
}
export const batchNorm = /* @__PURE__ */ op({ batchNorm_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYmF0Y2hub3JtLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvYmF0Y2hub3JtLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDakMsT0FBTyxFQUFDLGNBQWMsRUFBNEMsTUFBTSxpQkFBaUIsQ0FBQztBQUkxRixPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFFbkQsT0FBTyxLQUFLLElBQUksTUFBTSxTQUFTLENBQUM7QUFFaEMsT0FBTyxFQUFDLEtBQUssRUFBQyxNQUFNLGtCQUFrQixDQUFDO0FBQ3ZDLE9BQU8sRUFBQyxFQUFFLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFDL0IsT0FBTyxFQUFDLE9BQU8sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUVsQzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQXlCRztBQUNILFNBQVMsVUFBVSxDQUNmLENBQXVCLEVBQUUsSUFBbUMsRUFDNUQsUUFBdUMsRUFDdkMsTUFBc0MsRUFDdEMsS0FBcUMsRUFDckMsZUFBd0I7SUFDMUIsSUFBSSxlQUFlLElBQUksSUFBSSxFQUFFO1FBQzNCLGVBQWUsR0FBRyxLQUFLLENBQUM7S0FDekI7SUFDRCxNQUFNLEVBQUUsR0FBRyxlQUFlLENBQUMsQ0FBQyxFQUFFLEdBQUcsRUFBRSxXQUFXLENBQUMsQ0FBQztJQUNoRCxNQUFNLEtBQUssR0FBRyxlQUFlLENBQUMsSUFBSSxFQUFFLE1BQU0sRUFBRSxXQUFXLENBQUMsQ0FBQztJQUN6RCxNQUFNLFNBQVMsR0FBRyxlQUFlLENBQUMsUUFBUSxFQUFFLFVBQVUsRUFBRSxXQUFXLENBQUMsQ0FBQztJQUNyRSxJQUFJLE1BQTBCLENBQUM7SUFDL0IsSUFBSSxLQUFLLElBQUksSUFBSSxFQUFFO1FBQ2pCLE1BQU0sR0FBRyxlQUFlLENBQUMsS0FBSyxFQUFFLE9BQU8sRUFBRSxXQUFXLENBQUMsQ0FBQztLQUN2RDtJQUNELElBQUksT0FBMkIsQ0FBQztJQUNoQyxJQUFJLE1BQU0sSUFBSSxJQUFJLEVBQUU7UUFDbEIsT0FBTyxHQUFHLGVBQWUsQ0FBQyxNQUFNLEVBQUUsUUFBUSxFQUFFLFdBQVcsQ0FBQyxDQUFDO0tBQzFEO0lBRUQsSUFBSSxDQUFDLE1BQU0sQ0FDUCxLQUFLLENBQUMsSUFBSSxLQUFLLFNBQVMsQ0FBQyxJQUFJLEVBQzdCLEdBQUcsRUFBRSxDQUFDLGtFQUFrRTtRQUNwRSxjQUFjLENBQUMsQ0FBQztJQUN4QixJQUFJLENBQUMsTUFBTSxDQUNQLE9BQU8sSUFBSSxJQUFJLElBQUksS0FBSyxDQUFDLElBQUksS0FBSyxPQUFPLENBQUMsSUFBSSxFQUM5QyxHQUFHLEVBQUUsQ0FBQyxnRUFBZ0U7UUFDbEUsY0FBYyxDQUFDLENBQUM7SUFDeEIsSUFBSSxDQUFDLE1BQU0sQ0FDUCxNQUFNLElBQUksSUFBSSxJQUFJLEtBQUssQ0FBQyxJQUFJLEtBQUssTUFBTSxDQUFDLElBQUksRUFDNUMsR0FBRyxFQUFFLENBQUMsK0RBQStEO1FBQ2pFLGNBQWMsQ0FBQyxDQUFDO0lBRXhCLE1BQU0sR0FBRyxHQUFhLEtBQUssQ0FBQyxFQUFFLENBQUMsQ0FBQztJQUVoQyxNQUFNLE1BQU0sR0FBeUI7UUFDbkMsQ0FBQyxFQUFFLEdBQUc7UUFDTixLQUFLLEVBQUUsTUFBTTtRQUNiLE1BQU0sRUFBRSxPQUFPO1FBQ2YsSUFBSSxFQUFFLEtBQUs7UUFDWCxRQUFRLEVBQUUsU0FBUztLQUNwQixDQUFDO0lBRUYsTUFBTSxLQUFLLEdBQXdCLEVBQUMsZUFBZSxFQUFDLENBQUM7SUFFckQsMERBQTBEO0lBQzFELE1BQU0sR0FBRyxHQUFHLE1BQU0sQ0FBQyxTQUFTLENBQ1osY0FBYyxFQUFFLE1BQW1DLEVBQ25ELEtBQWdDLENBQWMsQ0FBQztJQUUvRCxPQUFPLE9BQU8sQ0FBQyxHQUFHLEVBQUUsRUFBRSxDQUFDLEtBQUssQ0FBQyxDQUFDO0FBQ2hDLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxTQUFTLEdBQUcsZUFBZSxDQUFDLEVBQUUsQ0FBQyxFQUFDLFVBQVUsRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi9lbmdpbmUnO1xuaW1wb3J0IHtGdXNlZEJhdGNoTm9ybSwgRnVzZWRCYXRjaE5vcm1BdHRycywgRnVzZWRCYXRjaE5vcm1JbnB1dHN9IGZyb20gJy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge05hbWVkQXR0ck1hcH0gZnJvbSAnLi4va2VybmVsX3JlZ2lzdHJ5JztcbmltcG9ydCB7VGVuc29yLCBUZW5zb3IxRCwgVGVuc29yNER9IGZyb20gJy4uL3RlbnNvcic7XG5pbXBvcnQge05hbWVkVGVuc29yTWFwfSBmcm9tICcuLi90ZW5zb3JfdHlwZXMnO1xuaW1wb3J0IHtjb252ZXJ0VG9UZW5zb3J9IGZyb20gJy4uL3RlbnNvcl91dGlsX2Vudic7XG5pbXBvcnQge1JhbmssIFRlbnNvckxpa2V9IGZyb20gJy4uL3R5cGVzJztcbmltcG9ydCAqIGFzIHV0aWwgZnJvbSAnLi4vdXRpbCc7XG5cbmltcG9ydCB7eEFzNER9IGZyb20gJy4vYmF0Y2hub3JtX3V0aWwnO1xuaW1wb3J0IHtvcH0gZnJvbSAnLi9vcGVyYXRpb24nO1xuaW1wb3J0IHtyZXNoYXBlfSBmcm9tICcuL3Jlc2hhcGUnO1xuXG4vKipcbiAqIEJhdGNoIG5vcm1hbGl6YXRpb24uXG4gKlxuICogQXMgZGVzY3JpYmVkIGluXG4gKiBbaHR0cDovL2FyeGl2Lm9yZy9hYnMvMTUwMi4wMzE2N10oaHR0cDovL2FyeGl2Lm9yZy9hYnMvMTUwMi4wMzE2NykuXG4gKlxuICogTWVhbiwgdmFyaWFuY2UsIHNjYWxlLCBhbmQgb2Zmc2V0IGNhbiBiZSBvZiB0d28gc2hhcGVzOlxuICogICAtIFRoZSBzYW1lIHNoYXBlIGFzIHRoZSBpbnB1dC5cbiAqICAgLSBJbiB0aGUgY29tbW9uIGNhc2UsIHRoZSBkZXB0aCBkaW1lbnNpb24gaXMgdGhlIGxhc3QgZGltZW5zaW9uIG9mIHgsIHNvXG4gKiAgICAgdGhlIHZhbHVlcyB3b3VsZCBiZSBhIGB0Zi5UZW5zb3IxRGAgb2Ygc2hhcGUgW2RlcHRoXS5cbiAqXG4gKiBBbHNvIGF2YWlsYWJsZSBhcmUgc3RyaWN0ZXIgcmFuay1zcGVjaWZpYyBtZXRob2RzIHdpdGggdGhlIHNhbWUgc2lnbmF0dXJlXG4gKiBhcyB0aGlzIG1ldGhvZCB0aGF0IGFzc2VydCB0aGF0IHBhcmFtZXRlcnMgcGFzc2VkIGFyZSBvZiBnaXZlbiByYW5rXG4gKiAgIC0gYHRmLmJhdGNoTm9ybTJkYFxuICogICAtIGB0Zi5iYXRjaE5vcm0zZGBcbiAqICAgLSBgdGYuYmF0Y2hOb3JtNGRgXG4gKlxuICogQHBhcmFtIHggVGhlIGlucHV0IFRlbnNvci5cbiAqIEBwYXJhbSBtZWFuIEEgbWVhbiBUZW5zb3IuXG4gKiBAcGFyYW0gdmFyaWFuY2UgQSB2YXJpYW5jZSBUZW5zb3IuXG4gKiBAcGFyYW0gb2Zmc2V0IEFuIG9mZnNldCBUZW5zb3IuXG4gKiBAcGFyYW0gc2NhbGUgQSBzY2FsZSBUZW5zb3IuXG4gKiBAcGFyYW0gdmFyaWFuY2VFcHNpbG9uIEEgc21hbGwgZmxvYXQgbnVtYmVyIHRvIGF2b2lkIGRpdmlkaW5nIGJ5IDAuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ09wZXJhdGlvbnMnLCBzdWJoZWFkaW5nOiAnTm9ybWFsaXphdGlvbid9XG4gKi9cbmZ1bmN0aW9uIGJhdGNoTm9ybV88UiBleHRlbmRzIFJhbms+KFxuICAgIHg6IFRlbnNvcjxSPnxUZW5zb3JMaWtlLCBtZWFuOiBUZW5zb3I8Uj58VGVuc29yMUR8VGVuc29yTGlrZSxcbiAgICB2YXJpYW5jZTogVGVuc29yPFI+fFRlbnNvcjFEfFRlbnNvckxpa2UsXG4gICAgb2Zmc2V0PzogVGVuc29yPFI+fFRlbnNvcjFEfFRlbnNvckxpa2UsXG4gICAgc2NhbGU/OiBUZW5zb3I8Uj58VGVuc29yMUR8VGVuc29yTGlrZSxcbiAgICB2YXJpYW5jZUVwc2lsb24/OiBudW1iZXIpOiBUZW5zb3I8Uj4ge1xuICBpZiAodmFyaWFuY2VFcHNpbG9uID09IG51bGwpIHtcbiAgICB2YXJpYW5jZUVwc2lsb24gPSAwLjAwMTtcbiAgfVxuICBjb25zdCAkeCA9IGNvbnZlcnRUb1RlbnNvcih4LCAneCcsICdiYXRjaE5vcm0nKTtcbiAgY29uc3QgJG1lYW4gPSBjb252ZXJ0VG9UZW5zb3IobWVhbiwgJ21lYW4nLCAnYmF0Y2hOb3JtJyk7XG4gIGNvbnN0ICR2YXJpYW5jZSA9IGNvbnZlcnRUb1RlbnNvcih2YXJpYW5jZSwgJ3ZhcmlhbmNlJywgJ2JhdGNoTm9ybScpO1xuICBsZXQgJHNjYWxlOiBUZW5zb3I8Uj58VGVuc29yMUQ7XG4gIGlmIChzY2FsZSAhPSBudWxsKSB7XG4gICAgJHNjYWxlID0gY29udmVydFRvVGVuc29yKHNjYWxlLCAnc2NhbGUnLCAnYmF0Y2hOb3JtJyk7XG4gIH1cbiAgbGV0ICRvZmZzZXQ6IFRlbnNvcjxSPnxUZW5zb3IxRDtcbiAgaWYgKG9mZnNldCAhPSBudWxsKSB7XG4gICAgJG9mZnNldCA9IGNvbnZlcnRUb1RlbnNvcihvZmZzZXQsICdvZmZzZXQnLCAnYmF0Y2hOb3JtJyk7XG4gIH1cblxuICB1dGlsLmFzc2VydChcbiAgICAgICRtZWFuLnJhbmsgPT09ICR2YXJpYW5jZS5yYW5rLFxuICAgICAgKCkgPT4gJ0JhdGNoIG5vcm1hbGl6YXRpb24gZ3JhZGllbnQgcmVxdWlyZXMgbWVhbiBhbmQgdmFyaWFuY2UgdG8gaGF2ZSAnICtcbiAgICAgICAgICAnZXF1YWwgcmFua3MuJyk7XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgJG9mZnNldCA9PSBudWxsIHx8ICRtZWFuLnJhbmsgPT09ICRvZmZzZXQucmFuayxcbiAgICAgICgpID0+ICdCYXRjaCBub3JtYWxpemF0aW9uIGdyYWRpZW50IHJlcXVpcmVzIG1lYW4gYW5kIG9mZnNldCB0byBoYXZlICcgK1xuICAgICAgICAgICdlcXVhbCByYW5rcy4nKTtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICAkc2NhbGUgPT0gbnVsbCB8fCAkbWVhbi5yYW5rID09PSAkc2NhbGUucmFuayxcbiAgICAgICgpID0+ICdCYXRjaCBub3JtYWxpemF0aW9uIGdyYWRpZW50IHJlcXVpcmVzIG1lYW4gYW5kIHNjYWxlIHRvIGhhdmUgJyArXG4gICAgICAgICAgJ2VxdWFsIHJhbmtzLicpO1xuXG4gIGNvbnN0IHg0RDogVGVuc29yNEQgPSB4QXM0RCgkeCk7XG5cbiAgY29uc3QgaW5wdXRzOiBGdXNlZEJhdGNoTm9ybUlucHV0cyA9IHtcbiAgICB4OiB4NEQsXG4gICAgc2NhbGU6ICRzY2FsZSxcbiAgICBvZmZzZXQ6ICRvZmZzZXQsXG4gICAgbWVhbjogJG1lYW4sXG4gICAgdmFyaWFuY2U6ICR2YXJpYW5jZVxuICB9O1xuXG4gIGNvbnN0IGF0dHJzOiBGdXNlZEJhdGNoTm9ybUF0dHJzID0ge3ZhcmlhbmNlRXBzaWxvbn07XG5cbiAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOiBuby11bm5lY2Vzc2FyeS10eXBlLWFzc2VydGlvblxuICBjb25zdCByZXMgPSBFTkdJTkUucnVuS2VybmVsKFxuICAgICAgICAgICAgICAgICAgRnVzZWRCYXRjaE5vcm0sIGlucHV0cyBhcyB1bmtub3duIGFzIE5hbWVkVGVuc29yTWFwLFxuICAgICAgICAgICAgICAgICAgYXR0cnMgYXMgdW5rbm93biBhcyBOYW1lZEF0dHJNYXApIGFzIFRlbnNvcjxSPjtcblxuICByZXR1cm4gcmVzaGFwZShyZXMsICR4LnNoYXBlKTtcbn1cblxuZXhwb3J0IGNvbnN0IGJhdGNoTm9ybSA9IC8qIEBfX1BVUkVfXyAqLyBvcCh7YmF0Y2hOb3JtX30pO1xuIl19