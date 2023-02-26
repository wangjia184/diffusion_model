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
import { PadV2 } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import { op } from './operation';
/**
 * Pads a `tf.Tensor` with a given value and paddings.
 *
 * This operation implements `CONSTANT` mode. For `REFLECT` and `SYMMETRIC`,
 * refer to `tf.mirrorPad`.
 *
 * Also available are stricter rank-specific methods with the same signature
 * as this method that assert that `paddings` is of given length.
 *   - `tf.pad1d`
 *   - `tf.pad2d`
 *   - `tf.pad3d`
 *   - `tf.pad4d`
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3, 4]);
 * x.pad([[1, 2]]).print();
 * ```
 * @param x The tensor to pad.
 * @param paddings An array of length `R` (the rank of the tensor), where
 * each element is a length-2 tuple of ints `[padBefore, padAfter]`,
 * specifying how much to pad along each dimension of the tensor.
 * @param constantValue The pad value to use. Defaults to 0.
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
function pad_(x, paddings, constantValue = 0) {
    const $x = convertToTensor(x, 'x', 'pad');
    if ($x.rank === 0) {
        throw new Error('pad(scalar) is not defined. Pass non-scalar to pad');
    }
    const attrs = { paddings, constantValue };
    const inputs = { x: $x };
    return ENGINE.runKernel(PadV2, inputs, attrs);
}
export const pad = /* @__PURE__ */ op({ pad_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicGFkLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvcGFkLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDakMsT0FBTyxFQUFDLEtBQUssRUFBMEIsTUFBTSxpQkFBaUIsQ0FBQztBQUkvRCxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFHbkQsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUUvQjs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBd0JHO0FBQ0gsU0FBUyxJQUFJLENBQ1QsQ0FBZSxFQUFFLFFBQWlDLEVBQUUsYUFBYSxHQUFHLENBQUM7SUFDdkUsTUFBTSxFQUFFLEdBQUcsZUFBZSxDQUFDLENBQUMsRUFBRSxHQUFHLEVBQUUsS0FBSyxDQUFDLENBQUM7SUFDMUMsSUFBSSxFQUFFLENBQUMsSUFBSSxLQUFLLENBQUMsRUFBRTtRQUNqQixNQUFNLElBQUksS0FBSyxDQUFDLG9EQUFvRCxDQUFDLENBQUM7S0FDdkU7SUFFRCxNQUFNLEtBQUssR0FBZSxFQUFDLFFBQVEsRUFBRSxhQUFhLEVBQUMsQ0FBQztJQUNwRCxNQUFNLE1BQU0sR0FBZ0IsRUFBQyxDQUFDLEVBQUUsRUFBRSxFQUFDLENBQUM7SUFDcEMsT0FBTyxNQUFNLENBQUMsU0FBUyxDQUNuQixLQUFLLEVBQUUsTUFBbUMsRUFDMUMsS0FBZ0MsQ0FBQyxDQUFDO0FBQ3hDLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxHQUFHLEdBQUcsZUFBZSxDQUFDLEVBQUUsQ0FBQyxFQUFDLElBQUksRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi9lbmdpbmUnO1xuaW1wb3J0IHtQYWRWMiwgUGFkVjJBdHRycywgUGFkVjJJbnB1dHN9IGZyb20gJy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge05hbWVkQXR0ck1hcH0gZnJvbSAnLi4va2VybmVsX3JlZ2lzdHJ5JztcbmltcG9ydCB7VGVuc29yfSBmcm9tICcuLi90ZW5zb3InO1xuaW1wb3J0IHtOYW1lZFRlbnNvck1hcH0gZnJvbSAnLi4vdGVuc29yX3R5cGVzJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5cbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcblxuLyoqXG4gKiBQYWRzIGEgYHRmLlRlbnNvcmAgd2l0aCBhIGdpdmVuIHZhbHVlIGFuZCBwYWRkaW5ncy5cbiAqXG4gKiBUaGlzIG9wZXJhdGlvbiBpbXBsZW1lbnRzIGBDT05TVEFOVGAgbW9kZS4gRm9yIGBSRUZMRUNUYCBhbmQgYFNZTU1FVFJJQ2AsXG4gKiByZWZlciB0byBgdGYubWlycm9yUGFkYC5cbiAqXG4gKiBBbHNvIGF2YWlsYWJsZSBhcmUgc3RyaWN0ZXIgcmFuay1zcGVjaWZpYyBtZXRob2RzIHdpdGggdGhlIHNhbWUgc2lnbmF0dXJlXG4gKiBhcyB0aGlzIG1ldGhvZCB0aGF0IGFzc2VydCB0aGF0IGBwYWRkaW5nc2AgaXMgb2YgZ2l2ZW4gbGVuZ3RoLlxuICogICAtIGB0Zi5wYWQxZGBcbiAqICAgLSBgdGYucGFkMmRgXG4gKiAgIC0gYHRmLnBhZDNkYFxuICogICAtIGB0Zi5wYWQ0ZGBcbiAqXG4gKiBgYGBqc1xuICogY29uc3QgeCA9IHRmLnRlbnNvcjFkKFsxLCAyLCAzLCA0XSk7XG4gKiB4LnBhZChbWzEsIDJdXSkucHJpbnQoKTtcbiAqIGBgYFxuICogQHBhcmFtIHggVGhlIHRlbnNvciB0byBwYWQuXG4gKiBAcGFyYW0gcGFkZGluZ3MgQW4gYXJyYXkgb2YgbGVuZ3RoIGBSYCAodGhlIHJhbmsgb2YgdGhlIHRlbnNvciksIHdoZXJlXG4gKiBlYWNoIGVsZW1lbnQgaXMgYSBsZW5ndGgtMiB0dXBsZSBvZiBpbnRzIGBbcGFkQmVmb3JlLCBwYWRBZnRlcl1gLFxuICogc3BlY2lmeWluZyBob3cgbXVjaCB0byBwYWQgYWxvbmcgZWFjaCBkaW1lbnNpb24gb2YgdGhlIHRlbnNvci5cbiAqIEBwYXJhbSBjb25zdGFudFZhbHVlIFRoZSBwYWQgdmFsdWUgdG8gdXNlLiBEZWZhdWx0cyB0byAwLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdUZW5zb3JzJywgc3ViaGVhZGluZzogJ1RyYW5zZm9ybWF0aW9ucyd9XG4gKi9cbmZ1bmN0aW9uIHBhZF88VCBleHRlbmRzIFRlbnNvcj4oXG4gICAgeDogVHxUZW5zb3JMaWtlLCBwYWRkaW5nczogQXJyYXk8W251bWJlciwgbnVtYmVyXT4sIGNvbnN0YW50VmFsdWUgPSAwKTogVCB7XG4gIGNvbnN0ICR4ID0gY29udmVydFRvVGVuc29yKHgsICd4JywgJ3BhZCcpO1xuICBpZiAoJHgucmFuayA9PT0gMCkge1xuICAgIHRocm93IG5ldyBFcnJvcigncGFkKHNjYWxhcikgaXMgbm90IGRlZmluZWQuIFBhc3Mgbm9uLXNjYWxhciB0byBwYWQnKTtcbiAgfVxuXG4gIGNvbnN0IGF0dHJzOiBQYWRWMkF0dHJzID0ge3BhZGRpbmdzLCBjb25zdGFudFZhbHVlfTtcbiAgY29uc3QgaW5wdXRzOiBQYWRWMklucHV0cyA9IHt4OiAkeH07XG4gIHJldHVybiBFTkdJTkUucnVuS2VybmVsKFxuICAgICAgUGFkVjIsIGlucHV0cyBhcyB1bmtub3duIGFzIE5hbWVkVGVuc29yTWFwLFxuICAgICAgYXR0cnMgYXMgdW5rbm93biBhcyBOYW1lZEF0dHJNYXApO1xufVxuXG5leHBvcnQgY29uc3QgcGFkID0gLyogQF9fUFVSRV9fICovIG9wKHtwYWRffSk7XG4iXX0=