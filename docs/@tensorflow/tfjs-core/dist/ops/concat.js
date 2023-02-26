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
import { Concat } from '../kernel_names';
import { convertToTensorArray } from '../tensor_util_env';
import { assert } from '../util';
import { clone } from './clone';
import { op } from './operation';
/**
 * Concatenates a list of `tf.Tensor`s along a given axis.
 *
 * The tensors ranks and types must match, and their sizes must match in all
 * dimensions except `axis`.
 *
 * Also available are stricter rank-specific methods that assert that
 * `tensors` are of the given rank:
 *   - `tf.concat1d`
 *   - `tf.concat2d`
 *   - `tf.concat3d`
 *   - `tf.concat4d`
 *
 * Except `tf.concat1d` (which does not have axis param), all methods have
 * same signature as this method.
 *
 * ```js
 * const a = tf.tensor1d([1, 2]);
 * const b = tf.tensor1d([3, 4]);
 * a.concat(b).print();  // or a.concat(b)
 * ```
 *
 * ```js
 * const a = tf.tensor1d([1, 2]);
 * const b = tf.tensor1d([3, 4]);
 * const c = tf.tensor1d([5, 6]);
 * tf.concat([a, b, c]).print();
 * ```
 *
 * ```js
 * const a = tf.tensor2d([[1, 2], [10, 20]]);
 * const b = tf.tensor2d([[3, 4], [30, 40]]);
 * const axis = 1;
 * tf.concat([a, b], axis).print();
 * ```
 * @param tensors A list of tensors to concatenate.
 * @param axis The axis to concatenate along. Defaults to 0 (the first dim).
 *
 * @doc {heading: 'Tensors', subheading: 'Slicing and Joining'}
 */
function concat_(tensors, axis = 0) {
    assert(tensors.length >= 1, () => 'Pass at least one tensor to concat');
    const $tensors = convertToTensorArray(tensors, 'tensors', 'concat', 'string_or_numeric');
    if ($tensors[0].dtype === 'complex64') {
        $tensors.forEach(tensor => {
            if (tensor.dtype !== 'complex64') {
                throw new Error(`Cannot concatenate complex64 tensors with a tensor
          with dtype ${tensor.dtype}. `);
            }
        });
    }
    if ($tensors.length === 1) {
        return clone($tensors[0]);
    }
    const inputs = $tensors;
    const attr = { axis };
    return ENGINE.runKernel(Concat, inputs, attr);
}
export const concat = /* @__PURE__ */ op({ concat_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY29uY2F0LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvY29uY2F0LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUNILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDakMsT0FBTyxFQUFDLE1BQU0sRUFBNEIsTUFBTSxpQkFBaUIsQ0FBQztBQUlsRSxPQUFPLEVBQUMsb0JBQW9CLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUV4RCxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sU0FBUyxDQUFDO0FBRS9CLE9BQU8sRUFBQyxLQUFLLEVBQUMsTUFBTSxTQUFTLENBQUM7QUFDOUIsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUUvQjs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBdUNHO0FBQ0gsU0FBUyxPQUFPLENBQW1CLE9BQTRCLEVBQUUsSUFBSSxHQUFHLENBQUM7SUFDdkUsTUFBTSxDQUFDLE9BQU8sQ0FBQyxNQUFNLElBQUksQ0FBQyxFQUFFLEdBQUcsRUFBRSxDQUFDLG9DQUFvQyxDQUFDLENBQUM7SUFFeEUsTUFBTSxRQUFRLEdBQ1Ysb0JBQW9CLENBQUMsT0FBTyxFQUFFLFNBQVMsRUFBRSxRQUFRLEVBQUUsbUJBQW1CLENBQUMsQ0FBQztJQUU1RSxJQUFJLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQyxLQUFLLEtBQUssV0FBVyxFQUFFO1FBQ3JDLFFBQVEsQ0FBQyxPQUFPLENBQUMsTUFBTSxDQUFDLEVBQUU7WUFDeEIsSUFBSSxNQUFNLENBQUMsS0FBSyxLQUFLLFdBQVcsRUFBRTtnQkFDaEMsTUFBTSxJQUFJLEtBQUssQ0FBQzt1QkFDRCxNQUFNLENBQUMsS0FBSyxJQUFJLENBQUMsQ0FBQzthQUNsQztRQUNILENBQUMsQ0FBQyxDQUFDO0tBQ0o7SUFFRCxJQUFJLFFBQVEsQ0FBQyxNQUFNLEtBQUssQ0FBQyxFQUFFO1FBQ3pCLE9BQU8sS0FBSyxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0tBQzNCO0lBRUQsTUFBTSxNQUFNLEdBQWlCLFFBQVEsQ0FBQztJQUN0QyxNQUFNLElBQUksR0FBZ0IsRUFBQyxJQUFJLEVBQUMsQ0FBQztJQUVqQyxPQUFPLE1BQU0sQ0FBQyxTQUFTLENBQ25CLE1BQU0sRUFBRSxNQUFtQyxFQUMzQyxJQUErQixDQUFDLENBQUM7QUFDdkMsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLE1BQU0sR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUMsT0FBTyxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi9lbmdpbmUnO1xuaW1wb3J0IHtDb25jYXQsIENvbmNhdEF0dHJzLCBDb25jYXRJbnB1dHN9IGZyb20gJy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge05hbWVkQXR0ck1hcH0gZnJvbSAnLi4va2VybmVsX3JlZ2lzdHJ5JztcbmltcG9ydCB7VGVuc29yfSBmcm9tICcuLi90ZW5zb3InO1xuaW1wb3J0IHtOYW1lZFRlbnNvck1hcH0gZnJvbSAnLi4vdGVuc29yX3R5cGVzJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yQXJyYXl9IGZyb20gJy4uL3RlbnNvcl91dGlsX2Vudic7XG5pbXBvcnQge1RlbnNvckxpa2V9IGZyb20gJy4uL3R5cGVzJztcbmltcG9ydCB7YXNzZXJ0fSBmcm9tICcuLi91dGlsJztcblxuaW1wb3J0IHtjbG9uZX0gZnJvbSAnLi9jbG9uZSc7XG5pbXBvcnQge29wfSBmcm9tICcuL29wZXJhdGlvbic7XG5cbi8qKlxuICogQ29uY2F0ZW5hdGVzIGEgbGlzdCBvZiBgdGYuVGVuc29yYHMgYWxvbmcgYSBnaXZlbiBheGlzLlxuICpcbiAqIFRoZSB0ZW5zb3JzIHJhbmtzIGFuZCB0eXBlcyBtdXN0IG1hdGNoLCBhbmQgdGhlaXIgc2l6ZXMgbXVzdCBtYXRjaCBpbiBhbGxcbiAqIGRpbWVuc2lvbnMgZXhjZXB0IGBheGlzYC5cbiAqXG4gKiBBbHNvIGF2YWlsYWJsZSBhcmUgc3RyaWN0ZXIgcmFuay1zcGVjaWZpYyBtZXRob2RzIHRoYXQgYXNzZXJ0IHRoYXRcbiAqIGB0ZW5zb3JzYCBhcmUgb2YgdGhlIGdpdmVuIHJhbms6XG4gKiAgIC0gYHRmLmNvbmNhdDFkYFxuICogICAtIGB0Zi5jb25jYXQyZGBcbiAqICAgLSBgdGYuY29uY2F0M2RgXG4gKiAgIC0gYHRmLmNvbmNhdDRkYFxuICpcbiAqIEV4Y2VwdCBgdGYuY29uY2F0MWRgICh3aGljaCBkb2VzIG5vdCBoYXZlIGF4aXMgcGFyYW0pLCBhbGwgbWV0aG9kcyBoYXZlXG4gKiBzYW1lIHNpZ25hdHVyZSBhcyB0aGlzIG1ldGhvZC5cbiAqXG4gKiBgYGBqc1xuICogY29uc3QgYSA9IHRmLnRlbnNvcjFkKFsxLCAyXSk7XG4gKiBjb25zdCBiID0gdGYudGVuc29yMWQoWzMsIDRdKTtcbiAqIGEuY29uY2F0KGIpLnByaW50KCk7ICAvLyBvciBhLmNvbmNhdChiKVxuICogYGBgXG4gKlxuICogYGBganNcbiAqIGNvbnN0IGEgPSB0Zi50ZW5zb3IxZChbMSwgMl0pO1xuICogY29uc3QgYiA9IHRmLnRlbnNvcjFkKFszLCA0XSk7XG4gKiBjb25zdCBjID0gdGYudGVuc29yMWQoWzUsIDZdKTtcbiAqIHRmLmNvbmNhdChbYSwgYiwgY10pLnByaW50KCk7XG4gKiBgYGBcbiAqXG4gKiBgYGBqc1xuICogY29uc3QgYSA9IHRmLnRlbnNvcjJkKFtbMSwgMl0sIFsxMCwgMjBdXSk7XG4gKiBjb25zdCBiID0gdGYudGVuc29yMmQoW1szLCA0XSwgWzMwLCA0MF1dKTtcbiAqIGNvbnN0IGF4aXMgPSAxO1xuICogdGYuY29uY2F0KFthLCBiXSwgYXhpcykucHJpbnQoKTtcbiAqIGBgYFxuICogQHBhcmFtIHRlbnNvcnMgQSBsaXN0IG9mIHRlbnNvcnMgdG8gY29uY2F0ZW5hdGUuXG4gKiBAcGFyYW0gYXhpcyBUaGUgYXhpcyB0byBjb25jYXRlbmF0ZSBhbG9uZy4gRGVmYXVsdHMgdG8gMCAodGhlIGZpcnN0IGRpbSkuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ1RlbnNvcnMnLCBzdWJoZWFkaW5nOiAnU2xpY2luZyBhbmQgSm9pbmluZyd9XG4gKi9cbmZ1bmN0aW9uIGNvbmNhdF88VCBleHRlbmRzIFRlbnNvcj4odGVuc29yczogQXJyYXk8VHxUZW5zb3JMaWtlPiwgYXhpcyA9IDApOiBUIHtcbiAgYXNzZXJ0KHRlbnNvcnMubGVuZ3RoID49IDEsICgpID0+ICdQYXNzIGF0IGxlYXN0IG9uZSB0ZW5zb3IgdG8gY29uY2F0Jyk7XG5cbiAgY29uc3QgJHRlbnNvcnMgPVxuICAgICAgY29udmVydFRvVGVuc29yQXJyYXkodGVuc29ycywgJ3RlbnNvcnMnLCAnY29uY2F0JywgJ3N0cmluZ19vcl9udW1lcmljJyk7XG5cbiAgaWYgKCR0ZW5zb3JzWzBdLmR0eXBlID09PSAnY29tcGxleDY0Jykge1xuICAgICR0ZW5zb3JzLmZvckVhY2godGVuc29yID0+IHtcbiAgICAgIGlmICh0ZW5zb3IuZHR5cGUgIT09ICdjb21wbGV4NjQnKSB7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcihgQ2Fubm90IGNvbmNhdGVuYXRlIGNvbXBsZXg2NCB0ZW5zb3JzIHdpdGggYSB0ZW5zb3JcbiAgICAgICAgICB3aXRoIGR0eXBlICR7dGVuc29yLmR0eXBlfS4gYCk7XG4gICAgICB9XG4gICAgfSk7XG4gIH1cblxuICBpZiAoJHRlbnNvcnMubGVuZ3RoID09PSAxKSB7XG4gICAgcmV0dXJuIGNsb25lKCR0ZW5zb3JzWzBdKTtcbiAgfVxuXG4gIGNvbnN0IGlucHV0czogQ29uY2F0SW5wdXRzID0gJHRlbnNvcnM7XG4gIGNvbnN0IGF0dHI6IENvbmNhdEF0dHJzID0ge2F4aXN9O1xuXG4gIHJldHVybiBFTkdJTkUucnVuS2VybmVsKFxuICAgICAgQ29uY2F0LCBpbnB1dHMgYXMgdW5rbm93biBhcyBOYW1lZFRlbnNvck1hcCxcbiAgICAgIGF0dHIgYXMgdW5rbm93biBhcyBOYW1lZEF0dHJNYXApO1xufVxuXG5leHBvcnQgY29uc3QgY29uY2F0ID0gLyogQF9fUFVSRV9fICovIG9wKHtjb25jYXRffSk7XG4iXX0=