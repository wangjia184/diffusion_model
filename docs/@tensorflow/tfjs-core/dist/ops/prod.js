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
import { Prod } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import { cast } from './cast';
import { op } from './operation';
/**
 * Computes the product of elements across dimensions of a `tf.Tensor`.
 *
 * Reduces the input along the dimensions given in `axes`. Unless `keepDims`
 * is true, the rank of the `tf.Tensor` is reduced by 1 for each entry in
 * `axes`. If `keepDims` is true, the reduced dimensions are retained with
 * length 1. If `axes` has no entries, all dimensions are reduced, and a
 * `tf.Tensor` with a single element is returned.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3]);
 *
 * x.prod().print();  // or tf.prod(x)
 * ```
 *
 * ```js
 * const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 *
 * const axis = 1;
 * x.prod(axis).print();  // or tf.prod(x, axis)
 * ```
 *
 * @param x The input tensor to compute the product over. If the dtype is `bool`
 *   it will be converted to `int32` and the output dtype will be `int32`.
 * @param axis The dimension(s) to reduce. By default it reduces
 *     all dimensions.
 * @param keepDims If true, retains reduced dimensions with size 1.
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
function prod_(x, axis = null, keepDims = false) {
    let $x = convertToTensor(x, 'x', 'prod');
    if ($x.dtype === 'bool') {
        // bool is not an allowed type for the underlying kernel.
        $x = cast($x, 'int32');
    }
    const inputs = { x: $x };
    const attrs = { axis, keepDims };
    return ENGINE.runKernel(Prod, inputs, attrs);
}
export const prod = /* @__PURE__ */ op({ prod_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicHJvZC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL3Byb2QudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMsSUFBSSxFQUF3QixNQUFNLGlCQUFpQixDQUFDO0FBSTVELE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUduRCxPQUFPLEVBQUMsSUFBSSxFQUFDLE1BQU0sUUFBUSxDQUFDO0FBQzVCLE9BQU8sRUFBQyxFQUFFLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFFL0I7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBNkJHO0FBQ0gsU0FBUyxLQUFLLENBQ1YsQ0FBb0IsRUFBRSxPQUF3QixJQUFJLEVBQUUsUUFBUSxHQUFHLEtBQUs7SUFDdEUsSUFBSSxFQUFFLEdBQUcsZUFBZSxDQUFDLENBQUMsRUFBRSxHQUFHLEVBQUUsTUFBTSxDQUFDLENBQUM7SUFFekMsSUFBSSxFQUFFLENBQUMsS0FBSyxLQUFLLE1BQU0sRUFBRTtRQUN2Qix5REFBeUQ7UUFDekQsRUFBRSxHQUFHLElBQUksQ0FBQyxFQUFFLEVBQUUsT0FBTyxDQUFDLENBQUM7S0FDeEI7SUFFRCxNQUFNLE1BQU0sR0FBZSxFQUFDLENBQUMsRUFBRSxFQUFFLEVBQUMsQ0FBQztJQUNuQyxNQUFNLEtBQUssR0FBYyxFQUFDLElBQUksRUFBRSxRQUFRLEVBQUMsQ0FBQztJQUUxQyxPQUFPLE1BQU0sQ0FBQyxTQUFTLENBQ25CLElBQUksRUFBRSxNQUFtQyxFQUN6QyxLQUFnQyxDQUFDLENBQUM7QUFDeEMsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLElBQUksR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUMsS0FBSyxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtFTkdJTkV9IGZyb20gJy4uL2VuZ2luZSc7XG5pbXBvcnQge1Byb2QsIFByb2RBdHRycywgUHJvZElucHV0c30gZnJvbSAnLi4va2VybmVsX25hbWVzJztcbmltcG9ydCB7TmFtZWRBdHRyTWFwfSBmcm9tICcuLi9rZXJuZWxfcmVnaXN0cnknO1xuaW1wb3J0IHtUZW5zb3J9IGZyb20gJy4uL3RlbnNvcic7XG5pbXBvcnQge05hbWVkVGVuc29yTWFwfSBmcm9tICcuLi90ZW5zb3JfdHlwZXMnO1xuaW1wb3J0IHtjb252ZXJ0VG9UZW5zb3J9IGZyb20gJy4uL3RlbnNvcl91dGlsX2Vudic7XG5pbXBvcnQge1RlbnNvckxpa2V9IGZyb20gJy4uL3R5cGVzJztcblxuaW1wb3J0IHtjYXN0fSBmcm9tICcuL2Nhc3QnO1xuaW1wb3J0IHtvcH0gZnJvbSAnLi9vcGVyYXRpb24nO1xuXG4vKipcbiAqIENvbXB1dGVzIHRoZSBwcm9kdWN0IG9mIGVsZW1lbnRzIGFjcm9zcyBkaW1lbnNpb25zIG9mIGEgYHRmLlRlbnNvcmAuXG4gKlxuICogUmVkdWNlcyB0aGUgaW5wdXQgYWxvbmcgdGhlIGRpbWVuc2lvbnMgZ2l2ZW4gaW4gYGF4ZXNgLiBVbmxlc3MgYGtlZXBEaW1zYFxuICogaXMgdHJ1ZSwgdGhlIHJhbmsgb2YgdGhlIGB0Zi5UZW5zb3JgIGlzIHJlZHVjZWQgYnkgMSBmb3IgZWFjaCBlbnRyeSBpblxuICogYGF4ZXNgLiBJZiBga2VlcERpbXNgIGlzIHRydWUsIHRoZSByZWR1Y2VkIGRpbWVuc2lvbnMgYXJlIHJldGFpbmVkIHdpdGhcbiAqIGxlbmd0aCAxLiBJZiBgYXhlc2AgaGFzIG5vIGVudHJpZXMsIGFsbCBkaW1lbnNpb25zIGFyZSByZWR1Y2VkLCBhbmQgYVxuICogYHRmLlRlbnNvcmAgd2l0aCBhIHNpbmdsZSBlbGVtZW50IGlzIHJldHVybmVkLlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCB4ID0gdGYudGVuc29yMWQoWzEsIDIsIDNdKTtcbiAqXG4gKiB4LnByb2QoKS5wcmludCgpOyAgLy8gb3IgdGYucHJvZCh4KVxuICogYGBgXG4gKlxuICogYGBganNcbiAqIGNvbnN0IHggPSB0Zi50ZW5zb3IyZChbMSwgMiwgMywgNF0sIFsyLCAyXSk7XG4gKlxuICogY29uc3QgYXhpcyA9IDE7XG4gKiB4LnByb2QoYXhpcykucHJpbnQoKTsgIC8vIG9yIHRmLnByb2QoeCwgYXhpcylcbiAqIGBgYFxuICpcbiAqIEBwYXJhbSB4IFRoZSBpbnB1dCB0ZW5zb3IgdG8gY29tcHV0ZSB0aGUgcHJvZHVjdCBvdmVyLiBJZiB0aGUgZHR5cGUgaXMgYGJvb2xgXG4gKiAgIGl0IHdpbGwgYmUgY29udmVydGVkIHRvIGBpbnQzMmAgYW5kIHRoZSBvdXRwdXQgZHR5cGUgd2lsbCBiZSBgaW50MzJgLlxuICogQHBhcmFtIGF4aXMgVGhlIGRpbWVuc2lvbihzKSB0byByZWR1Y2UuIEJ5IGRlZmF1bHQgaXQgcmVkdWNlc1xuICogICAgIGFsbCBkaW1lbnNpb25zLlxuICogQHBhcmFtIGtlZXBEaW1zIElmIHRydWUsIHJldGFpbnMgcmVkdWNlZCBkaW1lbnNpb25zIHdpdGggc2l6ZSAxLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdPcGVyYXRpb25zJywgc3ViaGVhZGluZzogJ1JlZHVjdGlvbid9XG4gKi9cbmZ1bmN0aW9uIHByb2RfPFQgZXh0ZW5kcyBUZW5zb3I+KFxuICAgIHg6IFRlbnNvcnxUZW5zb3JMaWtlLCBheGlzOiBudW1iZXJ8bnVtYmVyW10gPSBudWxsLCBrZWVwRGltcyA9IGZhbHNlKTogVCB7XG4gIGxldCAkeCA9IGNvbnZlcnRUb1RlbnNvcih4LCAneCcsICdwcm9kJyk7XG5cbiAgaWYgKCR4LmR0eXBlID09PSAnYm9vbCcpIHtcbiAgICAvLyBib29sIGlzIG5vdCBhbiBhbGxvd2VkIHR5cGUgZm9yIHRoZSB1bmRlcmx5aW5nIGtlcm5lbC5cbiAgICAkeCA9IGNhc3QoJHgsICdpbnQzMicpO1xuICB9XG5cbiAgY29uc3QgaW5wdXRzOiBQcm9kSW5wdXRzID0ge3g6ICR4fTtcbiAgY29uc3QgYXR0cnM6IFByb2RBdHRycyA9IHtheGlzLCBrZWVwRGltc307XG5cbiAgcmV0dXJuIEVOR0lORS5ydW5LZXJuZWwoXG4gICAgICBQcm9kLCBpbnB1dHMgYXMgdW5rbm93biBhcyBOYW1lZFRlbnNvck1hcCxcbiAgICAgIGF0dHJzIGFzIHVua25vd24gYXMgTmFtZWRBdHRyTWFwKTtcbn1cblxuZXhwb3J0IGNvbnN0IHByb2QgPSAvKiBAX19QVVJFX18gKi8gb3Aoe3Byb2RffSk7XG4iXX0=