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
import { ENGINE } from '../engine';
import { Sum } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import { cast } from './cast';
import { op } from './operation';
/**
 * Computes the sum of elements across dimensions of a `tf.Tensor`.
 *
 * Reduces the input along the dimensions given in `axes`. Unless `keepDims`
 * is true, the rank of the `tf.Tensor` is reduced by 1 for each entry in
 * `axes`. If `keepDims` is true, the reduced dimensions are retained with
 * length 1. If axes has no entries, all dimensions are reduced, and a
 * `tf.Tensor` with a single element is returned.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3]);
 *
 * x.sum().print();  // or tf.sum(x)
 * ```
 *
 * ```js
 * const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 *
 * const axis = 1;
 * x.sum(axis).print();  // or tf.sum(x, axis)
 * ```
 *
 * @param x The input tensor to compute the sum over. If the dtype is `bool`
 *   it will be converted to `int32` and the output dtype will be `int32`.
 * @param axis The dimension(s) to reduce. By default it reduces
 *     all dimensions.
 * @param keepDims If true, retains reduced dimensions with size 1.
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
function sum_(x, axis = null, keepDims = false) {
    let $x = convertToTensor(x, 'x', 'sum');
    if ($x.dtype === 'bool') {
        $x = cast($x, 'int32');
    }
    const inputs = { x: $x };
    const attrs = { axis, keepDims };
    return ENGINE.runKernel(Sum, inputs, attrs);
}
export const sum = /* @__PURE__ */ op({ sum_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic3VtLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvc3VtLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUNILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDakMsT0FBTyxFQUFDLEdBQUcsRUFBc0IsTUFBTSxpQkFBaUIsQ0FBQztBQUl6RCxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFHbkQsT0FBTyxFQUFDLElBQUksRUFBQyxNQUFNLFFBQVEsQ0FBQztBQUM1QixPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBRS9COzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQTZCRztBQUNILFNBQVMsSUFBSSxDQUNULENBQW9CLEVBQUUsT0FBd0IsSUFBSSxFQUFFLFFBQVEsR0FBRyxLQUFLO0lBQ3RFLElBQUksRUFBRSxHQUFHLGVBQWUsQ0FBQyxDQUFDLEVBQUUsR0FBRyxFQUFFLEtBQUssQ0FBQyxDQUFDO0lBQ3hDLElBQUksRUFBRSxDQUFDLEtBQUssS0FBSyxNQUFNLEVBQUU7UUFDdkIsRUFBRSxHQUFHLElBQUksQ0FBQyxFQUFFLEVBQUUsT0FBTyxDQUFDLENBQUM7S0FDeEI7SUFFRCxNQUFNLE1BQU0sR0FBYyxFQUFDLENBQUMsRUFBRSxFQUFFLEVBQUMsQ0FBQztJQUNsQyxNQUFNLEtBQUssR0FBYSxFQUFDLElBQUksRUFBRSxRQUFRLEVBQUMsQ0FBQztJQUV6QyxPQUFPLE1BQU0sQ0FBQyxTQUFTLENBQ25CLEdBQUcsRUFBRSxNQUFtQyxFQUN4QyxLQUFnQyxDQUFDLENBQUM7QUFDeEMsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLEdBQUcsR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUMsSUFBSSxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi9lbmdpbmUnO1xuaW1wb3J0IHtTdW0sIFN1bUF0dHJzLCBTdW1JbnB1dHN9IGZyb20gJy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge05hbWVkQXR0ck1hcH0gZnJvbSAnLi4va2VybmVsX3JlZ2lzdHJ5JztcbmltcG9ydCB7VGVuc29yfSBmcm9tICcuLi90ZW5zb3InO1xuaW1wb3J0IHtOYW1lZFRlbnNvck1hcH0gZnJvbSAnLi4vdGVuc29yX3R5cGVzJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5cbmltcG9ydCB7Y2FzdH0gZnJvbSAnLi9jYXN0JztcbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcblxuLyoqXG4gKiBDb21wdXRlcyB0aGUgc3VtIG9mIGVsZW1lbnRzIGFjcm9zcyBkaW1lbnNpb25zIG9mIGEgYHRmLlRlbnNvcmAuXG4gKlxuICogUmVkdWNlcyB0aGUgaW5wdXQgYWxvbmcgdGhlIGRpbWVuc2lvbnMgZ2l2ZW4gaW4gYGF4ZXNgLiBVbmxlc3MgYGtlZXBEaW1zYFxuICogaXMgdHJ1ZSwgdGhlIHJhbmsgb2YgdGhlIGB0Zi5UZW5zb3JgIGlzIHJlZHVjZWQgYnkgMSBmb3IgZWFjaCBlbnRyeSBpblxuICogYGF4ZXNgLiBJZiBga2VlcERpbXNgIGlzIHRydWUsIHRoZSByZWR1Y2VkIGRpbWVuc2lvbnMgYXJlIHJldGFpbmVkIHdpdGhcbiAqIGxlbmd0aCAxLiBJZiBheGVzIGhhcyBubyBlbnRyaWVzLCBhbGwgZGltZW5zaW9ucyBhcmUgcmVkdWNlZCwgYW5kIGFcbiAqIGB0Zi5UZW5zb3JgIHdpdGggYSBzaW5nbGUgZWxlbWVudCBpcyByZXR1cm5lZC5cbiAqXG4gKiBgYGBqc1xuICogY29uc3QgeCA9IHRmLnRlbnNvcjFkKFsxLCAyLCAzXSk7XG4gKlxuICogeC5zdW0oKS5wcmludCgpOyAgLy8gb3IgdGYuc3VtKHgpXG4gKiBgYGBcbiAqXG4gKiBgYGBqc1xuICogY29uc3QgeCA9IHRmLnRlbnNvcjJkKFsxLCAyLCAzLCA0XSwgWzIsIDJdKTtcbiAqXG4gKiBjb25zdCBheGlzID0gMTtcbiAqIHguc3VtKGF4aXMpLnByaW50KCk7ICAvLyBvciB0Zi5zdW0oeCwgYXhpcylcbiAqIGBgYFxuICpcbiAqIEBwYXJhbSB4IFRoZSBpbnB1dCB0ZW5zb3IgdG8gY29tcHV0ZSB0aGUgc3VtIG92ZXIuIElmIHRoZSBkdHlwZSBpcyBgYm9vbGBcbiAqICAgaXQgd2lsbCBiZSBjb252ZXJ0ZWQgdG8gYGludDMyYCBhbmQgdGhlIG91dHB1dCBkdHlwZSB3aWxsIGJlIGBpbnQzMmAuXG4gKiBAcGFyYW0gYXhpcyBUaGUgZGltZW5zaW9uKHMpIHRvIHJlZHVjZS4gQnkgZGVmYXVsdCBpdCByZWR1Y2VzXG4gKiAgICAgYWxsIGRpbWVuc2lvbnMuXG4gKiBAcGFyYW0ga2VlcERpbXMgSWYgdHJ1ZSwgcmV0YWlucyByZWR1Y2VkIGRpbWVuc2lvbnMgd2l0aCBzaXplIDEuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ09wZXJhdGlvbnMnLCBzdWJoZWFkaW5nOiAnUmVkdWN0aW9uJ31cbiAqL1xuZnVuY3Rpb24gc3VtXzxUIGV4dGVuZHMgVGVuc29yPihcbiAgICB4OiBUZW5zb3J8VGVuc29yTGlrZSwgYXhpczogbnVtYmVyfG51bWJlcltdID0gbnVsbCwga2VlcERpbXMgPSBmYWxzZSk6IFQge1xuICBsZXQgJHggPSBjb252ZXJ0VG9UZW5zb3IoeCwgJ3gnLCAnc3VtJyk7XG4gIGlmICgkeC5kdHlwZSA9PT0gJ2Jvb2wnKSB7XG4gICAgJHggPSBjYXN0KCR4LCAnaW50MzInKTtcbiAgfVxuXG4gIGNvbnN0IGlucHV0czogU3VtSW5wdXRzID0ge3g6ICR4fTtcbiAgY29uc3QgYXR0cnM6IFN1bUF0dHJzID0ge2F4aXMsIGtlZXBEaW1zfTtcblxuICByZXR1cm4gRU5HSU5FLnJ1bktlcm5lbChcbiAgICAgIFN1bSwgaW5wdXRzIGFzIHVua25vd24gYXMgTmFtZWRUZW5zb3JNYXAsXG4gICAgICBhdHRycyBhcyB1bmtub3duIGFzIE5hbWVkQXR0ck1hcCk7XG59XG5cbmV4cG9ydCBjb25zdCBzdW0gPSAvKiBAX19QVVJFX18gKi8gb3Aoe3N1bV99KTtcbiJdfQ==