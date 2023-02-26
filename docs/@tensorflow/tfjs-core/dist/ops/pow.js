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
import { Pow } from '../kernel_names';
import { makeTypesMatch } from '../tensor_util';
import { convertToTensor } from '../tensor_util_env';
import { op } from './operation';
/**
 * Computes the power of one `tf.Tensor` to another. Supports broadcasting.
 *
 * Given a `tf.Tensor` x and a `tf.Tensor` y, this operation computes x^y for
 * corresponding elements in x and y. The result's dtype will be the upcasted
 * type of the `base` and `exp` dtypes.
 *
 * ```js
 * const a = tf.tensor([[2, 3], [4, 5]])
 * const b = tf.tensor([[1, 2], [3, 0]]).toInt();
 *
 * a.pow(b).print();  // or tf.pow(a, b)
 * ```
 *
 * ```js
 * const a = tf.tensor([[1, 2], [3, 4]])
 * const b = tf.tensor(2).toInt();
 *
 * a.pow(b).print();  // or tf.pow(a, b)
 * ```
 * We also expose `powStrict` which has the same signature as this op and
 * asserts that `base` and `exp` are the same shape (does not broadcast).
 *
 * @param base The base `tf.Tensor` to pow element-wise.
 * @param exp The exponent `tf.Tensor` to pow element-wise.
 *
 * @doc {heading: 'Operations', subheading: 'Arithmetic'}
 */
function pow_(base, exp) {
    let $base = convertToTensor(base, 'base', 'pow');
    let $exp = convertToTensor(exp, 'exp', 'pow');
    [$base, $exp] = makeTypesMatch($base, $exp);
    const inputs = { a: $base, b: $exp };
    return ENGINE.runKernel(Pow, inputs);
}
export const pow = /* @__PURE__ */ op({ pow_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicG93LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvcG93LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUNILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDakMsT0FBTyxFQUFDLEdBQUcsRUFBWSxNQUFNLGlCQUFpQixDQUFDO0FBRy9DLE9BQU8sRUFBQyxjQUFjLEVBQUMsTUFBTSxnQkFBZ0IsQ0FBQztBQUM5QyxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFHbkQsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUUvQjs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBMkJHO0FBQ0gsU0FBUyxJQUFJLENBQ1QsSUFBdUIsRUFBRSxHQUFzQjtJQUNqRCxJQUFJLEtBQUssR0FBRyxlQUFlLENBQUMsSUFBSSxFQUFFLE1BQU0sRUFBRSxLQUFLLENBQUMsQ0FBQztJQUNqRCxJQUFJLElBQUksR0FBRyxlQUFlLENBQUMsR0FBRyxFQUFFLEtBQUssRUFBRSxLQUFLLENBQUMsQ0FBQztJQUM5QyxDQUFDLEtBQUssRUFBRSxJQUFJLENBQUMsR0FBRyxjQUFjLENBQUMsS0FBSyxFQUFFLElBQUksQ0FBQyxDQUFDO0lBRTVDLE1BQU0sTUFBTSxHQUFjLEVBQUMsQ0FBQyxFQUFFLEtBQUssRUFBRSxDQUFDLEVBQUUsSUFBSSxFQUFDLENBQUM7SUFFOUMsT0FBTyxNQUFNLENBQUMsU0FBUyxDQUFDLEdBQUcsRUFBRSxNQUFtQyxDQUFDLENBQUM7QUFDcEUsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLEdBQUcsR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUMsSUFBSSxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi9lbmdpbmUnO1xuaW1wb3J0IHtQb3csIFBvd0lucHV0c30gZnJvbSAnLi4va2VybmVsX25hbWVzJztcbmltcG9ydCB7VGVuc29yfSBmcm9tICcuLi90ZW5zb3InO1xuaW1wb3J0IHtOYW1lZFRlbnNvck1hcH0gZnJvbSAnLi4vdGVuc29yX3R5cGVzJztcbmltcG9ydCB7bWFrZVR5cGVzTWF0Y2h9IGZyb20gJy4uL3RlbnNvcl91dGlsJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5cbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcblxuLyoqXG4gKiBDb21wdXRlcyB0aGUgcG93ZXIgb2Ygb25lIGB0Zi5UZW5zb3JgIHRvIGFub3RoZXIuIFN1cHBvcnRzIGJyb2FkY2FzdGluZy5cbiAqXG4gKiBHaXZlbiBhIGB0Zi5UZW5zb3JgIHggYW5kIGEgYHRmLlRlbnNvcmAgeSwgdGhpcyBvcGVyYXRpb24gY29tcHV0ZXMgeF55IGZvclxuICogY29ycmVzcG9uZGluZyBlbGVtZW50cyBpbiB4IGFuZCB5LiBUaGUgcmVzdWx0J3MgZHR5cGUgd2lsbCBiZSB0aGUgdXBjYXN0ZWRcbiAqIHR5cGUgb2YgdGhlIGBiYXNlYCBhbmQgYGV4cGAgZHR5cGVzLlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBhID0gdGYudGVuc29yKFtbMiwgM10sIFs0LCA1XV0pXG4gKiBjb25zdCBiID0gdGYudGVuc29yKFtbMSwgMl0sIFszLCAwXV0pLnRvSW50KCk7XG4gKlxuICogYS5wb3coYikucHJpbnQoKTsgIC8vIG9yIHRmLnBvdyhhLCBiKVxuICogYGBgXG4gKlxuICogYGBganNcbiAqIGNvbnN0IGEgPSB0Zi50ZW5zb3IoW1sxLCAyXSwgWzMsIDRdXSlcbiAqIGNvbnN0IGIgPSB0Zi50ZW5zb3IoMikudG9JbnQoKTtcbiAqXG4gKiBhLnBvdyhiKS5wcmludCgpOyAgLy8gb3IgdGYucG93KGEsIGIpXG4gKiBgYGBcbiAqIFdlIGFsc28gZXhwb3NlIGBwb3dTdHJpY3RgIHdoaWNoIGhhcyB0aGUgc2FtZSBzaWduYXR1cmUgYXMgdGhpcyBvcCBhbmRcbiAqIGFzc2VydHMgdGhhdCBgYmFzZWAgYW5kIGBleHBgIGFyZSB0aGUgc2FtZSBzaGFwZSAoZG9lcyBub3QgYnJvYWRjYXN0KS5cbiAqXG4gKiBAcGFyYW0gYmFzZSBUaGUgYmFzZSBgdGYuVGVuc29yYCB0byBwb3cgZWxlbWVudC13aXNlLlxuICogQHBhcmFtIGV4cCBUaGUgZXhwb25lbnQgYHRmLlRlbnNvcmAgdG8gcG93IGVsZW1lbnQtd2lzZS5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnT3BlcmF0aW9ucycsIHN1YmhlYWRpbmc6ICdBcml0aG1ldGljJ31cbiAqL1xuZnVuY3Rpb24gcG93XzxUIGV4dGVuZHMgVGVuc29yPihcbiAgICBiYXNlOiBUZW5zb3J8VGVuc29yTGlrZSwgZXhwOiBUZW5zb3J8VGVuc29yTGlrZSk6IFQge1xuICBsZXQgJGJhc2UgPSBjb252ZXJ0VG9UZW5zb3IoYmFzZSwgJ2Jhc2UnLCAncG93Jyk7XG4gIGxldCAkZXhwID0gY29udmVydFRvVGVuc29yKGV4cCwgJ2V4cCcsICdwb3cnKTtcbiAgWyRiYXNlLCAkZXhwXSA9IG1ha2VUeXBlc01hdGNoKCRiYXNlLCAkZXhwKTtcblxuICBjb25zdCBpbnB1dHM6IFBvd0lucHV0cyA9IHthOiAkYmFzZSwgYjogJGV4cH07XG5cbiAgcmV0dXJuIEVOR0lORS5ydW5LZXJuZWwoUG93LCBpbnB1dHMgYXMgdW5rbm93biBhcyBOYW1lZFRlbnNvck1hcCk7XG59XG5cbmV4cG9ydCBjb25zdCBwb3cgPSAvKiBAX19QVVJFX18gKi8gb3Aoe3Bvd199KTtcbiJdfQ==