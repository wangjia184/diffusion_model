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
import { makeTypesMatch } from '../tensor_util';
import { convertToTensor } from '../tensor_util_env';
import { div } from './div';
import { equal } from './equal';
import { op } from './operation';
import { where } from './where';
import { zerosLike } from './zeros_like';
/**
 * Divides two `tf.Tensor`s element-wise, A / B. Supports broadcasting. Return 0
 * if denominator is 0.
 *
 *
 * ```js
 * const a = tf.tensor1d([1, 4, 9, 16]);
 * const b = tf.tensor1d([1, 2, 3, 4]);
 * const c = tf.tensor1d([0, 0, 0, 0]);
 *
 * a.divNoNan(b).print();  // or tf.divNoNan(a, b)
 * a.divNoNan(c).print();  // or tf.divNoNan(a, c)
 * ```
 *
 * ```js
 * // Broadcast div a with b.
 * const a = tf.tensor1d([2, 4, 6, 8]);
 * const b = tf.scalar(2);
 * const c = tf.scalar(0);
 *
 * a.divNoNan(b).print();  // or tf.divNoNan(a, b)
 * a.divNoNan(c).print();  // or tf.divNoNan(a, c)
 * ```
 *
 * @param a The first tensor as the numerator.
 * @param b The second tensor as the denominator. Must have the same dtype as
 * `a`.
 *
 * @doc {heading: 'Operations', subheading: 'Arithmetic'}
 */
function divNoNan_(a, b) {
    // TODO: Make this into its own kernel.
    let $a = convertToTensor(a, 'a', 'div');
    let $b = convertToTensor(b, 'b', 'div');
    [$a, $b] = makeTypesMatch($a, $b);
    const divResult = div($a, $b);
    const zeros = zerosLike(divResult);
    const bEqualsZero = equal($b, zeros);
    return where(bEqualsZero, zeros, divResult);
}
export const divNoNan = /* @__PURE__ */ op({ divNoNan_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZGl2X25vX25hbi5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL2Rpdl9ub19uYW4udHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBR0gsT0FBTyxFQUFDLGNBQWMsRUFBQyxNQUFNLGdCQUFnQixDQUFDO0FBQzlDLE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUduRCxPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sT0FBTyxDQUFDO0FBQzFCLE9BQU8sRUFBQyxLQUFLLEVBQUMsTUFBTSxTQUFTLENBQUM7QUFDOUIsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUMvQixPQUFPLEVBQUMsS0FBSyxFQUFDLE1BQU0sU0FBUyxDQUFDO0FBQzlCLE9BQU8sRUFBQyxTQUFTLEVBQUMsTUFBTSxjQUFjLENBQUM7QUFFdkM7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBNkJHO0FBQ0gsU0FBUyxTQUFTLENBQ2QsQ0FBb0IsRUFBRSxDQUFvQjtJQUM1Qyx1Q0FBdUM7SUFDdkMsSUFBSSxFQUFFLEdBQUcsZUFBZSxDQUFDLENBQUMsRUFBRSxHQUFHLEVBQUUsS0FBSyxDQUFDLENBQUM7SUFDeEMsSUFBSSxFQUFFLEdBQUcsZUFBZSxDQUFDLENBQUMsRUFBRSxHQUFHLEVBQUUsS0FBSyxDQUFDLENBQUM7SUFDeEMsQ0FBQyxFQUFFLEVBQUUsRUFBRSxDQUFDLEdBQUcsY0FBYyxDQUFDLEVBQUUsRUFBRSxFQUFFLENBQUMsQ0FBQztJQUVsQyxNQUFNLFNBQVMsR0FBRyxHQUFHLENBQUMsRUFBRSxFQUFFLEVBQUUsQ0FBQyxDQUFDO0lBQzlCLE1BQU0sS0FBSyxHQUFHLFNBQVMsQ0FBQyxTQUFTLENBQUMsQ0FBQztJQUNuQyxNQUFNLFdBQVcsR0FBRyxLQUFLLENBQUMsRUFBRSxFQUFFLEtBQUssQ0FBQyxDQUFDO0lBQ3JDLE9BQU8sS0FBSyxDQUFDLFdBQVcsRUFBRSxLQUFLLEVBQUUsU0FBUyxDQUFNLENBQUM7QUFDbkQsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLFFBQVEsR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUMsU0FBUyxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtUZW5zb3J9IGZyb20gJy4uL3RlbnNvcic7XG5pbXBvcnQge21ha2VUeXBlc01hdGNofSBmcm9tICcuLi90ZW5zb3JfdXRpbCc7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7VGVuc29yTGlrZX0gZnJvbSAnLi4vdHlwZXMnO1xuXG5pbXBvcnQge2Rpdn0gZnJvbSAnLi9kaXYnO1xuaW1wb3J0IHtlcXVhbH0gZnJvbSAnLi9lcXVhbCc7XG5pbXBvcnQge29wfSBmcm9tICcuL29wZXJhdGlvbic7XG5pbXBvcnQge3doZXJlfSBmcm9tICcuL3doZXJlJztcbmltcG9ydCB7emVyb3NMaWtlfSBmcm9tICcuL3plcm9zX2xpa2UnO1xuXG4vKipcbiAqIERpdmlkZXMgdHdvIGB0Zi5UZW5zb3JgcyBlbGVtZW50LXdpc2UsIEEgLyBCLiBTdXBwb3J0cyBicm9hZGNhc3RpbmcuIFJldHVybiAwXG4gKiBpZiBkZW5vbWluYXRvciBpcyAwLlxuICpcbiAqXG4gKiBgYGBqc1xuICogY29uc3QgYSA9IHRmLnRlbnNvcjFkKFsxLCA0LCA5LCAxNl0pO1xuICogY29uc3QgYiA9IHRmLnRlbnNvcjFkKFsxLCAyLCAzLCA0XSk7XG4gKiBjb25zdCBjID0gdGYudGVuc29yMWQoWzAsIDAsIDAsIDBdKTtcbiAqXG4gKiBhLmRpdk5vTmFuKGIpLnByaW50KCk7ICAvLyBvciB0Zi5kaXZOb05hbihhLCBiKVxuICogYS5kaXZOb05hbihjKS5wcmludCgpOyAgLy8gb3IgdGYuZGl2Tm9OYW4oYSwgYylcbiAqIGBgYFxuICpcbiAqIGBgYGpzXG4gKiAvLyBCcm9hZGNhc3QgZGl2IGEgd2l0aCBiLlxuICogY29uc3QgYSA9IHRmLnRlbnNvcjFkKFsyLCA0LCA2LCA4XSk7XG4gKiBjb25zdCBiID0gdGYuc2NhbGFyKDIpO1xuICogY29uc3QgYyA9IHRmLnNjYWxhcigwKTtcbiAqXG4gKiBhLmRpdk5vTmFuKGIpLnByaW50KCk7ICAvLyBvciB0Zi5kaXZOb05hbihhLCBiKVxuICogYS5kaXZOb05hbihjKS5wcmludCgpOyAgLy8gb3IgdGYuZGl2Tm9OYW4oYSwgYylcbiAqIGBgYFxuICpcbiAqIEBwYXJhbSBhIFRoZSBmaXJzdCB0ZW5zb3IgYXMgdGhlIG51bWVyYXRvci5cbiAqIEBwYXJhbSBiIFRoZSBzZWNvbmQgdGVuc29yIGFzIHRoZSBkZW5vbWluYXRvci4gTXVzdCBoYXZlIHRoZSBzYW1lIGR0eXBlIGFzXG4gKiBgYWAuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ09wZXJhdGlvbnMnLCBzdWJoZWFkaW5nOiAnQXJpdGhtZXRpYyd9XG4gKi9cbmZ1bmN0aW9uIGRpdk5vTmFuXzxUIGV4dGVuZHMgVGVuc29yPihcbiAgICBhOiBUZW5zb3J8VGVuc29yTGlrZSwgYjogVGVuc29yfFRlbnNvckxpa2UpOiBUIHtcbiAgLy8gVE9ETzogTWFrZSB0aGlzIGludG8gaXRzIG93biBrZXJuZWwuXG4gIGxldCAkYSA9IGNvbnZlcnRUb1RlbnNvcihhLCAnYScsICdkaXYnKTtcbiAgbGV0ICRiID0gY29udmVydFRvVGVuc29yKGIsICdiJywgJ2RpdicpO1xuICBbJGEsICRiXSA9IG1ha2VUeXBlc01hdGNoKCRhLCAkYik7XG5cbiAgY29uc3QgZGl2UmVzdWx0ID0gZGl2KCRhLCAkYik7XG4gIGNvbnN0IHplcm9zID0gemVyb3NMaWtlKGRpdlJlc3VsdCk7XG4gIGNvbnN0IGJFcXVhbHNaZXJvID0gZXF1YWwoJGIsIHplcm9zKTtcbiAgcmV0dXJuIHdoZXJlKGJFcXVhbHNaZXJvLCB6ZXJvcywgZGl2UmVzdWx0KSBhcyBUO1xufVxuXG5leHBvcnQgY29uc3QgZGl2Tm9OYW4gPSAvKiBAX19QVVJFX18gKi8gb3Aoe2Rpdk5vTmFuX30pO1xuIl19