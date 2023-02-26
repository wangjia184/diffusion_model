/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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
import { op } from './operation';
import { randomNormal } from './random_normal';
/**
 * Creates a `tf.Tensor` with values sampled from a normal distribution.
 *
 * The generated values will have mean 0 and standard deviation 1.
 *
 * ```js
 * tf.randomStandardNormal([2, 2]).print();
 * ```
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param dtype The data type of the output.
 * @param seed The seed for the random number generator.
 *
 * @doc {heading: 'Tensors', subheading: 'Random'}
 */
function randomStandardNormal_(shape, dtype, seed) {
    if (dtype != null && dtype === 'bool') {
        throw new Error(`Unsupported data type ${dtype}`);
    }
    return randomNormal(shape, 0, 1, dtype, seed);
}
export const randomStandardNormal = /* @__PURE__ */ op({ randomStandardNormal_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicmFuZG9tX3N0YW5kYXJkX25vcm1hbC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL3JhbmRvbV9zdGFuZGFyZF9ub3JtYWwudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBS0gsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUMvQixPQUFPLEVBQUMsWUFBWSxFQUFDLE1BQU0saUJBQWlCLENBQUM7QUFFN0M7Ozs7Ozs7Ozs7Ozs7O0dBY0c7QUFDSCxTQUFTLHFCQUFxQixDQUMxQixLQUFrQixFQUFFLEtBQXlCLEVBQUUsSUFBYTtJQUM5RCxJQUFJLEtBQUssSUFBSSxJQUFJLElBQUssS0FBa0IsS0FBSyxNQUFNLEVBQUU7UUFDbkQsTUFBTSxJQUFJLEtBQUssQ0FBQyx5QkFBeUIsS0FBSyxFQUFFLENBQUMsQ0FBQztLQUNuRDtJQUNELE9BQU8sWUFBWSxDQUFDLEtBQUssRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLEtBQUssRUFBRSxJQUFJLENBQUMsQ0FBQztBQUNoRCxDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sb0JBQW9CLEdBQUcsZUFBZSxDQUFDLEVBQUUsQ0FBQyxFQUFDLHFCQUFxQixFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIyIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtUZW5zb3J9IGZyb20gJy4uL3RlbnNvcic7XG5pbXBvcnQge0RhdGFUeXBlLCBSYW5rLCBTaGFwZU1hcH0gZnJvbSAnLi4vdHlwZXMnO1xuXG5pbXBvcnQge29wfSBmcm9tICcuL29wZXJhdGlvbic7XG5pbXBvcnQge3JhbmRvbU5vcm1hbH0gZnJvbSAnLi9yYW5kb21fbm9ybWFsJztcblxuLyoqXG4gKiBDcmVhdGVzIGEgYHRmLlRlbnNvcmAgd2l0aCB2YWx1ZXMgc2FtcGxlZCBmcm9tIGEgbm9ybWFsIGRpc3RyaWJ1dGlvbi5cbiAqXG4gKiBUaGUgZ2VuZXJhdGVkIHZhbHVlcyB3aWxsIGhhdmUgbWVhbiAwIGFuZCBzdGFuZGFyZCBkZXZpYXRpb24gMS5cbiAqXG4gKiBgYGBqc1xuICogdGYucmFuZG9tU3RhbmRhcmROb3JtYWwoWzIsIDJdKS5wcmludCgpO1xuICogYGBgXG4gKlxuICogQHBhcmFtIHNoYXBlIEFuIGFycmF5IG9mIGludGVnZXJzIGRlZmluaW5nIHRoZSBvdXRwdXQgdGVuc29yIHNoYXBlLlxuICogQHBhcmFtIGR0eXBlIFRoZSBkYXRhIHR5cGUgb2YgdGhlIG91dHB1dC5cbiAqIEBwYXJhbSBzZWVkIFRoZSBzZWVkIGZvciB0aGUgcmFuZG9tIG51bWJlciBnZW5lcmF0b3IuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ1RlbnNvcnMnLCBzdWJoZWFkaW5nOiAnUmFuZG9tJ31cbiAqL1xuZnVuY3Rpb24gcmFuZG9tU3RhbmRhcmROb3JtYWxfPFIgZXh0ZW5kcyBSYW5rPihcbiAgICBzaGFwZTogU2hhcGVNYXBbUl0sIGR0eXBlPzogJ2Zsb2F0MzInfCdpbnQzMicsIHNlZWQ/OiBudW1iZXIpOiBUZW5zb3I8Uj4ge1xuICBpZiAoZHR5cGUgIT0gbnVsbCAmJiAoZHR5cGUgYXMgRGF0YVR5cGUpID09PSAnYm9vbCcpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoYFVuc3VwcG9ydGVkIGRhdGEgdHlwZSAke2R0eXBlfWApO1xuICB9XG4gIHJldHVybiByYW5kb21Ob3JtYWwoc2hhcGUsIDAsIDEsIGR0eXBlLCBzZWVkKTtcbn1cblxuZXhwb3J0IGNvbnN0IHJhbmRvbVN0YW5kYXJkTm9ybWFsID0gLyogQF9fUFVSRV9fICovIG9wKHtyYW5kb21TdGFuZGFyZE5vcm1hbF99KTtcbiJdfQ==