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
import { Minimum } from '../kernel_names';
import { makeTypesMatch } from '../tensor_util';
import { convertToTensor } from '../tensor_util_env';
import { assertAndGetBroadcastShape } from './broadcast_util';
import { cast } from './cast';
import { op } from './operation';
/**
 * Returns the min of a and b (`a < b ? a : b`) element-wise.
 * Supports broadcasting.
 *
 * We also expose `minimumStrict` which has the same signature as this op and
 * asserts that `a` and `b` are the same shape (does not broadcast).
 *
 * ```js
 * const a = tf.tensor1d([1, 4, 3, 16]);
 * const b = tf.tensor1d([1, 2, 9, 4]);
 *
 * a.minimum(b).print();  // or tf.minimum(a, b)
 * ```
 *
 * ```js
 * // Broadcast minimum a with b.
 * const a = tf.tensor1d([2, 4, 6, 8]);
 * const b = tf.scalar(5);
 *
 * a.minimum(b).print();  // or tf.minimum(a, b)
 * ```
 *
 * @param a The first tensor.
 * @param b The second tensor. Must have the same type as `a`.
 *
 * @doc {heading: 'Operations', subheading: 'Arithmetic'}
 */
function minimum_(a, b) {
    let $a = convertToTensor(a, 'a', 'minimum');
    let $b = convertToTensor(b, 'b', 'minimum');
    [$a, $b] = makeTypesMatch($a, $b);
    if ($a.dtype === 'bool') {
        $a = cast($a, 'int32');
        $b = cast($b, 'int32');
    }
    assertAndGetBroadcastShape($a.shape, $b.shape);
    const inputs = { a: $a, b: $b };
    return ENGINE.runKernel(Minimum, inputs);
}
export const minimum = /* @__PURE__ */ op({ minimum_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibWluaW11bS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL21pbmltdW0udHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMsT0FBTyxFQUFnQixNQUFNLGlCQUFpQixDQUFDO0FBR3ZELE9BQU8sRUFBQyxjQUFjLEVBQUMsTUFBTSxnQkFBZ0IsQ0FBQztBQUM5QyxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFHbkQsT0FBTyxFQUFDLDBCQUEwQixFQUFDLE1BQU0sa0JBQWtCLENBQUM7QUFDNUQsT0FBTyxFQUFDLElBQUksRUFBQyxNQUFNLFFBQVEsQ0FBQztBQUM1QixPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBRS9COzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQTBCRztBQUNILFNBQVMsUUFBUSxDQUNiLENBQW9CLEVBQUUsQ0FBb0I7SUFDNUMsSUFBSSxFQUFFLEdBQUcsZUFBZSxDQUFDLENBQUMsRUFBRSxHQUFHLEVBQUUsU0FBUyxDQUFDLENBQUM7SUFDNUMsSUFBSSxFQUFFLEdBQUcsZUFBZSxDQUFDLENBQUMsRUFBRSxHQUFHLEVBQUUsU0FBUyxDQUFDLENBQUM7SUFDNUMsQ0FBQyxFQUFFLEVBQUUsRUFBRSxDQUFDLEdBQUcsY0FBYyxDQUFDLEVBQUUsRUFBRSxFQUFFLENBQUMsQ0FBQztJQUVsQyxJQUFJLEVBQUUsQ0FBQyxLQUFLLEtBQUssTUFBTSxFQUFFO1FBQ3ZCLEVBQUUsR0FBRyxJQUFJLENBQUMsRUFBRSxFQUFFLE9BQU8sQ0FBQyxDQUFDO1FBQ3ZCLEVBQUUsR0FBRyxJQUFJLENBQUMsRUFBRSxFQUFFLE9BQU8sQ0FBQyxDQUFDO0tBQ3hCO0lBRUQsMEJBQTBCLENBQUMsRUFBRSxDQUFDLEtBQUssRUFBRSxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUM7SUFFL0MsTUFBTSxNQUFNLEdBQWtCLEVBQUMsQ0FBQyxFQUFFLEVBQUUsRUFBRSxDQUFDLEVBQUUsRUFBRSxFQUFDLENBQUM7SUFFN0MsT0FBTyxNQUFNLENBQUMsU0FBUyxDQUFDLE9BQU8sRUFBRSxNQUFtQyxDQUFDLENBQUM7QUFDeEUsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLE9BQU8sR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUMsUUFBUSxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtFTkdJTkV9IGZyb20gJy4uL2VuZ2luZSc7XG5pbXBvcnQge01pbmltdW0sIE1pbmltdW1JbnB1dHN9IGZyb20gJy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7TmFtZWRUZW5zb3JNYXB9IGZyb20gJy4uL3RlbnNvcl90eXBlcyc7XG5pbXBvcnQge21ha2VUeXBlc01hdGNofSBmcm9tICcuLi90ZW5zb3JfdXRpbCc7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7VGVuc29yTGlrZX0gZnJvbSAnLi4vdHlwZXMnO1xuXG5pbXBvcnQge2Fzc2VydEFuZEdldEJyb2FkY2FzdFNoYXBlfSBmcm9tICcuL2Jyb2FkY2FzdF91dGlsJztcbmltcG9ydCB7Y2FzdH0gZnJvbSAnLi9jYXN0JztcbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcblxuLyoqXG4gKiBSZXR1cm5zIHRoZSBtaW4gb2YgYSBhbmQgYiAoYGEgPCBiID8gYSA6IGJgKSBlbGVtZW50LXdpc2UuXG4gKiBTdXBwb3J0cyBicm9hZGNhc3RpbmcuXG4gKlxuICogV2UgYWxzbyBleHBvc2UgYG1pbmltdW1TdHJpY3RgIHdoaWNoIGhhcyB0aGUgc2FtZSBzaWduYXR1cmUgYXMgdGhpcyBvcCBhbmRcbiAqIGFzc2VydHMgdGhhdCBgYWAgYW5kIGBiYCBhcmUgdGhlIHNhbWUgc2hhcGUgKGRvZXMgbm90IGJyb2FkY2FzdCkuXG4gKlxuICogYGBganNcbiAqIGNvbnN0IGEgPSB0Zi50ZW5zb3IxZChbMSwgNCwgMywgMTZdKTtcbiAqIGNvbnN0IGIgPSB0Zi50ZW5zb3IxZChbMSwgMiwgOSwgNF0pO1xuICpcbiAqIGEubWluaW11bShiKS5wcmludCgpOyAgLy8gb3IgdGYubWluaW11bShhLCBiKVxuICogYGBgXG4gKlxuICogYGBganNcbiAqIC8vIEJyb2FkY2FzdCBtaW5pbXVtIGEgd2l0aCBiLlxuICogY29uc3QgYSA9IHRmLnRlbnNvcjFkKFsyLCA0LCA2LCA4XSk7XG4gKiBjb25zdCBiID0gdGYuc2NhbGFyKDUpO1xuICpcbiAqIGEubWluaW11bShiKS5wcmludCgpOyAgLy8gb3IgdGYubWluaW11bShhLCBiKVxuICogYGBgXG4gKlxuICogQHBhcmFtIGEgVGhlIGZpcnN0IHRlbnNvci5cbiAqIEBwYXJhbSBiIFRoZSBzZWNvbmQgdGVuc29yLiBNdXN0IGhhdmUgdGhlIHNhbWUgdHlwZSBhcyBgYWAuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ09wZXJhdGlvbnMnLCBzdWJoZWFkaW5nOiAnQXJpdGhtZXRpYyd9XG4gKi9cbmZ1bmN0aW9uIG1pbmltdW1fPFQgZXh0ZW5kcyBUZW5zb3I+KFxuICAgIGE6IFRlbnNvcnxUZW5zb3JMaWtlLCBiOiBUZW5zb3J8VGVuc29yTGlrZSk6IFQge1xuICBsZXQgJGEgPSBjb252ZXJ0VG9UZW5zb3IoYSwgJ2EnLCAnbWluaW11bScpO1xuICBsZXQgJGIgPSBjb252ZXJ0VG9UZW5zb3IoYiwgJ2InLCAnbWluaW11bScpO1xuICBbJGEsICRiXSA9IG1ha2VUeXBlc01hdGNoKCRhLCAkYik7XG5cbiAgaWYgKCRhLmR0eXBlID09PSAnYm9vbCcpIHtcbiAgICAkYSA9IGNhc3QoJGEsICdpbnQzMicpO1xuICAgICRiID0gY2FzdCgkYiwgJ2ludDMyJyk7XG4gIH1cblxuICBhc3NlcnRBbmRHZXRCcm9hZGNhc3RTaGFwZSgkYS5zaGFwZSwgJGIuc2hhcGUpO1xuXG4gIGNvbnN0IGlucHV0czogTWluaW11bUlucHV0cyA9IHthOiAkYSwgYjogJGJ9O1xuXG4gIHJldHVybiBFTkdJTkUucnVuS2VybmVsKE1pbmltdW0sIGlucHV0cyBhcyB1bmtub3duIGFzIE5hbWVkVGVuc29yTWFwKTtcbn1cblxuZXhwb3J0IGNvbnN0IG1pbmltdW0gPSAvKiBAX19QVVJFX18gKi8gb3Aoe21pbmltdW1ffSk7XG4iXX0=