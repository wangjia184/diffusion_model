/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import { ENGINE } from '../engine';
import { Cumprod } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import { op } from './operation';
/**
 * Computes the cumulative product of a `tf.Tensor` along `axis`.
 *
 * ```js
 * const x = tf.tensor([1, 2, 3, 4]);
 * x.cumprod().print();
 * ```
 * ```js
 * const x = tf.tensor([[1, 2], [3, 4]]);
 * x.cumprod().print();
 * ```
 *
 * @param x The input tensor to cumulatively multiply.
 * @param axis The axis along which to multiply. Optional. Defaults to 0.
 * @param exclusive Whether to perform exclusive cumulative product. Optional.
 *     Defaults to false. If set to true then the product of each tensor entry
 *     does not include its own value, but only the values previous to it
 *     along the specified axis.
 * @param reverse Whether to multiply in the opposite direction. Optional.
 *     Defaults to false.
 *
 * @doc {heading: 'Operations', subheading: 'Scan'}
 */
function cumprod_(x, axis = 0, exclusive = false, reverse = false) {
    const $x = convertToTensor(x, 'x', 'cumprod');
    const inputs = { x: $x };
    const attrs = { axis, exclusive, reverse };
    return ENGINE.runKernel(Cumprod, inputs, attrs);
}
export const cumprod = /* @__PURE__ */ op({ cumprod_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY3VtcHJvZC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL2N1bXByb2QudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFFLE1BQU0sRUFBRSxNQUFNLFdBQVcsQ0FBQztBQUNuQyxPQUFPLEVBQUUsT0FBTyxFQUErQixNQUFNLGlCQUFpQixDQUFDO0FBSXZFLE9BQU8sRUFBRSxlQUFlLEVBQUUsTUFBTSxvQkFBb0IsQ0FBQztBQUdyRCxPQUFPLEVBQUUsRUFBRSxFQUFFLE1BQU0sYUFBYSxDQUFDO0FBRWpDOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBc0JHO0FBQ0gsU0FBUyxRQUFRLENBQ2YsQ0FBc0IsRUFDdEIsSUFBSSxHQUFHLENBQUMsRUFDUixTQUFTLEdBQUcsS0FBSyxFQUNqQixPQUFPLEdBQUcsS0FBSztJQUVmLE1BQU0sRUFBRSxHQUFHLGVBQWUsQ0FBQyxDQUFDLEVBQUUsR0FBRyxFQUFFLFNBQVMsQ0FBQyxDQUFDO0lBRTlDLE1BQU0sTUFBTSxHQUFrQixFQUFFLENBQUMsRUFBRSxFQUFFLEVBQUUsQ0FBQztJQUN4QyxNQUFNLEtBQUssR0FBaUIsRUFBRSxJQUFJLEVBQUUsU0FBUyxFQUFFLE9BQU8sRUFBRSxDQUFDO0lBRXpELE9BQU8sTUFBTSxDQUFDLFNBQVMsQ0FDckIsT0FBTyxFQUNQLE1BQW1DLEVBQ25DLEtBQWdDLENBQ2pDLENBQUM7QUFDSixDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sT0FBTyxHQUFHLGVBQWUsQ0FBQyxFQUFFLENBQUMsRUFBRSxRQUFRLEVBQUUsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjIgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSAnTGljZW5zZScpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gJ0FTIElTJyBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7IEVOR0lORSB9IGZyb20gJy4uL2VuZ2luZSc7XG5pbXBvcnQgeyBDdW1wcm9kLCBDdW1wcm9kQXR0cnMsIEN1bXByb2RJbnB1dHMgfSBmcm9tICcuLi9rZXJuZWxfbmFtZXMnO1xuaW1wb3J0IHsgTmFtZWRBdHRyTWFwIH0gZnJvbSAnLi4va2VybmVsX3JlZ2lzdHJ5JztcbmltcG9ydCB7IFRlbnNvciB9IGZyb20gJy4uL3RlbnNvcic7XG5pbXBvcnQgeyBOYW1lZFRlbnNvck1hcCB9IGZyb20gJy4uL3RlbnNvcl90eXBlcyc7XG5pbXBvcnQgeyBjb252ZXJ0VG9UZW5zb3IgfSBmcm9tICcuLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHsgVGVuc29yTGlrZSB9IGZyb20gJy4uL3R5cGVzJztcblxuaW1wb3J0IHsgb3AgfSBmcm9tICcuL29wZXJhdGlvbic7XG5cbi8qKlxuICogQ29tcHV0ZXMgdGhlIGN1bXVsYXRpdmUgcHJvZHVjdCBvZiBhIGB0Zi5UZW5zb3JgIGFsb25nIGBheGlzYC5cbiAqXG4gKiBgYGBqc1xuICogY29uc3QgeCA9IHRmLnRlbnNvcihbMSwgMiwgMywgNF0pO1xuICogeC5jdW1wcm9kKCkucHJpbnQoKTtcbiAqIGBgYFxuICogYGBganNcbiAqIGNvbnN0IHggPSB0Zi50ZW5zb3IoW1sxLCAyXSwgWzMsIDRdXSk7XG4gKiB4LmN1bXByb2QoKS5wcmludCgpO1xuICogYGBgXG4gKlxuICogQHBhcmFtIHggVGhlIGlucHV0IHRlbnNvciB0byBjdW11bGF0aXZlbHkgbXVsdGlwbHkuXG4gKiBAcGFyYW0gYXhpcyBUaGUgYXhpcyBhbG9uZyB3aGljaCB0byBtdWx0aXBseS4gT3B0aW9uYWwuIERlZmF1bHRzIHRvIDAuXG4gKiBAcGFyYW0gZXhjbHVzaXZlIFdoZXRoZXIgdG8gcGVyZm9ybSBleGNsdXNpdmUgY3VtdWxhdGl2ZSBwcm9kdWN0LiBPcHRpb25hbC5cbiAqICAgICBEZWZhdWx0cyB0byBmYWxzZS4gSWYgc2V0IHRvIHRydWUgdGhlbiB0aGUgcHJvZHVjdCBvZiBlYWNoIHRlbnNvciBlbnRyeVxuICogICAgIGRvZXMgbm90IGluY2x1ZGUgaXRzIG93biB2YWx1ZSwgYnV0IG9ubHkgdGhlIHZhbHVlcyBwcmV2aW91cyB0byBpdFxuICogICAgIGFsb25nIHRoZSBzcGVjaWZpZWQgYXhpcy5cbiAqIEBwYXJhbSByZXZlcnNlIFdoZXRoZXIgdG8gbXVsdGlwbHkgaW4gdGhlIG9wcG9zaXRlIGRpcmVjdGlvbi4gT3B0aW9uYWwuXG4gKiAgICAgRGVmYXVsdHMgdG8gZmFsc2UuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ09wZXJhdGlvbnMnLCBzdWJoZWFkaW5nOiAnU2Nhbid9XG4gKi9cbmZ1bmN0aW9uIGN1bXByb2RfPFQgZXh0ZW5kcyBUZW5zb3I+KFxuICB4OiBUZW5zb3IgfCBUZW5zb3JMaWtlLFxuICBheGlzID0gMCxcbiAgZXhjbHVzaXZlID0gZmFsc2UsXG4gIHJldmVyc2UgPSBmYWxzZVxuKTogVCB7XG4gIGNvbnN0ICR4ID0gY29udmVydFRvVGVuc29yKHgsICd4JywgJ2N1bXByb2QnKTtcblxuICBjb25zdCBpbnB1dHM6IEN1bXByb2RJbnB1dHMgPSB7IHg6ICR4IH07XG4gIGNvbnN0IGF0dHJzOiBDdW1wcm9kQXR0cnMgPSB7IGF4aXMsIGV4Y2x1c2l2ZSwgcmV2ZXJzZSB9O1xuXG4gIHJldHVybiBFTkdJTkUucnVuS2VybmVsKFxuICAgIEN1bXByb2QsXG4gICAgaW5wdXRzIGFzIHVua25vd24gYXMgTmFtZWRUZW5zb3JNYXAsXG4gICAgYXR0cnMgYXMgdW5rbm93biBhcyBOYW1lZEF0dHJNYXBcbiAgKTtcbn1cblxuZXhwb3J0IGNvbnN0IGN1bXByb2QgPSAvKiBAX19QVVJFX18gKi8gb3AoeyBjdW1wcm9kXyB9KTtcbiJdfQ==