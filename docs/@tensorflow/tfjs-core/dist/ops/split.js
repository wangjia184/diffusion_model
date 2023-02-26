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
import { SplitV } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import { op } from './operation';
/**
 * Splits a `tf.Tensor` into sub tensors.
 *
 * If `numOrSizeSplits` is a number, splits `x` along dimension `axis`
 * into `numOrSizeSplits` smaller tensors.
 * Requires that `numOrSizeSplits` evenly divides `x.shape[axis]`.
 *
 * If `numOrSizeSplits` is a number array, splits `x` into
 * `numOrSizeSplits.length` pieces. The shape of the `i`-th piece has the
 * same size as `x` except along dimension `axis` where the size is
 * `numOrSizeSplits[i]`.
 *
 * ```js
 * const x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
 * const [a, b] = tf.split(x, 2, 1);
 * a.print();
 * b.print();
 *
 * const [c, d, e] = tf.split(x, [1, 2, 1], 1);
 * c.print();
 * d.print();
 * e.print();
 * ```
 *
 * @param x The input tensor to split.
 * @param numOrSizeSplits Either an integer indicating the number of
 * splits along the axis or an array of integers containing the sizes of
 * each output tensor along the axis. If a number then it must evenly divide
 * `x.shape[axis]`; otherwise the sum of sizes must match `x.shape[axis]`.
 * Can contain one -1 indicating that dimension is to be inferred.
 * @param axis The dimension along which to split. Defaults to 0 (the first
 * dim).
 *
 * @doc {heading: 'Tensors', subheading: 'Slicing and Joining'}
 */
function split_(x, numOrSizeSplits, axis = 0) {
    const $x = convertToTensor(x, 'x', 'split');
    const inputs = { x: $x };
    const attr = { numOrSizeSplits, axis };
    return ENGINE.runKernel(SplitV, inputs, attr);
}
export const split = /* @__PURE__ */ op({ split_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic3BsaXQuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9zcGxpdC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFDSCxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQ2pDLE9BQU8sRUFBQyxNQUFNLEVBQTRCLE1BQU0saUJBQWlCLENBQUM7QUFJbEUsT0FBTyxFQUFDLGVBQWUsRUFBQyxNQUFNLG9CQUFvQixDQUFDO0FBR25ELE9BQU8sRUFBQyxFQUFFLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFFL0I7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7R0FrQ0c7QUFDSCxTQUFTLE1BQU0sQ0FDWCxDQUFvQixFQUFFLGVBQWdDLEVBQUUsSUFBSSxHQUFHLENBQUM7SUFDbEUsTUFBTSxFQUFFLEdBQUcsZUFBZSxDQUFDLENBQUMsRUFBRSxHQUFHLEVBQUUsT0FBTyxDQUFDLENBQUM7SUFFNUMsTUFBTSxNQUFNLEdBQWlCLEVBQUMsQ0FBQyxFQUFFLEVBQUUsRUFBQyxDQUFDO0lBQ3JDLE1BQU0sSUFBSSxHQUFnQixFQUFDLGVBQWUsRUFBRSxJQUFJLEVBQUMsQ0FBQztJQUVsRCxPQUFPLE1BQU0sQ0FBQyxTQUFTLENBQ1osTUFBTSxFQUFFLE1BQW1DLEVBQzNDLElBQStCLENBQW1CLENBQUM7QUFDaEUsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLEtBQUssR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUMsTUFBTSxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi9lbmdpbmUnO1xuaW1wb3J0IHtTcGxpdFYsIFNwbGl0VkF0dHJzLCBTcGxpdFZJbnB1dHN9IGZyb20gJy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge05hbWVkQXR0ck1hcH0gZnJvbSAnLi4va2VybmVsX3JlZ2lzdHJ5JztcbmltcG9ydCB7VGVuc29yfSBmcm9tICcuLi90ZW5zb3InO1xuaW1wb3J0IHtOYW1lZFRlbnNvck1hcH0gZnJvbSAnLi4vdGVuc29yX3R5cGVzJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5cbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcblxuLyoqXG4gKiBTcGxpdHMgYSBgdGYuVGVuc29yYCBpbnRvIHN1YiB0ZW5zb3JzLlxuICpcbiAqIElmIGBudW1PclNpemVTcGxpdHNgIGlzIGEgbnVtYmVyLCBzcGxpdHMgYHhgIGFsb25nIGRpbWVuc2lvbiBgYXhpc2BcbiAqIGludG8gYG51bU9yU2l6ZVNwbGl0c2Agc21hbGxlciB0ZW5zb3JzLlxuICogUmVxdWlyZXMgdGhhdCBgbnVtT3JTaXplU3BsaXRzYCBldmVubHkgZGl2aWRlcyBgeC5zaGFwZVtheGlzXWAuXG4gKlxuICogSWYgYG51bU9yU2l6ZVNwbGl0c2AgaXMgYSBudW1iZXIgYXJyYXksIHNwbGl0cyBgeGAgaW50b1xuICogYG51bU9yU2l6ZVNwbGl0cy5sZW5ndGhgIHBpZWNlcy4gVGhlIHNoYXBlIG9mIHRoZSBgaWAtdGggcGllY2UgaGFzIHRoZVxuICogc2FtZSBzaXplIGFzIGB4YCBleGNlcHQgYWxvbmcgZGltZW5zaW9uIGBheGlzYCB3aGVyZSB0aGUgc2l6ZSBpc1xuICogYG51bU9yU2l6ZVNwbGl0c1tpXWAuXG4gKlxuICogYGBganNcbiAqIGNvbnN0IHggPSB0Zi50ZW5zb3IyZChbMSwgMiwgMywgNCwgNSwgNiwgNywgOF0sIFsyLCA0XSk7XG4gKiBjb25zdCBbYSwgYl0gPSB0Zi5zcGxpdCh4LCAyLCAxKTtcbiAqIGEucHJpbnQoKTtcbiAqIGIucHJpbnQoKTtcbiAqXG4gKiBjb25zdCBbYywgZCwgZV0gPSB0Zi5zcGxpdCh4LCBbMSwgMiwgMV0sIDEpO1xuICogYy5wcmludCgpO1xuICogZC5wcmludCgpO1xuICogZS5wcmludCgpO1xuICogYGBgXG4gKlxuICogQHBhcmFtIHggVGhlIGlucHV0IHRlbnNvciB0byBzcGxpdC5cbiAqIEBwYXJhbSBudW1PclNpemVTcGxpdHMgRWl0aGVyIGFuIGludGVnZXIgaW5kaWNhdGluZyB0aGUgbnVtYmVyIG9mXG4gKiBzcGxpdHMgYWxvbmcgdGhlIGF4aXMgb3IgYW4gYXJyYXkgb2YgaW50ZWdlcnMgY29udGFpbmluZyB0aGUgc2l6ZXMgb2ZcbiAqIGVhY2ggb3V0cHV0IHRlbnNvciBhbG9uZyB0aGUgYXhpcy4gSWYgYSBudW1iZXIgdGhlbiBpdCBtdXN0IGV2ZW5seSBkaXZpZGVcbiAqIGB4LnNoYXBlW2F4aXNdYDsgb3RoZXJ3aXNlIHRoZSBzdW0gb2Ygc2l6ZXMgbXVzdCBtYXRjaCBgeC5zaGFwZVtheGlzXWAuXG4gKiBDYW4gY29udGFpbiBvbmUgLTEgaW5kaWNhdGluZyB0aGF0IGRpbWVuc2lvbiBpcyB0byBiZSBpbmZlcnJlZC5cbiAqIEBwYXJhbSBheGlzIFRoZSBkaW1lbnNpb24gYWxvbmcgd2hpY2ggdG8gc3BsaXQuIERlZmF1bHRzIHRvIDAgKHRoZSBmaXJzdFxuICogZGltKS5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnVGVuc29ycycsIHN1YmhlYWRpbmc6ICdTbGljaW5nIGFuZCBKb2luaW5nJ31cbiAqL1xuZnVuY3Rpb24gc3BsaXRfPFQgZXh0ZW5kcyBUZW5zb3I+KFxuICAgIHg6IFRlbnNvcnxUZW5zb3JMaWtlLCBudW1PclNpemVTcGxpdHM6IG51bWJlcltdfG51bWJlciwgYXhpcyA9IDApOiBUW10ge1xuICBjb25zdCAkeCA9IGNvbnZlcnRUb1RlbnNvcih4LCAneCcsICdzcGxpdCcpO1xuXG4gIGNvbnN0IGlucHV0czogU3BsaXRWSW5wdXRzID0ge3g6ICR4fTtcbiAgY29uc3QgYXR0cjogU3BsaXRWQXR0cnMgPSB7bnVtT3JTaXplU3BsaXRzLCBheGlzfTtcblxuICByZXR1cm4gRU5HSU5FLnJ1bktlcm5lbChcbiAgICAgICAgICAgICBTcGxpdFYsIGlucHV0cyBhcyB1bmtub3duIGFzIE5hbWVkVGVuc29yTWFwLFxuICAgICAgICAgICAgIGF0dHIgYXMgdW5rbm93biBhcyBOYW1lZEF0dHJNYXApIGFzIHVua25vd24gYXMgVFtdO1xufVxuXG5leHBvcnQgY29uc3Qgc3BsaXQgPSAvKiBAX19QVVJFX18gKi8gb3Aoe3NwbGl0X30pO1xuIl19