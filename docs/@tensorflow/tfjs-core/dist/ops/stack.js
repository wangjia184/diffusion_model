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
import { Pack } from '../kernel_names';
import { convertToTensorArray } from '../tensor_util_env';
import * as util from '../util';
import { op } from './operation';
/**
 * Stacks a list of rank-`R` `tf.Tensor`s into one rank-`(R+1)` `tf.Tensor`.
 *
 * ```js
 * const a = tf.tensor1d([1, 2]);
 * const b = tf.tensor1d([3, 4]);
 * const c = tf.tensor1d([5, 6]);
 * tf.stack([a, b, c]).print();
 * ```
 *
 * @param tensors A list of tensor objects with the same shape and dtype.
 * @param axis The axis to stack along. Defaults to 0 (the first dim).
 *
 * @doc {heading: 'Tensors', subheading: 'Slicing and Joining'}
 */
function stack_(tensors, axis = 0) {
    const $tensors = convertToTensorArray(tensors, 'tensors', 'stack', 'string_or_numeric');
    util.assert($tensors.length >= 1, () => 'Pass at least one tensor to tf.stack');
    if ($tensors.length > 0) {
        util.assert(axis <= $tensors[0].rank, () => 'Axis must be <= rank of the tensor');
    }
    const inputs = $tensors;
    const attrs = { axis };
    return ENGINE.runKernel(Pack, inputs, attrs);
}
export const stack = /* @__PURE__ */ op({ stack_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic3RhY2suanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9zdGFjay50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQ2pDLE9BQU8sRUFBQyxJQUFJLEVBQXdCLE1BQU0saUJBQWlCLENBQUM7QUFJNUQsT0FBTyxFQUFDLG9CQUFvQixFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFFeEQsT0FBTyxLQUFLLElBQUksTUFBTSxTQUFTLENBQUM7QUFFaEMsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUUvQjs7Ozs7Ozs7Ozs7Ozs7R0FjRztBQUNILFNBQVMsTUFBTSxDQUNYLE9BQTRCLEVBQUUsSUFBSSxHQUFHLENBQUM7SUFDeEMsTUFBTSxRQUFRLEdBQ1Ysb0JBQW9CLENBQUMsT0FBTyxFQUFFLFNBQVMsRUFBRSxPQUFPLEVBQUUsbUJBQW1CLENBQUMsQ0FBQztJQUUzRSxJQUFJLENBQUMsTUFBTSxDQUNQLFFBQVEsQ0FBQyxNQUFNLElBQUksQ0FBQyxFQUFFLEdBQUcsRUFBRSxDQUFDLHNDQUFzQyxDQUFDLENBQUM7SUFFeEUsSUFBSSxRQUFRLENBQUMsTUFBTSxHQUFHLENBQUMsRUFBRTtRQUN2QixJQUFJLENBQUMsTUFBTSxDQUNQLElBQUksSUFBSSxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUMsSUFBSSxFQUFFLEdBQUcsRUFBRSxDQUFDLG9DQUFvQyxDQUFDLENBQUM7S0FDM0U7SUFFRCxNQUFNLE1BQU0sR0FBZSxRQUFRLENBQUM7SUFDcEMsTUFBTSxLQUFLLEdBQWMsRUFBQyxJQUFJLEVBQUMsQ0FBQztJQUVoQyxPQUFPLE1BQU0sQ0FBQyxTQUFTLENBQ25CLElBQUksRUFBRSxNQUFtQyxFQUN6QyxLQUFnQyxDQUFDLENBQUM7QUFDeEMsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLEtBQUssR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUMsTUFBTSxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtFTkdJTkV9IGZyb20gJy4uL2VuZ2luZSc7XG5pbXBvcnQge1BhY2ssIFBhY2tBdHRycywgUGFja0lucHV0c30gZnJvbSAnLi4va2VybmVsX25hbWVzJztcbmltcG9ydCB7TmFtZWRBdHRyTWFwfSBmcm9tICcuLi9rZXJuZWxfcmVnaXN0cnknO1xuaW1wb3J0IHtUZW5zb3J9IGZyb20gJy4uL3RlbnNvcic7XG5pbXBvcnQge05hbWVkVGVuc29yTWFwfSBmcm9tICcuLi90ZW5zb3JfdHlwZXMnO1xuaW1wb3J0IHtjb252ZXJ0VG9UZW5zb3JBcnJheX0gZnJvbSAnLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7VGVuc29yTGlrZX0gZnJvbSAnLi4vdHlwZXMnO1xuaW1wb3J0ICogYXMgdXRpbCBmcm9tICcuLi91dGlsJztcblxuaW1wb3J0IHtvcH0gZnJvbSAnLi9vcGVyYXRpb24nO1xuXG4vKipcbiAqIFN0YWNrcyBhIGxpc3Qgb2YgcmFuay1gUmAgYHRmLlRlbnNvcmBzIGludG8gb25lIHJhbmstYChSKzEpYCBgdGYuVGVuc29yYC5cbiAqXG4gKiBgYGBqc1xuICogY29uc3QgYSA9IHRmLnRlbnNvcjFkKFsxLCAyXSk7XG4gKiBjb25zdCBiID0gdGYudGVuc29yMWQoWzMsIDRdKTtcbiAqIGNvbnN0IGMgPSB0Zi50ZW5zb3IxZChbNSwgNl0pO1xuICogdGYuc3RhY2soW2EsIGIsIGNdKS5wcmludCgpO1xuICogYGBgXG4gKlxuICogQHBhcmFtIHRlbnNvcnMgQSBsaXN0IG9mIHRlbnNvciBvYmplY3RzIHdpdGggdGhlIHNhbWUgc2hhcGUgYW5kIGR0eXBlLlxuICogQHBhcmFtIGF4aXMgVGhlIGF4aXMgdG8gc3RhY2sgYWxvbmcuIERlZmF1bHRzIHRvIDAgKHRoZSBmaXJzdCBkaW0pLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdUZW5zb3JzJywgc3ViaGVhZGluZzogJ1NsaWNpbmcgYW5kIEpvaW5pbmcnfVxuICovXG5mdW5jdGlvbiBzdGFja188VCBleHRlbmRzIFRlbnNvcj4oXG4gICAgdGVuc29yczogQXJyYXk8VHxUZW5zb3JMaWtlPiwgYXhpcyA9IDApOiBUZW5zb3Ige1xuICBjb25zdCAkdGVuc29ycyA9XG4gICAgICBjb252ZXJ0VG9UZW5zb3JBcnJheSh0ZW5zb3JzLCAndGVuc29ycycsICdzdGFjaycsICdzdHJpbmdfb3JfbnVtZXJpYycpO1xuXG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgJHRlbnNvcnMubGVuZ3RoID49IDEsICgpID0+ICdQYXNzIGF0IGxlYXN0IG9uZSB0ZW5zb3IgdG8gdGYuc3RhY2snKTtcblxuICBpZiAoJHRlbnNvcnMubGVuZ3RoID4gMCkge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBheGlzIDw9ICR0ZW5zb3JzWzBdLnJhbmssICgpID0+ICdBeGlzIG11c3QgYmUgPD0gcmFuayBvZiB0aGUgdGVuc29yJyk7XG4gIH1cblxuICBjb25zdCBpbnB1dHM6IFBhY2tJbnB1dHMgPSAkdGVuc29ycztcbiAgY29uc3QgYXR0cnM6IFBhY2tBdHRycyA9IHtheGlzfTtcblxuICByZXR1cm4gRU5HSU5FLnJ1bktlcm5lbChcbiAgICAgIFBhY2ssIGlucHV0cyBhcyB1bmtub3duIGFzIE5hbWVkVGVuc29yTWFwLFxuICAgICAgYXR0cnMgYXMgdW5rbm93biBhcyBOYW1lZEF0dHJNYXApO1xufVxuXG5leHBvcnQgY29uc3Qgc3RhY2sgPSAvKiBAX19QVVJFX18gKi8gb3Aoe3N0YWNrX30pO1xuIl19