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
import { AddN } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import * as util from '../util';
import { op } from './operation';
/**
 * Adds a list of `tf.Tensor`s element-wise, each with the same shape and dtype.
 *
 * ```js
 * const a = tf.tensor1d([1, 2]);
 * const b = tf.tensor1d([3, 4]);
 * const c = tf.tensor1d([5, 6]);
 *
 * tf.addN([a, b, c]).print();
 * ```
 * @param tensors A list of tensors with the same shape and dtype.
 * @doc {heading: 'Operations', subheading: 'Arithmetic'}
 */
function addN_(tensors) {
    util.assert(Array.isArray(tensors), () => 'The argument passed to tf.addN() must be a list of tensors');
    util.assert(tensors.length >= 1, () => `Must pass at least one tensor to tf.addN(), but got ` +
        `${tensors.length}`);
    const $tensors = tensors.map((t, i) => convertToTensor(t, `tensors${i}`, 'addN'));
    const firstTensor = $tensors[0];
    $tensors.forEach(t => {
        if (t.dtype !== firstTensor.dtype) {
            throw new Error('All tensors passed to tf.addN() must have the same dtype');
        }
    });
    $tensors.forEach(t => {
        if (!util.arraysEqual(t.shape, firstTensor.shape)) {
            throw new Error('All tensors passed to tf.addN() must have the same shape');
        }
    });
    const inputs = $tensors;
    return ENGINE.runKernel(AddN, inputs);
}
export const addN = /* @__PURE__ */ op({ addN_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYWRkX24uanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9hZGRfbi50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFDSCxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQ2pDLE9BQU8sRUFBQyxJQUFJLEVBQWEsTUFBTSxpQkFBaUIsQ0FBQztBQUdqRCxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFFbkQsT0FBTyxLQUFLLElBQUksTUFBTSxTQUFTLENBQUM7QUFFaEMsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUUvQjs7Ozs7Ozs7Ozs7O0dBWUc7QUFDSCxTQUFTLEtBQUssQ0FBbUIsT0FBNEI7SUFDM0QsSUFBSSxDQUFDLE1BQU0sQ0FDUCxLQUFLLENBQUMsT0FBTyxDQUFDLE9BQU8sQ0FBQyxFQUN0QixHQUFHLEVBQUUsQ0FBQyw0REFBNEQsQ0FBQyxDQUFDO0lBQ3hFLElBQUksQ0FBQyxNQUFNLENBQ1AsT0FBTyxDQUFDLE1BQU0sSUFBSSxDQUFDLEVBQ25CLEdBQUcsRUFBRSxDQUFDLHNEQUFzRDtRQUN4RCxHQUFHLE9BQU8sQ0FBQyxNQUFNLEVBQUUsQ0FBQyxDQUFDO0lBRTdCLE1BQU0sUUFBUSxHQUNWLE9BQU8sQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxlQUFlLENBQUMsQ0FBQyxFQUFFLFVBQVUsQ0FBQyxFQUFFLEVBQUUsTUFBTSxDQUFDLENBQUMsQ0FBQztJQUVyRSxNQUFNLFdBQVcsR0FBRyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDaEMsUUFBUSxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsRUFBRTtRQUNuQixJQUFJLENBQUMsQ0FBQyxLQUFLLEtBQUssV0FBVyxDQUFDLEtBQUssRUFBRTtZQUNqQyxNQUFNLElBQUksS0FBSyxDQUNYLDBEQUEwRCxDQUFDLENBQUM7U0FDakU7SUFDSCxDQUFDLENBQUMsQ0FBQztJQUVILFFBQVEsQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLEVBQUU7UUFDbkIsSUFBSSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQyxDQUFDLEtBQUssRUFBRSxXQUFXLENBQUMsS0FBSyxDQUFDLEVBQUU7WUFDakQsTUFBTSxJQUFJLEtBQUssQ0FDWCwwREFBMEQsQ0FBQyxDQUFDO1NBQ2pFO0lBQ0gsQ0FBQyxDQUFDLENBQUM7SUFFSCxNQUFNLE1BQU0sR0FBZSxRQUFRLENBQUM7SUFFcEMsT0FBTyxNQUFNLENBQUMsU0FBUyxDQUFDLElBQUksRUFBRSxNQUFtQyxDQUFDLENBQUM7QUFDckUsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLElBQUksR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUMsS0FBSyxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi9lbmdpbmUnO1xuaW1wb3J0IHtBZGROLCBBZGROSW5wdXRzfSBmcm9tICcuLi9rZXJuZWxfbmFtZXMnO1xuaW1wb3J0IHtUZW5zb3J9IGZyb20gJy4uL3RlbnNvcic7XG5pbXBvcnQge05hbWVkVGVuc29yTWFwfSBmcm9tICcuLi90ZW5zb3JfdHlwZXMnO1xuaW1wb3J0IHtjb252ZXJ0VG9UZW5zb3J9IGZyb20gJy4uL3RlbnNvcl91dGlsX2Vudic7XG5pbXBvcnQge1RlbnNvckxpa2V9IGZyb20gJy4uL3R5cGVzJztcbmltcG9ydCAqIGFzIHV0aWwgZnJvbSAnLi4vdXRpbCc7XG5cbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcblxuLyoqXG4gKiBBZGRzIGEgbGlzdCBvZiBgdGYuVGVuc29yYHMgZWxlbWVudC13aXNlLCBlYWNoIHdpdGggdGhlIHNhbWUgc2hhcGUgYW5kIGR0eXBlLlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBhID0gdGYudGVuc29yMWQoWzEsIDJdKTtcbiAqIGNvbnN0IGIgPSB0Zi50ZW5zb3IxZChbMywgNF0pO1xuICogY29uc3QgYyA9IHRmLnRlbnNvcjFkKFs1LCA2XSk7XG4gKlxuICogdGYuYWRkTihbYSwgYiwgY10pLnByaW50KCk7XG4gKiBgYGBcbiAqIEBwYXJhbSB0ZW5zb3JzIEEgbGlzdCBvZiB0ZW5zb3JzIHdpdGggdGhlIHNhbWUgc2hhcGUgYW5kIGR0eXBlLlxuICogQGRvYyB7aGVhZGluZzogJ09wZXJhdGlvbnMnLCBzdWJoZWFkaW5nOiAnQXJpdGhtZXRpYyd9XG4gKi9cbmZ1bmN0aW9uIGFkZE5fPFQgZXh0ZW5kcyBUZW5zb3I+KHRlbnNvcnM6IEFycmF5PFR8VGVuc29yTGlrZT4pOiBUIHtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICBBcnJheS5pc0FycmF5KHRlbnNvcnMpLFxuICAgICAgKCkgPT4gJ1RoZSBhcmd1bWVudCBwYXNzZWQgdG8gdGYuYWRkTigpIG11c3QgYmUgYSBsaXN0IG9mIHRlbnNvcnMnKTtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICB0ZW5zb3JzLmxlbmd0aCA+PSAxLFxuICAgICAgKCkgPT4gYE11c3QgcGFzcyBhdCBsZWFzdCBvbmUgdGVuc29yIHRvIHRmLmFkZE4oKSwgYnV0IGdvdCBgICtcbiAgICAgICAgICBgJHt0ZW5zb3JzLmxlbmd0aH1gKTtcblxuICBjb25zdCAkdGVuc29ycyA9XG4gICAgICB0ZW5zb3JzLm1hcCgodCwgaSkgPT4gY29udmVydFRvVGVuc29yKHQsIGB0ZW5zb3JzJHtpfWAsICdhZGROJykpO1xuXG4gIGNvbnN0IGZpcnN0VGVuc29yID0gJHRlbnNvcnNbMF07XG4gICR0ZW5zb3JzLmZvckVhY2godCA9PiB7XG4gICAgaWYgKHQuZHR5cGUgIT09IGZpcnN0VGVuc29yLmR0eXBlKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgJ0FsbCB0ZW5zb3JzIHBhc3NlZCB0byB0Zi5hZGROKCkgbXVzdCBoYXZlIHRoZSBzYW1lIGR0eXBlJyk7XG4gICAgfVxuICB9KTtcblxuICAkdGVuc29ycy5mb3JFYWNoKHQgPT4ge1xuICAgIGlmICghdXRpbC5hcnJheXNFcXVhbCh0LnNoYXBlLCBmaXJzdFRlbnNvci5zaGFwZSkpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAnQWxsIHRlbnNvcnMgcGFzc2VkIHRvIHRmLmFkZE4oKSBtdXN0IGhhdmUgdGhlIHNhbWUgc2hhcGUnKTtcbiAgICB9XG4gIH0pO1xuXG4gIGNvbnN0IGlucHV0czogQWRkTklucHV0cyA9ICR0ZW5zb3JzO1xuXG4gIHJldHVybiBFTkdJTkUucnVuS2VybmVsKEFkZE4sIGlucHV0cyBhcyB1bmtub3duIGFzIE5hbWVkVGVuc29yTWFwKTtcbn1cblxuZXhwb3J0IGNvbnN0IGFkZE4gPSAvKiBAX19QVVJFX18gKi8gb3Aoe2FkZE5ffSk7XG4iXX0=