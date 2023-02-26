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
import { Softmax } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import { op } from './operation';
/**
 * Computes the softmax normalized vector given the logits.
 *
 * ```js
 * const a = tf.tensor1d([1, 2, 3]);
 *
 * a.softmax().print();  // or tf.softmax(a)
 * ```
 *
 * ```js
 * const a = tf.tensor2d([2, 4, 6, 1, 2, 3], [2, 3]);
 *
 * a.softmax().print();  // or tf.softmax(a)
 * ```
 *
 * @param logits The logits array.
 * @param dim The dimension softmax would be performed on. Defaults to `-1`
 *     which indicates the last dimension.
 *
 * @doc {heading: 'Operations', subheading: 'Normalization'}
 */
function softmax_(logits, dim = -1) {
    const $logits = convertToTensor(logits, 'logits', 'softmax', 'float32');
    if (dim === -1) {
        dim = $logits.rank - 1;
    }
    if (dim !== $logits.rank - 1) {
        throw Error('Softmax along a non-last dimension is not yet supported. ' +
            `Logits was rank ${$logits.rank} and dim was ${dim}`);
    }
    const inputs = { logits: $logits };
    const attrs = { dim };
    return ENGINE.runKernel(Softmax, inputs, attrs);
}
export const softmax = /* @__PURE__ */ op({ softmax_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic29mdG1heC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL3NvZnRtYXgudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMsT0FBTyxFQUE4QixNQUFNLGlCQUFpQixDQUFDO0FBSXJFLE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUduRCxPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBRS9COzs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQW9CRztBQUNILFNBQVMsUUFBUSxDQUFtQixNQUFvQixFQUFFLEdBQUcsR0FBRyxDQUFDLENBQUM7SUFDaEUsTUFBTSxPQUFPLEdBQUcsZUFBZSxDQUFDLE1BQU0sRUFBRSxRQUFRLEVBQUUsU0FBUyxFQUFFLFNBQVMsQ0FBQyxDQUFDO0lBRXhFLElBQUksR0FBRyxLQUFLLENBQUMsQ0FBQyxFQUFFO1FBQ2QsR0FBRyxHQUFHLE9BQU8sQ0FBQyxJQUFJLEdBQUcsQ0FBQyxDQUFDO0tBQ3hCO0lBQ0QsSUFBSSxHQUFHLEtBQUssT0FBTyxDQUFDLElBQUksR0FBRyxDQUFDLEVBQUU7UUFDNUIsTUFBTSxLQUFLLENBQ1AsMkRBQTJEO1lBQzNELG1CQUFtQixPQUFPLENBQUMsSUFBSSxnQkFBZ0IsR0FBRyxFQUFFLENBQUMsQ0FBQztLQUMzRDtJQUVELE1BQU0sTUFBTSxHQUFrQixFQUFDLE1BQU0sRUFBRSxPQUFPLEVBQUMsQ0FBQztJQUNoRCxNQUFNLEtBQUssR0FBaUIsRUFBQyxHQUFHLEVBQUMsQ0FBQztJQUVsQyxPQUFPLE1BQU0sQ0FBQyxTQUFTLENBQ25CLE9BQU8sRUFBRSxNQUFtQyxFQUM1QyxLQUFnQyxDQUFDLENBQUM7QUFDeEMsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLE9BQU8sR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUMsUUFBUSxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtFTkdJTkV9IGZyb20gJy4uL2VuZ2luZSc7XG5pbXBvcnQge1NvZnRtYXgsIFNvZnRtYXhBdHRycywgU29mdG1heElucHV0c30gZnJvbSAnLi4va2VybmVsX25hbWVzJztcbmltcG9ydCB7TmFtZWRBdHRyTWFwfSBmcm9tICcuLi9rZXJuZWxfcmVnaXN0cnknO1xuaW1wb3J0IHtUZW5zb3J9IGZyb20gJy4uL3RlbnNvcic7XG5pbXBvcnQge05hbWVkVGVuc29yTWFwfSBmcm9tICcuLi90ZW5zb3JfdHlwZXMnO1xuaW1wb3J0IHtjb252ZXJ0VG9UZW5zb3J9IGZyb20gJy4uL3RlbnNvcl91dGlsX2Vudic7XG5pbXBvcnQge1RlbnNvckxpa2V9IGZyb20gJy4uL3R5cGVzJztcblxuaW1wb3J0IHtvcH0gZnJvbSAnLi9vcGVyYXRpb24nO1xuXG4vKipcbiAqIENvbXB1dGVzIHRoZSBzb2Z0bWF4IG5vcm1hbGl6ZWQgdmVjdG9yIGdpdmVuIHRoZSBsb2dpdHMuXG4gKlxuICogYGBganNcbiAqIGNvbnN0IGEgPSB0Zi50ZW5zb3IxZChbMSwgMiwgM10pO1xuICpcbiAqIGEuc29mdG1heCgpLnByaW50KCk7ICAvLyBvciB0Zi5zb2Z0bWF4KGEpXG4gKiBgYGBcbiAqXG4gKiBgYGBqc1xuICogY29uc3QgYSA9IHRmLnRlbnNvcjJkKFsyLCA0LCA2LCAxLCAyLCAzXSwgWzIsIDNdKTtcbiAqXG4gKiBhLnNvZnRtYXgoKS5wcmludCgpOyAgLy8gb3IgdGYuc29mdG1heChhKVxuICogYGBgXG4gKlxuICogQHBhcmFtIGxvZ2l0cyBUaGUgbG9naXRzIGFycmF5LlxuICogQHBhcmFtIGRpbSBUaGUgZGltZW5zaW9uIHNvZnRtYXggd291bGQgYmUgcGVyZm9ybWVkIG9uLiBEZWZhdWx0cyB0byBgLTFgXG4gKiAgICAgd2hpY2ggaW5kaWNhdGVzIHRoZSBsYXN0IGRpbWVuc2lvbi5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnT3BlcmF0aW9ucycsIHN1YmhlYWRpbmc6ICdOb3JtYWxpemF0aW9uJ31cbiAqL1xuZnVuY3Rpb24gc29mdG1heF88VCBleHRlbmRzIFRlbnNvcj4obG9naXRzOiBUfFRlbnNvckxpa2UsIGRpbSA9IC0xKTogVCB7XG4gIGNvbnN0ICRsb2dpdHMgPSBjb252ZXJ0VG9UZW5zb3IobG9naXRzLCAnbG9naXRzJywgJ3NvZnRtYXgnLCAnZmxvYXQzMicpO1xuXG4gIGlmIChkaW0gPT09IC0xKSB7XG4gICAgZGltID0gJGxvZ2l0cy5yYW5rIC0gMTtcbiAgfVxuICBpZiAoZGltICE9PSAkbG9naXRzLnJhbmsgLSAxKSB7XG4gICAgdGhyb3cgRXJyb3IoXG4gICAgICAgICdTb2Z0bWF4IGFsb25nIGEgbm9uLWxhc3QgZGltZW5zaW9uIGlzIG5vdCB5ZXQgc3VwcG9ydGVkLiAnICtcbiAgICAgICAgYExvZ2l0cyB3YXMgcmFuayAkeyRsb2dpdHMucmFua30gYW5kIGRpbSB3YXMgJHtkaW19YCk7XG4gIH1cblxuICBjb25zdCBpbnB1dHM6IFNvZnRtYXhJbnB1dHMgPSB7bG9naXRzOiAkbG9naXRzfTtcbiAgY29uc3QgYXR0cnM6IFNvZnRtYXhBdHRycyA9IHtkaW19O1xuXG4gIHJldHVybiBFTkdJTkUucnVuS2VybmVsKFxuICAgICAgU29mdG1heCwgaW5wdXRzIGFzIHVua25vd24gYXMgTmFtZWRUZW5zb3JNYXAsXG4gICAgICBhdHRycyBhcyB1bmtub3duIGFzIE5hbWVkQXR0ck1hcCk7XG59XG5cbmV4cG9ydCBjb25zdCBzb2Z0bWF4ID0gLyogQF9fUFVSRV9fICovIG9wKHtzb2Z0bWF4X30pO1xuIl19