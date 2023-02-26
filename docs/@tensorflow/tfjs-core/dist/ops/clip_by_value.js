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
import { ClipByValue } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import * as util from '../util';
import { fill } from './fill';
import { op } from './operation';
/**
 * Clips values element-wise. `max(min(x, clipValueMax), clipValueMin)`
 *
 * ```js
 * const x = tf.tensor1d([-1, 2, -3, 4]);
 *
 * x.clipByValue(-2, 3).print();  // or tf.clipByValue(x, -2, 3)
 * ```
 * @param x The input tensor.
 * @param clipValueMin Lower bound of range to be clipped to.
 * @param clipValueMax Upper bound of range to be clipped to.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function clipByValue_(x, clipValueMin, clipValueMax) {
    const $x = convertToTensor(x, 'x', 'clipByValue');
    util.assert((clipValueMin <= clipValueMax), () => `Error in clip: min (${clipValueMin}) must be ` +
        `less than or equal to max (${clipValueMax}).`);
    if (clipValueMin === clipValueMax) {
        return fill($x.shape, clipValueMin, $x.dtype);
    }
    const inputs = { x: $x };
    const attrs = { clipValueMin, clipValueMax };
    return ENGINE.runKernel(ClipByValue, inputs, attrs);
}
export const clipByValue = /* @__PURE__ */ op({ clipByValue_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY2xpcF9ieV92YWx1ZS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL2NsaXBfYnlfdmFsdWUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBQ0gsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMsV0FBVyxFQUFzQyxNQUFNLGlCQUFpQixDQUFDO0FBSWpGLE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUVuRCxPQUFPLEtBQUssSUFBSSxNQUFNLFNBQVMsQ0FBQztBQUNoQyxPQUFPLEVBQUMsSUFBSSxFQUFDLE1BQU0sUUFBUSxDQUFDO0FBRTVCLE9BQU8sRUFBQyxFQUFFLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFFL0I7Ozs7Ozs7Ozs7Ozs7R0FhRztBQUNILFNBQVMsWUFBWSxDQUNqQixDQUFlLEVBQUUsWUFBb0IsRUFBRSxZQUFvQjtJQUM3RCxNQUFNLEVBQUUsR0FBRyxlQUFlLENBQUMsQ0FBQyxFQUFFLEdBQUcsRUFBRSxhQUFhLENBQUMsQ0FBQztJQUNsRCxJQUFJLENBQUMsTUFBTSxDQUNQLENBQUMsWUFBWSxJQUFJLFlBQVksQ0FBQyxFQUM5QixHQUFHLEVBQUUsQ0FBQyx1QkFBdUIsWUFBWSxZQUFZO1FBQ2pELDhCQUE4QixZQUFZLElBQUksQ0FBQyxDQUFDO0lBRXhELElBQUksWUFBWSxLQUFLLFlBQVksRUFBRTtRQUNqQyxPQUFPLElBQUksQ0FBQyxFQUFFLENBQUMsS0FBSyxFQUFFLFlBQVksRUFBRSxFQUFFLENBQUMsS0FBSyxDQUFNLENBQUM7S0FDcEQ7SUFFRCxNQUFNLE1BQU0sR0FBc0IsRUFBQyxDQUFDLEVBQUUsRUFBRSxFQUFDLENBQUM7SUFDMUMsTUFBTSxLQUFLLEdBQXFCLEVBQUMsWUFBWSxFQUFFLFlBQVksRUFBQyxDQUFDO0lBRTdELE9BQU8sTUFBTSxDQUFDLFNBQVMsQ0FDbkIsV0FBVyxFQUFFLE1BQW1DLEVBQ2hELEtBQWdDLENBQUMsQ0FBQztBQUN4QyxDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sV0FBVyxHQUFHLGVBQWUsQ0FBQyxFQUFFLENBQUMsRUFBQyxZQUFZLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTggR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuaW1wb3J0IHtFTkdJTkV9IGZyb20gJy4uL2VuZ2luZSc7XG5pbXBvcnQge0NsaXBCeVZhbHVlLCBDbGlwQnlWYWx1ZUF0dHJzLCBDbGlwQnlWYWx1ZUlucHV0c30gZnJvbSAnLi4va2VybmVsX25hbWVzJztcbmltcG9ydCB7TmFtZWRBdHRyTWFwfSBmcm9tICcuLi9rZXJuZWxfcmVnaXN0cnknO1xuaW1wb3J0IHtUZW5zb3J9IGZyb20gJy4uL3RlbnNvcic7XG5pbXBvcnQge05hbWVkVGVuc29yTWFwfSBmcm9tICcuLi90ZW5zb3JfdHlwZXMnO1xuaW1wb3J0IHtjb252ZXJ0VG9UZW5zb3J9IGZyb20gJy4uL3RlbnNvcl91dGlsX2Vudic7XG5pbXBvcnQge1RlbnNvckxpa2V9IGZyb20gJy4uL3R5cGVzJztcbmltcG9ydCAqIGFzIHV0aWwgZnJvbSAnLi4vdXRpbCc7XG5pbXBvcnQge2ZpbGx9IGZyb20gJy4vZmlsbCc7XG5cbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcblxuLyoqXG4gKiBDbGlwcyB2YWx1ZXMgZWxlbWVudC13aXNlLiBgbWF4KG1pbih4LCBjbGlwVmFsdWVNYXgpLCBjbGlwVmFsdWVNaW4pYFxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCB4ID0gdGYudGVuc29yMWQoWy0xLCAyLCAtMywgNF0pO1xuICpcbiAqIHguY2xpcEJ5VmFsdWUoLTIsIDMpLnByaW50KCk7ICAvLyBvciB0Zi5jbGlwQnlWYWx1ZSh4LCAtMiwgMylcbiAqIGBgYFxuICogQHBhcmFtIHggVGhlIGlucHV0IHRlbnNvci5cbiAqIEBwYXJhbSBjbGlwVmFsdWVNaW4gTG93ZXIgYm91bmQgb2YgcmFuZ2UgdG8gYmUgY2xpcHBlZCB0by5cbiAqIEBwYXJhbSBjbGlwVmFsdWVNYXggVXBwZXIgYm91bmQgb2YgcmFuZ2UgdG8gYmUgY2xpcHBlZCB0by5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnT3BlcmF0aW9ucycsIHN1YmhlYWRpbmc6ICdCYXNpYyBtYXRoJ31cbiAqL1xuZnVuY3Rpb24gY2xpcEJ5VmFsdWVfPFQgZXh0ZW5kcyBUZW5zb3I+KFxuICAgIHg6IFR8VGVuc29yTGlrZSwgY2xpcFZhbHVlTWluOiBudW1iZXIsIGNsaXBWYWx1ZU1heDogbnVtYmVyKTogVCB7XG4gIGNvbnN0ICR4ID0gY29udmVydFRvVGVuc29yKHgsICd4JywgJ2NsaXBCeVZhbHVlJyk7XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgKGNsaXBWYWx1ZU1pbiA8PSBjbGlwVmFsdWVNYXgpLFxuICAgICAgKCkgPT4gYEVycm9yIGluIGNsaXA6IG1pbiAoJHtjbGlwVmFsdWVNaW59KSBtdXN0IGJlIGAgK1xuICAgICAgICAgIGBsZXNzIHRoYW4gb3IgZXF1YWwgdG8gbWF4ICgke2NsaXBWYWx1ZU1heH0pLmApO1xuXG4gIGlmIChjbGlwVmFsdWVNaW4gPT09IGNsaXBWYWx1ZU1heCkge1xuICAgIHJldHVybiBmaWxsKCR4LnNoYXBlLCBjbGlwVmFsdWVNaW4sICR4LmR0eXBlKSBhcyBUO1xuICB9XG5cbiAgY29uc3QgaW5wdXRzOiBDbGlwQnlWYWx1ZUlucHV0cyA9IHt4OiAkeH07XG4gIGNvbnN0IGF0dHJzOiBDbGlwQnlWYWx1ZUF0dHJzID0ge2NsaXBWYWx1ZU1pbiwgY2xpcFZhbHVlTWF4fTtcblxuICByZXR1cm4gRU5HSU5FLnJ1bktlcm5lbChcbiAgICAgIENsaXBCeVZhbHVlLCBpbnB1dHMgYXMgdW5rbm93biBhcyBOYW1lZFRlbnNvck1hcCxcbiAgICAgIGF0dHJzIGFzIHVua25vd24gYXMgTmFtZWRBdHRyTWFwKTtcbn1cblxuZXhwb3J0IGNvbnN0IGNsaXBCeVZhbHVlID0gLyogQF9fUFVSRV9fICovIG9wKHtjbGlwQnlWYWx1ZV99KTtcbiJdfQ==