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
import { Reshape } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import { op } from './operation';
/**
 * Reshapes a `tf.Tensor` to a given shape.
 *
 * Given an input tensor, returns a new tensor with the same values as the
 * input tensor with shape `shape`.
 *
 * If one component of shape is the special value -1, the size of that
 * dimension is computed so that the total size remains constant. In
 * particular, a shape of [-1] flattens into 1-D. At most one component of
 * shape can be -1.
 *
 * If shape is 1-D or higher, then the operation returns a tensor with shape
 * shape filled with the values of tensor. In this case, the number of
 * elements implied by shape must be the same as the number of elements in
 * tensor.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3, 4]);
 * x.reshape([2, 2]).print();
 * ```
 *
 * @param x The input tensor to be reshaped.
 * @param shape An array of integers defining the output tensor shape.
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
function reshape_(x, shape) {
    const $x = convertToTensor(x, 'x', 'reshape', 'string_or_numeric');
    const inputs = { x: $x };
    const attrs = { shape };
    return ENGINE.runKernel(Reshape, inputs, attrs);
}
export const reshape = /* @__PURE__ */ op({ reshape_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicmVzaGFwZS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL3Jlc2hhcGUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMsT0FBTyxFQUE4QixNQUFNLGlCQUFpQixDQUFDO0FBSXJFLE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUduRCxPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBRS9COzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBeUJHO0FBQ0gsU0FBUyxRQUFRLENBQ2IsQ0FBb0IsRUFBRSxLQUFrQjtJQUMxQyxNQUFNLEVBQUUsR0FBRyxlQUFlLENBQUMsQ0FBQyxFQUFFLEdBQUcsRUFBRSxTQUFTLEVBQUUsbUJBQW1CLENBQUMsQ0FBQztJQUVuRSxNQUFNLE1BQU0sR0FBa0IsRUFBQyxDQUFDLEVBQUUsRUFBRSxFQUFDLENBQUM7SUFDdEMsTUFBTSxLQUFLLEdBQWlCLEVBQUMsS0FBSyxFQUFDLENBQUM7SUFDcEMsT0FBTyxNQUFNLENBQUMsU0FBUyxDQUNuQixPQUFPLEVBQUUsTUFBbUMsRUFDNUMsS0FBZ0MsQ0FBQyxDQUFDO0FBQ3hDLENBQUM7QUFDRCxNQUFNLENBQUMsTUFBTSxPQUFPLEdBQUcsZUFBZSxDQUFDLEVBQUUsQ0FBQyxFQUFDLFFBQVEsRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi9lbmdpbmUnO1xuaW1wb3J0IHtSZXNoYXBlLCBSZXNoYXBlQXR0cnMsIFJlc2hhcGVJbnB1dHN9IGZyb20gJy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge05hbWVkQXR0ck1hcH0gZnJvbSAnLi4va2VybmVsX3JlZ2lzdHJ5JztcbmltcG9ydCB7VGVuc29yfSBmcm9tICcuLi90ZW5zb3InO1xuaW1wb3J0IHtOYW1lZFRlbnNvck1hcH0gZnJvbSAnLi4vdGVuc29yX3R5cGVzJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtSYW5rLCBTaGFwZU1hcCwgVGVuc29yTGlrZX0gZnJvbSAnLi4vdHlwZXMnO1xuXG5pbXBvcnQge29wfSBmcm9tICcuL29wZXJhdGlvbic7XG5cbi8qKlxuICogUmVzaGFwZXMgYSBgdGYuVGVuc29yYCB0byBhIGdpdmVuIHNoYXBlLlxuICpcbiAqIEdpdmVuIGFuIGlucHV0IHRlbnNvciwgcmV0dXJucyBhIG5ldyB0ZW5zb3Igd2l0aCB0aGUgc2FtZSB2YWx1ZXMgYXMgdGhlXG4gKiBpbnB1dCB0ZW5zb3Igd2l0aCBzaGFwZSBgc2hhcGVgLlxuICpcbiAqIElmIG9uZSBjb21wb25lbnQgb2Ygc2hhcGUgaXMgdGhlIHNwZWNpYWwgdmFsdWUgLTEsIHRoZSBzaXplIG9mIHRoYXRcbiAqIGRpbWVuc2lvbiBpcyBjb21wdXRlZCBzbyB0aGF0IHRoZSB0b3RhbCBzaXplIHJlbWFpbnMgY29uc3RhbnQuIEluXG4gKiBwYXJ0aWN1bGFyLCBhIHNoYXBlIG9mIFstMV0gZmxhdHRlbnMgaW50byAxLUQuIEF0IG1vc3Qgb25lIGNvbXBvbmVudCBvZlxuICogc2hhcGUgY2FuIGJlIC0xLlxuICpcbiAqIElmIHNoYXBlIGlzIDEtRCBvciBoaWdoZXIsIHRoZW4gdGhlIG9wZXJhdGlvbiByZXR1cm5zIGEgdGVuc29yIHdpdGggc2hhcGVcbiAqIHNoYXBlIGZpbGxlZCB3aXRoIHRoZSB2YWx1ZXMgb2YgdGVuc29yLiBJbiB0aGlzIGNhc2UsIHRoZSBudW1iZXIgb2ZcbiAqIGVsZW1lbnRzIGltcGxpZWQgYnkgc2hhcGUgbXVzdCBiZSB0aGUgc2FtZSBhcyB0aGUgbnVtYmVyIG9mIGVsZW1lbnRzIGluXG4gKiB0ZW5zb3IuXG4gKlxuICogYGBganNcbiAqIGNvbnN0IHggPSB0Zi50ZW5zb3IxZChbMSwgMiwgMywgNF0pO1xuICogeC5yZXNoYXBlKFsyLCAyXSkucHJpbnQoKTtcbiAqIGBgYFxuICpcbiAqIEBwYXJhbSB4IFRoZSBpbnB1dCB0ZW5zb3IgdG8gYmUgcmVzaGFwZWQuXG4gKiBAcGFyYW0gc2hhcGUgQW4gYXJyYXkgb2YgaW50ZWdlcnMgZGVmaW5pbmcgdGhlIG91dHB1dCB0ZW5zb3Igc2hhcGUuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ1RlbnNvcnMnLCBzdWJoZWFkaW5nOiAnVHJhbnNmb3JtYXRpb25zJ31cbiAqL1xuZnVuY3Rpb24gcmVzaGFwZV88UiBleHRlbmRzIFJhbms+KFxuICAgIHg6IFRlbnNvcnxUZW5zb3JMaWtlLCBzaGFwZTogU2hhcGVNYXBbUl0pOiBUZW5zb3I8Uj4ge1xuICBjb25zdCAkeCA9IGNvbnZlcnRUb1RlbnNvcih4LCAneCcsICdyZXNoYXBlJywgJ3N0cmluZ19vcl9udW1lcmljJyk7XG5cbiAgY29uc3QgaW5wdXRzOiBSZXNoYXBlSW5wdXRzID0ge3g6ICR4fTtcbiAgY29uc3QgYXR0cnM6IFJlc2hhcGVBdHRycyA9IHtzaGFwZX07XG4gIHJldHVybiBFTkdJTkUucnVuS2VybmVsKFxuICAgICAgUmVzaGFwZSwgaW5wdXRzIGFzIHVua25vd24gYXMgTmFtZWRUZW5zb3JNYXAsXG4gICAgICBhdHRycyBhcyB1bmtub3duIGFzIE5hbWVkQXR0ck1hcCk7XG59XG5leHBvcnQgY29uc3QgcmVzaGFwZSA9IC8qIEBfX1BVUkVfXyAqLyBvcCh7cmVzaGFwZV99KTtcbiJdfQ==