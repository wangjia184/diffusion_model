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
import { Complex } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import * as util from '../util';
import { op } from './operation';
/**
 * Converts two real numbers to a complex number.
 *
 * Given a tensor `real` representing the real part of a complex number, and a
 * tensor `imag` representing the imaginary part of a complex number, this
 * operation returns complex numbers elementwise of the form [r0, i0, r1, i1],
 * where r represents the real part and i represents the imag part.
 *
 * The input tensors real and imag must have the same shape.
 *
 * ```js
 * const real = tf.tensor1d([2.25, 3.25]);
 * const imag = tf.tensor1d([4.75, 5.75]);
 * const complex = tf.complex(real, imag);
 *
 * complex.print();
 * ```
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function complex_(real, imag) {
    const $real = convertToTensor(real, 'real', 'complex');
    const $imag = convertToTensor(imag, 'imag', 'complex');
    util.assertShapesMatch($real.shape, $imag.shape, `real and imag shapes, ${$real.shape} and ${$imag.shape}, ` +
        `must match in call to tf.complex().`);
    const inputs = { real: $real, imag: $imag };
    return ENGINE.runKernel(Complex, inputs);
}
export const complex = /* @__PURE__ */ op({ complex_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY29tcGxleC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL2NvbXBsZXgudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBQ0gsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMsT0FBTyxFQUFnQixNQUFNLGlCQUFpQixDQUFDO0FBR3ZELE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUVuRCxPQUFPLEtBQUssSUFBSSxNQUFNLFNBQVMsQ0FBQztBQUVoQyxPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBRS9COzs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBbUJHO0FBQ0gsU0FBUyxRQUFRLENBQW1CLElBQWtCLEVBQUUsSUFBa0I7SUFDeEUsTUFBTSxLQUFLLEdBQUcsZUFBZSxDQUFDLElBQUksRUFBRSxNQUFNLEVBQUUsU0FBUyxDQUFDLENBQUM7SUFDdkQsTUFBTSxLQUFLLEdBQUcsZUFBZSxDQUFDLElBQUksRUFBRSxNQUFNLEVBQUUsU0FBUyxDQUFDLENBQUM7SUFDdkQsSUFBSSxDQUFDLGlCQUFpQixDQUNsQixLQUFLLENBQUMsS0FBSyxFQUFFLEtBQUssQ0FBQyxLQUFLLEVBQ3hCLHlCQUF5QixLQUFLLENBQUMsS0FBSyxRQUFRLEtBQUssQ0FBQyxLQUFLLElBQUk7UUFDdkQscUNBQXFDLENBQUMsQ0FBQztJQUUvQyxNQUFNLE1BQU0sR0FBa0IsRUFBQyxJQUFJLEVBQUUsS0FBSyxFQUFFLElBQUksRUFBRSxLQUFLLEVBQUMsQ0FBQztJQUN6RCxPQUFPLE1BQU0sQ0FBQyxTQUFTLENBQUMsT0FBTyxFQUFFLE1BQW1DLENBQUMsQ0FBQztBQUN4RSxDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sT0FBTyxHQUFHLGVBQWUsQ0FBQyxFQUFFLENBQUMsRUFBQyxRQUFRLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuaW1wb3J0IHtFTkdJTkV9IGZyb20gJy4uL2VuZ2luZSc7XG5pbXBvcnQge0NvbXBsZXgsIENvbXBsZXhJbnB1dHN9IGZyb20gJy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7TmFtZWRUZW5zb3JNYXB9IGZyb20gJy4uL3RlbnNvcl90eXBlcyc7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7VGVuc29yTGlrZX0gZnJvbSAnLi4vdHlwZXMnO1xuaW1wb3J0ICogYXMgdXRpbCBmcm9tICcuLi91dGlsJztcblxuaW1wb3J0IHtvcH0gZnJvbSAnLi9vcGVyYXRpb24nO1xuXG4vKipcbiAqIENvbnZlcnRzIHR3byByZWFsIG51bWJlcnMgdG8gYSBjb21wbGV4IG51bWJlci5cbiAqXG4gKiBHaXZlbiBhIHRlbnNvciBgcmVhbGAgcmVwcmVzZW50aW5nIHRoZSByZWFsIHBhcnQgb2YgYSBjb21wbGV4IG51bWJlciwgYW5kIGFcbiAqIHRlbnNvciBgaW1hZ2AgcmVwcmVzZW50aW5nIHRoZSBpbWFnaW5hcnkgcGFydCBvZiBhIGNvbXBsZXggbnVtYmVyLCB0aGlzXG4gKiBvcGVyYXRpb24gcmV0dXJucyBjb21wbGV4IG51bWJlcnMgZWxlbWVudHdpc2Ugb2YgdGhlIGZvcm0gW3IwLCBpMCwgcjEsIGkxXSxcbiAqIHdoZXJlIHIgcmVwcmVzZW50cyB0aGUgcmVhbCBwYXJ0IGFuZCBpIHJlcHJlc2VudHMgdGhlIGltYWcgcGFydC5cbiAqXG4gKiBUaGUgaW5wdXQgdGVuc29ycyByZWFsIGFuZCBpbWFnIG11c3QgaGF2ZSB0aGUgc2FtZSBzaGFwZS5cbiAqXG4gKiBgYGBqc1xuICogY29uc3QgcmVhbCA9IHRmLnRlbnNvcjFkKFsyLjI1LCAzLjI1XSk7XG4gKiBjb25zdCBpbWFnID0gdGYudGVuc29yMWQoWzQuNzUsIDUuNzVdKTtcbiAqIGNvbnN0IGNvbXBsZXggPSB0Zi5jb21wbGV4KHJlYWwsIGltYWcpO1xuICpcbiAqIGNvbXBsZXgucHJpbnQoKTtcbiAqIGBgYFxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdUZW5zb3JzJywgc3ViaGVhZGluZzogJ0NyZWF0aW9uJ31cbiAqL1xuZnVuY3Rpb24gY29tcGxleF88VCBleHRlbmRzIFRlbnNvcj4ocmVhbDogVHxUZW5zb3JMaWtlLCBpbWFnOiBUfFRlbnNvckxpa2UpOiBUIHtcbiAgY29uc3QgJHJlYWwgPSBjb252ZXJ0VG9UZW5zb3IocmVhbCwgJ3JlYWwnLCAnY29tcGxleCcpO1xuICBjb25zdCAkaW1hZyA9IGNvbnZlcnRUb1RlbnNvcihpbWFnLCAnaW1hZycsICdjb21wbGV4Jyk7XG4gIHV0aWwuYXNzZXJ0U2hhcGVzTWF0Y2goXG4gICAgICAkcmVhbC5zaGFwZSwgJGltYWcuc2hhcGUsXG4gICAgICBgcmVhbCBhbmQgaW1hZyBzaGFwZXMsICR7JHJlYWwuc2hhcGV9IGFuZCAkeyRpbWFnLnNoYXBlfSwgYCArXG4gICAgICAgICAgYG11c3QgbWF0Y2ggaW4gY2FsbCB0byB0Zi5jb21wbGV4KCkuYCk7XG5cbiAgY29uc3QgaW5wdXRzOiBDb21wbGV4SW5wdXRzID0ge3JlYWw6ICRyZWFsLCBpbWFnOiAkaW1hZ307XG4gIHJldHVybiBFTkdJTkUucnVuS2VybmVsKENvbXBsZXgsIGlucHV0cyBhcyB1bmtub3duIGFzIE5hbWVkVGVuc29yTWFwKTtcbn1cblxuZXhwb3J0IGNvbnN0IGNvbXBsZXggPSAvKiBAX19QVVJFX18gKi8gb3Aoe2NvbXBsZXhffSk7XG4iXX0=