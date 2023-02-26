/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
import { Einsum } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import { op } from './operation';
/**
 * Tensor contraction over specified indices and outer product.
 *
 * `einsum` allows defining Tensors by defining their element-wise computation.
 * This computation is based on
 * [Einstein summation](https://en.wikipedia.org/wiki/Einstein_notation).
 *
 * Some special cases include:
 *
 * Matrix multiplication:
 * ```js
 * const x = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);
 * const y = tf.tensor2d([[0, 1], [2, 3], [4, 5]]);
 * x.print();
 * y.print();
 * tf.einsum('ij,jk->ik', x, y).print();
 * ```
 *
 * Dot product:
 * ```js
 * const x = tf.tensor1d([1, 2, 3]);
 * const y = tf.tensor1d([0, 1, 2]);
 * x.print();
 * y.print();
 * tf.einsum('i,i->', x, y).print();
 * ```
 *
 * Batch dot product:
 * ```js
 * const x = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);
 * const y = tf.tensor2d([[0, 1, 2], [3, 4, 5]]);
 * x.print();
 * y.print();
 * tf.einsum('bi,bi->b', x, y).print();
 * ```
 *
 * Outer prouduct:
 * ```js
 * const x = tf.tensor1d([1, 3, 5]);
 * const y = tf.tensor1d([2, 4, 6]);
 * x.print();
 * y.print();
 * tf.einsum('i,j->ij', x, y).print();
 * ```
 *
 * Matrix transpose:
 * ```js
 * const x = tf.tensor2d([[1, 2], [3, 4]]);
 * x.print();
 * tf.einsum('ij->ji', x).print();
 * ```
 *
 * Batch matrix transpose:
 * ```js
 * const x = tf.tensor3d([[[1, 2], [3, 4]], [[-1, -2], [-3, -4]]]);
 * x.print();
 * tf.einsum('bij->bji', x).print();
 * ```
 *
 * Limitations:
 *
 * This implementation of einsum has the following limitations:
 *
 * - Does not support >2 input tensors.
 * - Does not support duplicate axes for any given input tensor. E.g., equation
 *   'ii->' is not supported.
 * - The `...` notation is not supported.
 *
 * @param equation a string describing the contraction, in the same format as
 * [numpy.einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html).
 * @param tensors the input(s) to contract (each one a Tensor), whose shapes
 *     should be consistent with equation.
 * @returns The output tensor.
 *
 * @doc {heading: 'Tensors', subheading: 'Matrices'}
 */
export function einsum_(equation, ...tensors) {
    const $tensors = tensors.map((t, i) => convertToTensor(t, `tensors${i}`, 'einsum'));
    const attrs = { equation };
    return ENGINE.runKernel(Einsum, $tensors, attrs);
}
export const einsum = /* @__PURE__ */ op({ einsum_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZWluc3VtLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvZWluc3VtLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDakMsT0FBTyxFQUFDLE1BQU0sRUFBYyxNQUFNLGlCQUFpQixDQUFDO0FBSXBELE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUVuRCxPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBRS9COzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7R0EyRUc7QUFDSCxNQUFNLFVBQVUsT0FBTyxDQUFDLFFBQWdCLEVBQUUsR0FBRyxPQUFpQjtJQUM1RCxNQUFNLFFBQVEsR0FDVixPQUFPLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUMsZUFBZSxDQUFDLENBQUMsRUFBRSxVQUFVLENBQUMsRUFBRSxFQUFFLFFBQVEsQ0FBQyxDQUFDLENBQUM7SUFDdkUsTUFBTSxLQUFLLEdBQWdCLEVBQUMsUUFBUSxFQUFDLENBQUM7SUFDdEMsT0FBTyxNQUFNLENBQUMsU0FBUyxDQUNuQixNQUFNLEVBQUUsUUFBcUMsRUFDN0MsS0FBZ0MsQ0FBQyxDQUFDO0FBQ3hDLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxNQUFNLEdBQUcsZUFBZSxDQUFDLEVBQUUsQ0FBQyxFQUFDLE9BQU8sRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMSBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi9lbmdpbmUnO1xuaW1wb3J0IHtFaW5zdW0sIEVpbnN1bUF0dHJzfSBmcm9tICcuLi9rZXJuZWxfbmFtZXMnO1xuaW1wb3J0IHtOYW1lZEF0dHJNYXB9IGZyb20gJy4uL2tlcm5lbF9yZWdpc3RyeSc7XG5pbXBvcnQge1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7TmFtZWRUZW5zb3JNYXB9IGZyb20gJy4uL3RlbnNvcl90eXBlcyc7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yX3V0aWxfZW52JztcblxuaW1wb3J0IHtvcH0gZnJvbSAnLi9vcGVyYXRpb24nO1xuXG4vKipcbiAqIFRlbnNvciBjb250cmFjdGlvbiBvdmVyIHNwZWNpZmllZCBpbmRpY2VzIGFuZCBvdXRlciBwcm9kdWN0LlxuICpcbiAqIGBlaW5zdW1gIGFsbG93cyBkZWZpbmluZyBUZW5zb3JzIGJ5IGRlZmluaW5nIHRoZWlyIGVsZW1lbnQtd2lzZSBjb21wdXRhdGlvbi5cbiAqIFRoaXMgY29tcHV0YXRpb24gaXMgYmFzZWQgb25cbiAqIFtFaW5zdGVpbiBzdW1tYXRpb25dKGh0dHBzOi8vZW4ud2lraXBlZGlhLm9yZy93aWtpL0VpbnN0ZWluX25vdGF0aW9uKS5cbiAqXG4gKiBTb21lIHNwZWNpYWwgY2FzZXMgaW5jbHVkZTpcbiAqXG4gKiBNYXRyaXggbXVsdGlwbGljYXRpb246XG4gKiBgYGBqc1xuICogY29uc3QgeCA9IHRmLnRlbnNvcjJkKFtbMSwgMiwgM10sIFs0LCA1LCA2XV0pO1xuICogY29uc3QgeSA9IHRmLnRlbnNvcjJkKFtbMCwgMV0sIFsyLCAzXSwgWzQsIDVdXSk7XG4gKiB4LnByaW50KCk7XG4gKiB5LnByaW50KCk7XG4gKiB0Zi5laW5zdW0oJ2lqLGprLT5paycsIHgsIHkpLnByaW50KCk7XG4gKiBgYGBcbiAqXG4gKiBEb3QgcHJvZHVjdDpcbiAqIGBgYGpzXG4gKiBjb25zdCB4ID0gdGYudGVuc29yMWQoWzEsIDIsIDNdKTtcbiAqIGNvbnN0IHkgPSB0Zi50ZW5zb3IxZChbMCwgMSwgMl0pO1xuICogeC5wcmludCgpO1xuICogeS5wcmludCgpO1xuICogdGYuZWluc3VtKCdpLGktPicsIHgsIHkpLnByaW50KCk7XG4gKiBgYGBcbiAqXG4gKiBCYXRjaCBkb3QgcHJvZHVjdDpcbiAqIGBgYGpzXG4gKiBjb25zdCB4ID0gdGYudGVuc29yMmQoW1sxLCAyLCAzXSwgWzQsIDUsIDZdXSk7XG4gKiBjb25zdCB5ID0gdGYudGVuc29yMmQoW1swLCAxLCAyXSwgWzMsIDQsIDVdXSk7XG4gKiB4LnByaW50KCk7XG4gKiB5LnByaW50KCk7XG4gKiB0Zi5laW5zdW0oJ2JpLGJpLT5iJywgeCwgeSkucHJpbnQoKTtcbiAqIGBgYFxuICpcbiAqIE91dGVyIHByb3VkdWN0OlxuICogYGBganNcbiAqIGNvbnN0IHggPSB0Zi50ZW5zb3IxZChbMSwgMywgNV0pO1xuICogY29uc3QgeSA9IHRmLnRlbnNvcjFkKFsyLCA0LCA2XSk7XG4gKiB4LnByaW50KCk7XG4gKiB5LnByaW50KCk7XG4gKiB0Zi5laW5zdW0oJ2ksai0+aWonLCB4LCB5KS5wcmludCgpO1xuICogYGBgXG4gKlxuICogTWF0cml4IHRyYW5zcG9zZTpcbiAqIGBgYGpzXG4gKiBjb25zdCB4ID0gdGYudGVuc29yMmQoW1sxLCAyXSwgWzMsIDRdXSk7XG4gKiB4LnByaW50KCk7XG4gKiB0Zi5laW5zdW0oJ2lqLT5qaScsIHgpLnByaW50KCk7XG4gKiBgYGBcbiAqXG4gKiBCYXRjaCBtYXRyaXggdHJhbnNwb3NlOlxuICogYGBganNcbiAqIGNvbnN0IHggPSB0Zi50ZW5zb3IzZChbW1sxLCAyXSwgWzMsIDRdXSwgW1stMSwgLTJdLCBbLTMsIC00XV1dKTtcbiAqIHgucHJpbnQoKTtcbiAqIHRmLmVpbnN1bSgnYmlqLT5iamknLCB4KS5wcmludCgpO1xuICogYGBgXG4gKlxuICogTGltaXRhdGlvbnM6XG4gKlxuICogVGhpcyBpbXBsZW1lbnRhdGlvbiBvZiBlaW5zdW0gaGFzIHRoZSBmb2xsb3dpbmcgbGltaXRhdGlvbnM6XG4gKlxuICogLSBEb2VzIG5vdCBzdXBwb3J0ID4yIGlucHV0IHRlbnNvcnMuXG4gKiAtIERvZXMgbm90IHN1cHBvcnQgZHVwbGljYXRlIGF4ZXMgZm9yIGFueSBnaXZlbiBpbnB1dCB0ZW5zb3IuIEUuZy4sIGVxdWF0aW9uXG4gKiAgICdpaS0+JyBpcyBub3Qgc3VwcG9ydGVkLlxuICogLSBUaGUgYC4uLmAgbm90YXRpb24gaXMgbm90IHN1cHBvcnRlZC5cbiAqXG4gKiBAcGFyYW0gZXF1YXRpb24gYSBzdHJpbmcgZGVzY3JpYmluZyB0aGUgY29udHJhY3Rpb24sIGluIHRoZSBzYW1lIGZvcm1hdCBhc1xuICogW251bXB5LmVpbnN1bV0oaHR0cHM6Ly9udW1weS5vcmcvZG9jL3N0YWJsZS9yZWZlcmVuY2UvZ2VuZXJhdGVkL251bXB5LmVpbnN1bS5odG1sKS5cbiAqIEBwYXJhbSB0ZW5zb3JzIHRoZSBpbnB1dChzKSB0byBjb250cmFjdCAoZWFjaCBvbmUgYSBUZW5zb3IpLCB3aG9zZSBzaGFwZXNcbiAqICAgICBzaG91bGQgYmUgY29uc2lzdGVudCB3aXRoIGVxdWF0aW9uLlxuICogQHJldHVybnMgVGhlIG91dHB1dCB0ZW5zb3IuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ1RlbnNvcnMnLCBzdWJoZWFkaW5nOiAnTWF0cmljZXMnfVxuICovXG5leHBvcnQgZnVuY3Rpb24gZWluc3VtXyhlcXVhdGlvbjogc3RyaW5nLCAuLi50ZW5zb3JzOiBUZW5zb3JbXSk6IFRlbnNvciB7XG4gIGNvbnN0ICR0ZW5zb3JzID1cbiAgICAgIHRlbnNvcnMubWFwKCh0LCBpKSA9PiBjb252ZXJ0VG9UZW5zb3IodCwgYHRlbnNvcnMke2l9YCwgJ2VpbnN1bScpKTtcbiAgY29uc3QgYXR0cnM6IEVpbnN1bUF0dHJzID0ge2VxdWF0aW9ufTtcbiAgcmV0dXJuIEVOR0lORS5ydW5LZXJuZWwoXG4gICAgICBFaW5zdW0sICR0ZW5zb3JzIGFzIHVua25vd24gYXMgTmFtZWRUZW5zb3JNYXAsXG4gICAgICBhdHRycyBhcyB1bmtub3duIGFzIE5hbWVkQXR0ck1hcCk7XG59XG5cbmV4cG9ydCBjb25zdCBlaW5zdW0gPSAvKiBAX19QVVJFX18gKi8gb3Aoe2VpbnN1bV99KTtcbiJdfQ==