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
import { Unique } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import { assert } from '../util';
import { op } from './operation';
/**
 * Finds unique elements along an axis of a tensor.
 *
 * It returns a tensor `values` containing all of the unique elements along the
 * `axis` of the given tensor `x` in the same order that they occur along the
 * `axis` in `x`; `x` does not need to be sorted. It also returns a tensor
 * `indices` the same size as the number of the elements in `x` along the `axis`
 * dimension. It contains the index in the unique output `values`.
 *
 * ```js
 * // A 1-D tensor
 * const a = tf.tensor1d([1, 1, 2, 4, 4, 4, 7, 8, 8]);
 * const {values, indices} = tf.unique(a);
 * values.print();   // [1, 2, 4, 7, 8,]
 * indices.print();  // [0, 0, 1, 2, 2, 2, 3, 4, 4]
 * ```
 *
 * ```js
 * // A 2-D tensor with axis=0
 * //
 * // 'a' is: [[1, 0, 0],
 * //          [1, 0, 0],
 * //          [2, 0, 0]]
 * const a = tf.tensor2d([[1, 0, 0], [1, 0, 0], [2, 0, 0]]);
 * const {values, indices} = tf.unique(a, 0)
 * values.print();   // [[1, 0, 0],
 *                   //  [2, 0, 0]]
 * indices.print();  // [0, 0, 1]
 * ```
 *
 * ```js
 * // A 2-D tensor with axis=1
 * //
 * // 'a' is: [[1, 0, 0],
 * //          [1, 0, 0],
 * //          [2, 0, 0]]
 * const a = tf.tensor2d([[1, 0, 0], [1, 0, 0], [2, 0, 0]]);
 * const {values, indices} = tf.unique(a, 1)
 * values.print();   // [[1, 0],
 *                   //  [1, 0],
 *                   //  [2, 0]]
 * indices.print();  // [0, 1, 1]
 * ```
 * @param x A tensor (int32, string, bool).
 * @param axis The axis of the tensor to find the unique elements.
 * @returns [uniqueElements, indices] (see above for details)
 *
 * @doc {heading: 'Operations', subheading: 'Evaluation'}
 */
function unique_(x, axis = 0) {
    const $x = convertToTensor(x, 'x', 'unique', 'string_or_numeric');
    assert($x.rank > 0, () => 'The input tensor must be at least 1D');
    const inputs = { x: $x };
    const attrs = { axis };
    const [values, indices] = ENGINE.runKernel(Unique, inputs, attrs);
    return { values, indices };
}
export const unique = /* @__PURE__ */ op({ unique_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidW5pcXVlLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvdW5pcXVlLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDakMsT0FBTyxFQUFDLE1BQU0sRUFBNEIsTUFBTSxpQkFBaUIsQ0FBQztBQUlsRSxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFFbkQsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFNBQVMsQ0FBQztBQUUvQixPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBRS9COzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7R0FnREc7QUFDSCxTQUFTLE9BQU8sQ0FDWixDQUFlLEVBQUUsSUFBSSxHQUFHLENBQUM7SUFDM0IsTUFBTSxFQUFFLEdBQUcsZUFBZSxDQUFDLENBQUMsRUFBRSxHQUFHLEVBQUUsUUFBUSxFQUFFLG1CQUFtQixDQUFDLENBQUM7SUFDbEUsTUFBTSxDQUFDLEVBQUUsQ0FBQyxJQUFJLEdBQUcsQ0FBQyxFQUFFLEdBQUcsRUFBRSxDQUFDLHNDQUFzQyxDQUFDLENBQUM7SUFFbEUsTUFBTSxNQUFNLEdBQWlCLEVBQUMsQ0FBQyxFQUFFLEVBQUUsRUFBQyxDQUFDO0lBQ3JDLE1BQU0sS0FBSyxHQUFnQixFQUFDLElBQUksRUFBQyxDQUFDO0lBQ2xDLE1BQU0sQ0FBQyxNQUFNLEVBQUUsT0FBTyxDQUFDLEdBQUcsTUFBTSxDQUFDLFNBQVMsQ0FDdEMsTUFBTSxFQUFFLE1BQW1DLEVBQzNDLEtBQWdDLENBQWtCLENBQUM7SUFDdkQsT0FBTyxFQUFDLE1BQU0sRUFBRSxPQUFPLEVBQUMsQ0FBQztBQUMzQixDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sTUFBTSxHQUFHLGVBQWUsQ0FBQyxFQUFFLENBQUMsRUFBQyxPQUFPLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0VOR0lORX0gZnJvbSAnLi4vZW5naW5lJztcbmltcG9ydCB7VW5pcXVlLCBVbmlxdWVBdHRycywgVW5pcXVlSW5wdXRzfSBmcm9tICcuLi9rZXJuZWxfbmFtZXMnO1xuaW1wb3J0IHtOYW1lZEF0dHJNYXB9IGZyb20gJy4uL2tlcm5lbF9yZWdpc3RyeSc7XG5pbXBvcnQge1RlbnNvciwgVGVuc29yMUR9IGZyb20gJy4uL3RlbnNvcic7XG5pbXBvcnQge05hbWVkVGVuc29yTWFwfSBmcm9tICcuLi90ZW5zb3JfdHlwZXMnO1xuaW1wb3J0IHtjb252ZXJ0VG9UZW5zb3J9IGZyb20gJy4uL3RlbnNvcl91dGlsX2Vudic7XG5pbXBvcnQge1RlbnNvckxpa2V9IGZyb20gJy4uL3R5cGVzJztcbmltcG9ydCB7YXNzZXJ0fSBmcm9tICcuLi91dGlsJztcblxuaW1wb3J0IHtvcH0gZnJvbSAnLi9vcGVyYXRpb24nO1xuXG4vKipcbiAqIEZpbmRzIHVuaXF1ZSBlbGVtZW50cyBhbG9uZyBhbiBheGlzIG9mIGEgdGVuc29yLlxuICpcbiAqIEl0IHJldHVybnMgYSB0ZW5zb3IgYHZhbHVlc2AgY29udGFpbmluZyBhbGwgb2YgdGhlIHVuaXF1ZSBlbGVtZW50cyBhbG9uZyB0aGVcbiAqIGBheGlzYCBvZiB0aGUgZ2l2ZW4gdGVuc29yIGB4YCBpbiB0aGUgc2FtZSBvcmRlciB0aGF0IHRoZXkgb2NjdXIgYWxvbmcgdGhlXG4gKiBgYXhpc2AgaW4gYHhgOyBgeGAgZG9lcyBub3QgbmVlZCB0byBiZSBzb3J0ZWQuIEl0IGFsc28gcmV0dXJucyBhIHRlbnNvclxuICogYGluZGljZXNgIHRoZSBzYW1lIHNpemUgYXMgdGhlIG51bWJlciBvZiB0aGUgZWxlbWVudHMgaW4gYHhgIGFsb25nIHRoZSBgYXhpc2BcbiAqIGRpbWVuc2lvbi4gSXQgY29udGFpbnMgdGhlIGluZGV4IGluIHRoZSB1bmlxdWUgb3V0cHV0IGB2YWx1ZXNgLlxuICpcbiAqIGBgYGpzXG4gKiAvLyBBIDEtRCB0ZW5zb3JcbiAqIGNvbnN0IGEgPSB0Zi50ZW5zb3IxZChbMSwgMSwgMiwgNCwgNCwgNCwgNywgOCwgOF0pO1xuICogY29uc3Qge3ZhbHVlcywgaW5kaWNlc30gPSB0Zi51bmlxdWUoYSk7XG4gKiB2YWx1ZXMucHJpbnQoKTsgICAvLyBbMSwgMiwgNCwgNywgOCxdXG4gKiBpbmRpY2VzLnByaW50KCk7ICAvLyBbMCwgMCwgMSwgMiwgMiwgMiwgMywgNCwgNF1cbiAqIGBgYFxuICpcbiAqIGBgYGpzXG4gKiAvLyBBIDItRCB0ZW5zb3Igd2l0aCBheGlzPTBcbiAqIC8vXG4gKiAvLyAnYScgaXM6IFtbMSwgMCwgMF0sXG4gKiAvLyAgICAgICAgICBbMSwgMCwgMF0sXG4gKiAvLyAgICAgICAgICBbMiwgMCwgMF1dXG4gKiBjb25zdCBhID0gdGYudGVuc29yMmQoW1sxLCAwLCAwXSwgWzEsIDAsIDBdLCBbMiwgMCwgMF1dKTtcbiAqIGNvbnN0IHt2YWx1ZXMsIGluZGljZXN9ID0gdGYudW5pcXVlKGEsIDApXG4gKiB2YWx1ZXMucHJpbnQoKTsgICAvLyBbWzEsIDAsIDBdLFxuICogICAgICAgICAgICAgICAgICAgLy8gIFsyLCAwLCAwXV1cbiAqIGluZGljZXMucHJpbnQoKTsgIC8vIFswLCAwLCAxXVxuICogYGBgXG4gKlxuICogYGBganNcbiAqIC8vIEEgMi1EIHRlbnNvciB3aXRoIGF4aXM9MVxuICogLy9cbiAqIC8vICdhJyBpczogW1sxLCAwLCAwXSxcbiAqIC8vICAgICAgICAgIFsxLCAwLCAwXSxcbiAqIC8vICAgICAgICAgIFsyLCAwLCAwXV1cbiAqIGNvbnN0IGEgPSB0Zi50ZW5zb3IyZChbWzEsIDAsIDBdLCBbMSwgMCwgMF0sIFsyLCAwLCAwXV0pO1xuICogY29uc3Qge3ZhbHVlcywgaW5kaWNlc30gPSB0Zi51bmlxdWUoYSwgMSlcbiAqIHZhbHVlcy5wcmludCgpOyAgIC8vIFtbMSwgMF0sXG4gKiAgICAgICAgICAgICAgICAgICAvLyAgWzEsIDBdLFxuICogICAgICAgICAgICAgICAgICAgLy8gIFsyLCAwXV1cbiAqIGluZGljZXMucHJpbnQoKTsgIC8vIFswLCAxLCAxXVxuICogYGBgXG4gKiBAcGFyYW0geCBBIHRlbnNvciAoaW50MzIsIHN0cmluZywgYm9vbCkuXG4gKiBAcGFyYW0gYXhpcyBUaGUgYXhpcyBvZiB0aGUgdGVuc29yIHRvIGZpbmQgdGhlIHVuaXF1ZSBlbGVtZW50cy5cbiAqIEByZXR1cm5zIFt1bmlxdWVFbGVtZW50cywgaW5kaWNlc10gKHNlZSBhYm92ZSBmb3IgZGV0YWlscylcbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnT3BlcmF0aW9ucycsIHN1YmhlYWRpbmc6ICdFdmFsdWF0aW9uJ31cbiAqL1xuZnVuY3Rpb24gdW5pcXVlXzxUIGV4dGVuZHMgVGVuc29yPihcbiAgICB4OiBUfFRlbnNvckxpa2UsIGF4aXMgPSAwKToge3ZhbHVlczogVCwgaW5kaWNlczogVGVuc29yMUR9IHtcbiAgY29uc3QgJHggPSBjb252ZXJ0VG9UZW5zb3IoeCwgJ3gnLCAndW5pcXVlJywgJ3N0cmluZ19vcl9udW1lcmljJyk7XG4gIGFzc2VydCgkeC5yYW5rID4gMCwgKCkgPT4gJ1RoZSBpbnB1dCB0ZW5zb3IgbXVzdCBiZSBhdCBsZWFzdCAxRCcpO1xuXG4gIGNvbnN0IGlucHV0czogVW5pcXVlSW5wdXRzID0ge3g6ICR4fTtcbiAgY29uc3QgYXR0cnM6IFVuaXF1ZUF0dHJzID0ge2F4aXN9O1xuICBjb25zdCBbdmFsdWVzLCBpbmRpY2VzXSA9IEVOR0lORS5ydW5LZXJuZWwoXG4gICAgICBVbmlxdWUsIGlucHV0cyBhcyB1bmtub3duIGFzIE5hbWVkVGVuc29yTWFwLFxuICAgICAgYXR0cnMgYXMgdW5rbm93biBhcyBOYW1lZEF0dHJNYXApIGFzIFtULCBUZW5zb3IxRF07XG4gIHJldHVybiB7dmFsdWVzLCBpbmRpY2VzfTtcbn1cblxuZXhwb3J0IGNvbnN0IHVuaXF1ZSA9IC8qIEBfX1BVUkVfXyAqLyBvcCh7dW5pcXVlX30pO1xuIl19