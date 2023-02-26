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
import { GatherV2 } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import { op } from './operation';
/**
 * Gather slices from tensor `x`'s axis `axis` according to `indices`.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3, 4]);
 * const indices = tf.tensor1d([1, 3, 3], 'int32');
 *
 * x.gather(indices).print();
 * ```
 *
 * ```js
 * const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 * const indices = tf.tensor1d([1, 1, 0], 'int32');
 *
 * x.gather(indices).print();
 * ```
 * @param x The input tensor whose slices are to be gathered.
 * @param indices The indices of the values to extract.
 * @param axis The axis over which to select values. Defaults to 0.
 * @param batchDims Optional. The number of batch dimensions. It must be less
 *     than or equal to rank(indices). Defaults to 0.
 *     The output tensor will have shape of
 *     `x.shape[:axis] + indices.shape[batchDims:] + x.shape[axis + 1:]`
 *
 * @doc {heading: 'Tensors', subheading: 'Slicing and Joining'}
 */
function gather_(x, indices, axis = 0, batchDims = 0) {
    const $x = convertToTensor(x, 'x', 'gather');
    const $indices = convertToTensor(indices, 'indices', 'gather', 'int32');
    const inputs = { x: $x, indices: $indices };
    const attrs = { axis, batchDims };
    return ENGINE.runKernel(GatherV2, inputs, attrs);
}
export const gather = /* @__PURE__ */ op({ gather_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZ2F0aGVyLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvZ2F0aGVyLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDakMsT0FBTyxFQUFDLFFBQVEsRUFBZ0MsTUFBTSxpQkFBaUIsQ0FBQztBQUl4RSxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFHbkQsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUUvQjs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQXlCRztBQUNILFNBQVMsT0FBTyxDQUNaLENBQWUsRUFBRSxPQUEwQixFQUFFLElBQUksR0FBRyxDQUFDLEVBQUUsU0FBUyxHQUFHLENBQUM7SUFDdEUsTUFBTSxFQUFFLEdBQUcsZUFBZSxDQUFDLENBQUMsRUFBRSxHQUFHLEVBQUUsUUFBUSxDQUFDLENBQUM7SUFDN0MsTUFBTSxRQUFRLEdBQUcsZUFBZSxDQUFDLE9BQU8sRUFBRSxTQUFTLEVBQUUsUUFBUSxFQUFFLE9BQU8sQ0FBQyxDQUFDO0lBRXhFLE1BQU0sTUFBTSxHQUFtQixFQUFDLENBQUMsRUFBRSxFQUFFLEVBQUUsT0FBTyxFQUFFLFFBQVEsRUFBQyxDQUFDO0lBQzFELE1BQU0sS0FBSyxHQUFrQixFQUFDLElBQUksRUFBRSxTQUFTLEVBQUMsQ0FBQztJQUUvQyxPQUFPLE1BQU0sQ0FBQyxTQUFTLENBQ25CLFFBQVEsRUFBRSxNQUFtQyxFQUM3QyxLQUFnQyxDQUFDLENBQUM7QUFDeEMsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLE1BQU0sR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUMsT0FBTyxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtFTkdJTkV9IGZyb20gJy4uL2VuZ2luZSc7XG5pbXBvcnQge0dhdGhlclYyLCBHYXRoZXJWMkF0dHJzLCBHYXRoZXJWMklucHV0c30gZnJvbSAnLi4va2VybmVsX25hbWVzJztcbmltcG9ydCB7TmFtZWRBdHRyTWFwfSBmcm9tICcuLi9rZXJuZWxfcmVnaXN0cnknO1xuaW1wb3J0IHtUZW5zb3J9IGZyb20gJy4uL3RlbnNvcic7XG5pbXBvcnQge05hbWVkVGVuc29yTWFwfSBmcm9tICcuLi90ZW5zb3JfdHlwZXMnO1xuaW1wb3J0IHtjb252ZXJ0VG9UZW5zb3J9IGZyb20gJy4uL3RlbnNvcl91dGlsX2Vudic7XG5pbXBvcnQge1RlbnNvckxpa2V9IGZyb20gJy4uL3R5cGVzJztcblxuaW1wb3J0IHtvcH0gZnJvbSAnLi9vcGVyYXRpb24nO1xuXG4vKipcbiAqIEdhdGhlciBzbGljZXMgZnJvbSB0ZW5zb3IgYHhgJ3MgYXhpcyBgYXhpc2AgYWNjb3JkaW5nIHRvIGBpbmRpY2VzYC5cbiAqXG4gKiBgYGBqc1xuICogY29uc3QgeCA9IHRmLnRlbnNvcjFkKFsxLCAyLCAzLCA0XSk7XG4gKiBjb25zdCBpbmRpY2VzID0gdGYudGVuc29yMWQoWzEsIDMsIDNdLCAnaW50MzInKTtcbiAqXG4gKiB4LmdhdGhlcihpbmRpY2VzKS5wcmludCgpO1xuICogYGBgXG4gKlxuICogYGBganNcbiAqIGNvbnN0IHggPSB0Zi50ZW5zb3IyZChbMSwgMiwgMywgNF0sIFsyLCAyXSk7XG4gKiBjb25zdCBpbmRpY2VzID0gdGYudGVuc29yMWQoWzEsIDEsIDBdLCAnaW50MzInKTtcbiAqXG4gKiB4LmdhdGhlcihpbmRpY2VzKS5wcmludCgpO1xuICogYGBgXG4gKiBAcGFyYW0geCBUaGUgaW5wdXQgdGVuc29yIHdob3NlIHNsaWNlcyBhcmUgdG8gYmUgZ2F0aGVyZWQuXG4gKiBAcGFyYW0gaW5kaWNlcyBUaGUgaW5kaWNlcyBvZiB0aGUgdmFsdWVzIHRvIGV4dHJhY3QuXG4gKiBAcGFyYW0gYXhpcyBUaGUgYXhpcyBvdmVyIHdoaWNoIHRvIHNlbGVjdCB2YWx1ZXMuIERlZmF1bHRzIHRvIDAuXG4gKiBAcGFyYW0gYmF0Y2hEaW1zIE9wdGlvbmFsLiBUaGUgbnVtYmVyIG9mIGJhdGNoIGRpbWVuc2lvbnMuIEl0IG11c3QgYmUgbGVzc1xuICogICAgIHRoYW4gb3IgZXF1YWwgdG8gcmFuayhpbmRpY2VzKS4gRGVmYXVsdHMgdG8gMC5cbiAqICAgICBUaGUgb3V0cHV0IHRlbnNvciB3aWxsIGhhdmUgc2hhcGUgb2ZcbiAqICAgICBgeC5zaGFwZVs6YXhpc10gKyBpbmRpY2VzLnNoYXBlW2JhdGNoRGltczpdICsgeC5zaGFwZVtheGlzICsgMTpdYFxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdUZW5zb3JzJywgc3ViaGVhZGluZzogJ1NsaWNpbmcgYW5kIEpvaW5pbmcnfVxuICovXG5mdW5jdGlvbiBnYXRoZXJfPFQgZXh0ZW5kcyBUZW5zb3I+KFxuICAgIHg6IFR8VGVuc29yTGlrZSwgaW5kaWNlczogVGVuc29yfFRlbnNvckxpa2UsIGF4aXMgPSAwLCBiYXRjaERpbXMgPSAwKTogVCB7XG4gIGNvbnN0ICR4ID0gY29udmVydFRvVGVuc29yKHgsICd4JywgJ2dhdGhlcicpO1xuICBjb25zdCAkaW5kaWNlcyA9IGNvbnZlcnRUb1RlbnNvcihpbmRpY2VzLCAnaW5kaWNlcycsICdnYXRoZXInLCAnaW50MzInKTtcblxuICBjb25zdCBpbnB1dHM6IEdhdGhlclYySW5wdXRzID0ge3g6ICR4LCBpbmRpY2VzOiAkaW5kaWNlc307XG4gIGNvbnN0IGF0dHJzOiBHYXRoZXJWMkF0dHJzID0ge2F4aXMsIGJhdGNoRGltc307XG5cbiAgcmV0dXJuIEVOR0lORS5ydW5LZXJuZWwoXG4gICAgICBHYXRoZXJWMiwgaW5wdXRzIGFzIHVua25vd24gYXMgTmFtZWRUZW5zb3JNYXAsXG4gICAgICBhdHRycyBhcyB1bmtub3duIGFzIE5hbWVkQXR0ck1hcCk7XG59XG5cbmV4cG9ydCBjb25zdCBnYXRoZXIgPSAvKiBAX19QVVJFX18gKi8gb3Aoe2dhdGhlcl99KTtcbiJdfQ==